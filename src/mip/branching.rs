//! Branch variable selection.

use super::params::{self, PSEUDOCOST_INIT_EPS, SCORE_EPS};
use super::MipState;
use crate::solver::{Basis, Solver};
use crate::{Error, StopReason, VarDomain};

fn fractionality(val: f64) -> f64 {
    (val - val.round()).abs()
}

fn is_int_domain(d: &VarDomain) -> bool {
    matches!(d, VarDomain::Integer | VarDomain::Boolean)
}

/// True if every integer-domained structural var is within `int_tol` of an integer.
pub(crate) fn is_integral(solver: &Solver, domains: &[VarDomain], int_tol: f64) -> bool {
    domains
        .iter()
        .enumerate()
        .filter(|(_, d)| is_int_domain(d))
        .all(|(v, _)| fractionality(*solver.get_value(v)) <= int_tol)
}

/// Per-variable average objective degradation per unit of fractionality, per
/// branching direction. Falls back to |obj coeff| before any observation.
#[derive(Clone, Debug)]
pub(crate) struct PseudoCosts {
    up_sum: Vec<f64>,
    up_n: Vec<u32>,
    down_sum: Vec<f64>,
    down_n: Vec<u32>,
    init: Vec<f64>,
}

impl PseudoCosts {
    pub(crate) fn new(obj_coeffs: &[f64], num_vars: usize) -> Self {
        let init = (0..num_vars)
            .map(|v| obj_coeffs.get(v).copied().unwrap_or(0.0).abs() + PSEUDOCOST_INIT_EPS)
            .collect();
        Self {
            up_sum: vec![0.0; num_vars],
            up_n: vec![0; num_vars],
            down_sum: vec![0.0; num_vars],
            down_n: vec![0; num_vars],
            init,
        }
    }

    pub(crate) fn record(&mut self, var: usize, up: bool, degradation_per_unit: f64) {
        if up {
            self.up_sum[var] += degradation_per_unit;
            self.up_n[var] += 1;
        } else {
            self.down_sum[var] += degradation_per_unit;
            self.down_n[var] += 1;
        }
    }

    pub(crate) fn estimate(&self, var: usize, up: bool) -> f64 {
        let (sum, n) = if up {
            (self.up_sum[var], self.up_n[var])
        } else {
            (self.down_sum[var], self.down_n[var])
        };
        if n > 0 {
            sum / n as f64
        } else {
            self.init[var]
        }
    }

    /// Observations folded in for `var`, as `(down, up)`. Reliability branching
    /// strong-branches a variable while either side is below its threshold.
    pub(crate) fn observations(&self, var: usize) -> (u32, u32) {
        (self.down_n[var], self.up_n[var])
    }

    /// Total number of `record` calls folded in so far, across every variable
    /// and direction. A cheap summary for diagnostics (e.g. `Debug` impls)
    /// that avoids printing the full per-variable vectors.
    pub(crate) fn observation_count(&self) -> u64 {
        self.up_n.iter().map(|&n| u64::from(n)).sum::<u64>()
            + self.down_n.iter().map(|&n| u64::from(n)).sum::<u64>()
    }
}

/// Pseudocost product rule: pick the fractional int var maximizing
/// max(est_down·f_down, ε) · max(est_up·f_up, ε).
pub(crate) fn choose_branch_var(
    solver: &Solver,
    domains: &[VarDomain],
    int_tol: f64,
    pc: &PseudoCosts,
) -> Option<usize> {
    let mut best: Option<(usize, f64)> = None;
    for (v, d) in domains.iter().enumerate() {
        if !is_int_domain(d) {
            continue;
        }
        // A fixed var (lo == hi; integral bounds make hi - lo < 0.5 the
        // robust test) cannot move: branching on it reproduces the parent
        // node verbatim. Its LP value can still carry sub-EPS basic noise
        // (e.g. -5e-16 on a var fixed to 0), including when this function is
        // called with int_tol = 0 after a rounded candidate is rejected.
        let (lo, hi) = solver.get_var_bounds(v);
        if hi - lo < 0.5 {
            continue;
        }
        let val = *solver.get_value(v);
        if fractionality(val) <= int_tol {
            continue;
        }
        let f_down = val - val.floor();
        let f_up = 1.0 - f_down;
        let score = (pc.estimate(v, false) * f_down).max(SCORE_EPS)
            * (pc.estimate(v, true) * f_up).max(SCORE_EPS);
        if best.is_none_or(|(_, s)| score > s) {
            best = Some((v, score));
        }
    }
    best.map(|(v, _)| v)
}

/// The split point used for `var` at the current LP value: children are
/// `[lo, k]` and `[k + 1, hi]`. Mirrors `mip::branch`, so a strong-branching probe
/// solves exactly the child the search would create.
pub(crate) fn split_point(solver: &Solver, var: usize, int_tol: f64) -> f64 {
    let val = *solver.get_value(var);
    let (lo, hi) = solver.get_var_bounds(var);
    let near = val.round();
    let k = if (val - near).abs() <= int_tol {
        near
    } else {
        val.floor()
    };
    k.clamp(lo, (hi - 1.0).max(lo))
}

/// One strong-branching probe: solve the child that fixes `var` to `[lo, hi]`, capped at
/// `params::SB_MAX_ITERS_PER_LP` simplex iterations, then restore the node's bounds and
/// basis. `Ok(None)` means the probe was cut short (iteration cap or deadline) and says
/// nothing about the child.
fn probe(
    state: &mut MipState,
    var: usize,
    (lo, hi): (f64, f64),
    basis: &Basis,
) -> Result<Option<ProbeResult>, Error> {
    let bounds = state.solver.get_var_bounds(var);
    state.stats.strong_branch_lps += 1;
    state.sb_budget = state.sb_budget.saturating_sub(1);
    state
        .solver
        .set_var_bounds(var, lo, hi)
        .expect("probe bounds cannot cross");
    state.solver.set_iteration_limit(Some(params::SB_MAX_ITERS_PER_LP));
    let outcome = state.solver.reoptimize();
    state.solver.set_iteration_limit(None);
    let result = match outcome {
        Ok(StopReason::Finished) => Some(ProbeResult::Bound(state.solver.cur_obj_val)),
        Ok(StopReason::Limit) => None,
        Err(Error::Infeasible) => Some(ProbeResult::Infeasible),
        Err(error) => return Err(error),
    };

    state
        .solver
        .set_var_bounds(var, bounds.0, bounds.1)
        .expect("node bounds cannot cross");
    if state.solver.load_basis(basis).is_err() {
        let slack = state.solver.slack_basis();
        state
            .solver
            .load_basis(&slack)
            .map_err(|e| Error::InternalError(format!("slack basis load failed: {}", e)))?;
        state.solver.reoptimize()?;
    }
    Ok(result)
}

enum ProbeResult {
    Bound(f64),
    Infeasible,
}

/// Reliability branching: the pseudocost product rule, but a candidate whose pseudocosts
/// rest on fewer than `SolveOptions::strong_branch_reliability` observations per side is
/// probed first (see [`probe`]), and the probe's objective degradation is folded into its
/// pseudocosts. Probing is limited to the best `params::SB_MAX_CANDIDATES` candidates per
/// node and a global budget, so it fades out as the tree grows — which is the point:
/// pseudocosts are unreliable exactly at the top, where the branching matters most.
///
/// A candidate with one infeasible side is taken immediately (that child is closed before
/// it is ever solved). `Ok(None)` when nothing can be branched on, including when a
/// candidate's two sides are both infeasible, which closes the node.
pub(crate) fn select_branch_var(
    state: &mut MipState,
    domains: &[VarDomain],
    int_tol: f64,
) -> Result<Option<usize>, Error> {
    let mut candidates: Vec<usize> = Vec::new();
    for (v, d) in domains.iter().enumerate() {
        if !is_int_domain(d) {
            continue;
        }
        let (lo, hi) = state.solver.get_var_bounds(v);
        if hi - lo < 0.5 {
            continue;
        }
        if fractionality(*state.solver.get_value(v)) > int_tol {
            candidates.push(v);
        }
    }
    let reliability = state.options.strong_branch_reliability;
    if candidates.is_empty() || reliability == 0 || state.sb_budget == 0 {
        return Ok(choose_branch_var(&state.solver, domains, int_tol, &state.pseudocosts));
    }

    let score = |state: &MipState, v: usize| {
        let val = *state.solver.get_value(v);
        let f_down = val - val.floor();
        let f_up = 1.0 - f_down;
        (state.pseudocosts.estimate(v, false) * f_down).max(SCORE_EPS)
            * (state.pseudocosts.estimate(v, true) * f_up).max(SCORE_EPS)
    };
    candidates.sort_by(|&a, &b| {
        score(state, b)
            .total_cmp(&score(state, a))
            .then(a.cmp(&b))
    });

    let node_obj = state.solver.cur_obj_val;
    let basis = state.solver.snapshot_basis();
    let mut probed = 0;
    for &v in &candidates {
        if probed >= params::SB_MAX_CANDIDATES || state.sb_budget == 0 {
            break;
        }
        let (n_down, n_up) = state.pseudocosts.observations(v);
        if n_down.min(n_up) >= reliability {
            continue;
        }
        probed += 1;

        let val = *state.solver.get_value(v);
        let (lo, hi) = state.solver.get_var_bounds(v);
        let k = split_point(&state.solver, v, int_tol);
        let f_down = (val - k).clamp(0.0, 1.0);
        let f_up = 1.0 - f_down;

        let down = probe(state, v, (lo, k), &basis)?;
        let up = probe(state, v, (k + 1.0, hi), &basis)?;
        // A probe that proves a child infeasible makes this variable the obvious choice:
        // that subtree dies as soon as the search visits it. The probe only *chooses*
        // here — it never closes the node itself, so the node budget still accounts for
        // every node the search concludes anything from.
        if matches!(down, Some(ProbeResult::Infeasible))
            || matches!(up, Some(ProbeResult::Infeasible))
        {
            debug!("strong branching: var {} has an infeasible child; branching on it", v);
            return Ok(Some(v));
        }
        for (result, up_side, frac) in [(down, false, f_down), (up, true, f_up)] {
            if let Some(ProbeResult::Bound(obj)) = result {
                state.pseudocosts.record(
                    v,
                    up_side,
                    (obj - node_obj).max(0.0) / frac.max(params::BRANCH_FRAC_GUARD),
                );
            }
        }
    }

    Ok(choose_branch_var(&state.solver, domains, int_tol, &state.pseudocosts))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::Solver;
    use crate::{ComparisonOp, VarDomain};

    fn to_sparse(values: &[f64]) -> crate::CsVec {
        let mut indices = vec![];
        let mut data = vec![];
        for (i, &v) in values.iter().enumerate() {
            if v != 0.0 {
                indices.push(i);
                data.push(v);
            }
        }
        crate::CsVec::new(values.len(), indices, data)
    }

    #[test]
    fn most_fractional_var_is_chosen() {
        // minimize -x - y s.t. x + 2y <= 3.2, x <= 1.9; x,y integer-domained.
        // LP optimum: x = 1.9, y = 0.65 → fractional parts 0.9 and 0.65;
        // most-fractional metric |v - round(v)|: x → 0.1, y → 0.35 → picks y (idx 1).
        // With a fresh PseudoCosts (uniform init, no recorded data), the product score
        // est_down·f_down · est_up·f_up is maximized by the most-fractional var too:
        // var 1: 0.65·0.35 ≈ 0.23 beats var 0: 0.9·0.1 = 0.09.
        let mut solver = Solver::try_new(
            &[-1.0, -1.0],
            &[0.0, 0.0],
            &[1.9, 10.0],
            &[(to_sparse(&[1.0, 2.0]), ComparisonOp::Le, 3.2)],
            &[VarDomain::Integer, VarDomain::Integer],
            None,
        )
        .unwrap();
        solver.initial_solve().unwrap();
        assert!(!is_integral(
            &solver,
            solver.orig_var_domains.clone().as_slice(),
            1e-6
        ));
        let pc = PseudoCosts::new(&[-1.0, -1.0], 2);
        assert_eq!(
            choose_branch_var(
                &solver,
                solver.orig_var_domains.clone().as_slice(),
                1e-6,
                &pc
            ),
            Some(1)
        );
    }

    #[test]
    fn integral_solution_yields_no_branch_var() {
        // minimize x s.t. x >= 2, x integer in [0, 10] → LP optimum x = 2 (integral).
        let mut solver = Solver::try_new(
            &[1.0],
            &[0.0],
            &[10.0],
            &[(to_sparse(&[2.0]), ComparisonOp::Ge, 4.0)],
            &[VarDomain::Integer],
            None,
        )
        .unwrap();
        solver.initial_solve().unwrap();
        assert!(is_integral(
            &solver,
            solver.orig_var_domains.clone().as_slice(),
            1e-6
        ));
        let pc = PseudoCosts::new(&[1.0], 1);
        assert_eq!(
            choose_branch_var(
                &solver,
                solver.orig_var_domains.clone().as_slice(),
                1e-6,
                &pc
            ),
            None
        );
    }

    #[test]
    fn pseudocosts_average_and_fall_back_to_init() {
        // 2 vars, obj coeffs 3 and 0 → init estimates 3+1e-6 and 1e-6... clamped by new().
        let mut pc = PseudoCosts::new(&[3.0, 0.0], 2);
        assert!((pc.estimate(0, true) - 3.0).abs() < 1e-3);
        pc.record(0, true, 10.0);
        pc.record(0, true, 20.0);
        assert!((pc.estimate(0, true) - 15.0).abs() < 1e-9); // average of observations
        assert!((pc.estimate(0, false) - 3.0).abs() < 1e-3); // down side still init
    }

    #[test]
    fn pseudocost_selection_prefers_high_degradation_var() {
        // Two fractional int vars; var 1 has recorded huge degradations → must be chosen.
        let mut solver = Solver::try_new(
            &[-1.0, -1.0],
            &[0.0, 0.0],
            &[1.5, 10.0],
            &[(to_sparse(&[1.0, 2.0]), ComparisonOp::Le, 3.0)],
            &[VarDomain::Integer, VarDomain::Integer],
            None,
        )
        .unwrap();
        solver.initial_solve().unwrap();
        // LP optimum: x=1.5, y=0.75 → both fractional.
        let mut pc = PseudoCosts::new(&[-1.0, -1.0], 2);
        pc.record(1, true, 100.0);
        pc.record(1, false, 100.0);
        let domains = solver.orig_var_domains.clone();
        assert_eq!(choose_branch_var(&solver, &domains, 1e-6, &pc), Some(1));
    }
}
