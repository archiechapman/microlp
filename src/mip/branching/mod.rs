//! Branch variable selection.

use super::params::{self, PSEUDOCOST_INIT_EPS, SCORE_EPS};
use super::MipState;
use crate::solver::Solver;
use crate::{Error, StopReason, VarDomain};

fn fractionality(val: f64) -> f64 {
    (val - val.round()).abs()
}

fn is_int_domain(d: &VarDomain) -> bool {
    matches!(d, VarDomain::Integer | VarDomain::Boolean)
}

/// Whether an integer var with bounds `lo..hi` (integral) is fixed: then it
/// cannot move, and the engine holds it exactly at its bound.
fn fixed(lo: f64, hi: f64) -> bool {
    hi - lo < 0.5
}

/// True if every integer-domained structural var the search can branch on
/// is within `int_tol` of an integer.
pub(crate) fn is_integral(solver: &Solver, domains: &[VarDomain], int_tol: f64) -> bool {
    domains
        .iter()
        .enumerate()
        .filter(|&(v, d)| {
            let (lo, hi) = solver.get_var_bounds(v);
            is_int_domain(d) && !fixed(lo, hi)
        })
        .all(|(v, _)| fractionality(solver.get_value(v)) <= int_tol)
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

    /// The observations folded in for `var`, as `(down, up)`.
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
        // node verbatim. `is_integral` leaves it out too.
        let (lo, hi) = solver.get_var_bounds(v);
        if fixed(lo, hi) {
            continue;
        }
        let val = solver.get_value(v);
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

/// The split point `k` of an integer var at `val` with bounds `lo..=hi`: its
/// children are `x ≤ k` and `x ≥ k + 1`. A value within `int_tol` of an
/// integer takes that integer, not its floor: `floor(−8e-16) = −1` would make
/// the up child `(max(0, lo), hi)` reproduce its parent and the search descend
/// forever, on the rejected-candidate re-branch at `int_tol = 0`. `k` is then
/// clamped into `[lo, hi − 1]`, so both children strictly tighten `[lo, hi]`
/// whenever `hi − lo ≥ 1` (integral bounds, as every branchable var has).
pub(crate) fn split_point(val: f64, (lo, hi): (f64, f64), int_tol: f64) -> f64 {
    let near = val.round();
    let k = if (val - near).abs() <= int_tol {
        near
    } else {
        val.floor()
    };
    k.clamp(lo, (hi - 1.0).max(lo))
}

/// What reliability branching chose at a node.
pub(crate) enum Choice {
    /// Branch on this var.
    Var(usize),
    /// No integer var is fractional.
    None,
    /// The deadline struck while the node's LP was being restored after the
    /// probes: the node's LP must be solved again before anything is read
    /// from the solver.
    Interrupted,
}

/// What a strong-branching probe found.
enum Probe {
    /// The child's LP optimum, in internal (minimize) space.
    Bound(f64),
    /// The child's LP is infeasible.
    Infeasible,
    /// The solve failed; it says nothing, and the next probe starts from
    /// the node's basis.
    Failed,
}

/// Reliability branching (`SolveOptions::strong_branch_reliability`): the
/// pseudocost product rule of [`choose_branch_var`], after probing the best
/// `params::SB_MAX_CANDIDATES` candidates whose pseudocosts rest on fewer
/// than `reliability` observations in either direction. A probe solves the
/// child's LP from the node's basis ([`probe`]) and records its objective
/// change as an observation, as a solved child does. A candidate with a
/// child shown infeasible is chosen at once. The node's bounds and basis are
/// restored, and its LP re-solved to verify them, before the caller reads
/// the solver, so it reads the node's optimum, as without probes.
pub(crate) fn select_branch_var(
    state: &mut MipState,
    domains: &[VarDomain],
    int_tol: f64,
) -> Result<Choice, Error> {
    let reliability = state.options.strong_branch_reliability;
    let plain = |state: &MipState| {
        choose_branch_var(&state.solver, domains, int_tol, &state.pseudocosts)
            .map_or(Choice::None, Choice::Var)
    };
    if reliability == 0 || state.sb_budget == 0 {
        return Ok(plain(state));
    }
    let score = |state: &MipState, v: usize| {
        let val = state.solver.get_value(v);
        let f_down = val - val.floor();
        (state.pseudocosts.estimate(v, false) * f_down).max(SCORE_EPS)
            * (state.pseudocosts.estimate(v, true) * (1.0 - f_down)).max(SCORE_EPS)
    };
    let mut candidates: Vec<(usize, f64)> = domains
        .iter()
        .enumerate()
        .filter(|&(v, d)| {
            let (lo, hi) = state.solver.get_var_bounds(v);
            is_int_domain(d) && !fixed(lo, hi) && fractionality(state.solver.get_value(v)) > int_tol
        })
        .map(|(v, _)| (v, score(state, v)))
        .collect();
    candidates.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    let unreliable: Vec<usize> = candidates
        .iter()
        .map(|&(v, _)| v)
        .filter(|&v| {
            let (down, up) = state.pseudocosts.observations(v);
            down.min(up) < reliability
        })
        .take(params::SB_MAX_CANDIDATES)
        .collect();
    if unreliable.is_empty() {
        return Ok(plain(state));
    }

    let node_obj = state.solver.cur_obj_val;
    let basis = state.solver.snapshot_basis();
    // The node's values: probes move the solver off them.
    let node_values: Vec<f64> = (0..state.solver.num_vars)
        .map(|v| state.solver.get_value(v))
        .collect();
    // Each candidate's score at this node: the product rule on the probes'
    // own objective changes for a candidate probed here, on its pseudocosts
    // for a reliable one. An unreliable candidate left unprobed (beyond the
    // limit or the budget, or whose probes said nothing) has only its
    // initial estimate, which would outrank measured ones arbitrarily, so it
    // is chosen only when no other candidate is scored.
    let mut scored: Vec<(usize, f64)> = Vec::new();
    let mut chosen = None;
    let mut reload = false;
    for &(v, pseudo) in &candidates {
        let (down_n, up_n) = state.pseudocosts.observations(v);
        if down_n.min(up_n) >= reliability {
            scored.push((v, pseudo));
            continue;
        }
        if !unreliable.contains(&v) || state.sb_budget == 0 {
            continue;
        }
        if reload {
            // The last probe failed or proved its child infeasible: start
            // the next one from the node's basis.
            if state.solver.load_basis(&basis).is_err() {
                break;
            }
            reload = false;
        }
        let val = node_values[v];
        let (lo, hi) = state.solver.get_var_bounds(v);
        let k = split_point(val, (lo, hi), int_tol);
        let f_down = (val - k).clamp(0.0, 1.0);
        let down = probe(state, v, (lo, k), (lo, hi));
        reload |= !matches!(down, Some(Probe::Bound(_)) | None);
        let up = probe(state, v, (k + 1.0, hi), (lo, hi));
        reload |= !matches!(up, Some(Probe::Bound(_)) | None);
        if matches!(down, Some(Probe::Infeasible)) || matches!(up, Some(Probe::Infeasible)) {
            debug!("strong branching: var {v} has an infeasible child; branching on it");
            chosen = Some(v);
            break;
        }
        let mut gains = [None, None];
        for (side, (result, frac)) in [(down, f_down), (up, 1.0 - f_down)].into_iter().enumerate() {
            if let Some(Probe::Bound(obj)) = result {
                let gain = (obj - node_obj).max(0.0);
                state
                    .pseudocosts
                    .record(v, side == 1, gain / frac.max(params::BRANCH_FRAC_GUARD));
                gains[side] = Some(gain);
            }
        }
        if let [Some(down_gain), Some(up_gain)] = gains {
            scored.push((v, down_gain.max(SCORE_EPS) * up_gain.max(SCORE_EPS)));
        }
    }

    // Back to the node's optimum: its basis under its bounds, re-solved in
    // case the load had to repair it, which otherwise only verifies it.
    state.solver.load_basis(&basis)?;
    if state.solver.reoptimize()? == StopReason::Limit {
        return Ok(Choice::Interrupted);
    }
    if let Some(v) = chosen {
        return Ok(Choice::Var(v));
    }
    let best = scored
        .iter()
        .max_by(|a, b| a.1.total_cmp(&b.1).then(b.0.cmp(&a.0)))
        .map(|&(v, _)| v);
    Ok(best.map_or_else(|| plain(state), Choice::Var))
}

/// One strong-branching probe: the LP of the child that gives `var` the
/// bounds `child`, warm from wherever the solver stands (the node's optimum,
/// or the last probe's end), within `params::SB_MAX_ITERS_PER_LP`
/// iterations; then `var` back to its node bounds `node`. `None` when the
/// probe says nothing: it reached the cap or the deadline, or failed (the
/// search's own solve of that child, should it come, has the recovery ladder
/// of a node LP).
fn probe(state: &mut MipState, var: usize, child: (f64, f64), node: (f64, f64)) -> Option<Probe> {
    state.stats.strong_branch_lps += 1;
    state.sb_budget = state.sb_budget.saturating_sub(1);
    state
        .solver
        .set_var_bounds(var, child.0, child.1)
        .expect("a child's bounds cannot cross");
    state
        .solver
        .set_iteration_limit(Some(params::SB_MAX_ITERS_PER_LP));
    let outcome = state.solver.reoptimize();
    state.solver.set_iteration_limit(None);
    let result = match outcome {
        Ok(StopReason::Finished) => Some(Probe::Bound(state.solver.cur_obj_val)),
        Err(Error::Infeasible) => Some(Probe::Infeasible),
        Ok(StopReason::Limit) => None,
        Err(_) => Some(Probe::Failed),
    };
    state
        .solver
        .set_var_bounds(var, node.0, node.1)
        .expect("a node's bounds cannot cross");
    result
}

#[cfg(test)]
mod tests;
