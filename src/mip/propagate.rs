//! Node bound propagation (`SolveOptions::node_propagation`): before a node's
//! LP, the rows the engine holds, read at the node's bounds, tighten the
//! bounds of its integer vars, and a contradiction closes the node with no
//! LP solved.
//!
//! For a row `lo ≤ a·x ≤ hi` and an integer var `x_j` in it, the other terms'
//! extreme activities bound `a_j x_j`; the bound is rounded inward to an
//! integer. A continuous var's bounds are never changed, as in presolve
//! (`crate::presolve`, module docs): what a deduced continuous bound would
//! take from the room the row leaves the var could add up beyond a row's
//! tolerance across several vars.
//!
//! Every deduction keeps every point of the node whose integer vars are
//! exactly integral and that meets each row within its contract and each
//! bound within its tolerance (*Tolerances*), so a node closed by a
//! contradiction holds no such point, and pruning it loses nothing the LP
//! could have found.
//!
//! # Tolerances
//!
//! A row of magnitude `m` (`|b| + Σ|a_k|·max(|lo_k|, |hi_k|)` over its finite
//! terms) is held to `row_tolerance(feasibility, m)`; each bound `l` of a var
//! to `bound_tolerance(feasibility, l)`. A deduction widens the row's bound
//! by its contract, by what the other terms can move past their bounds
//! within their tolerances, and by a margin that dominates the round-off of
//! the plain `f64` sums of the activity (`4 (n + 2) u m`, `u` the unit
//! round-off), before dividing by `a_j`. The quotient is then rounded inward
//! only past a further relative `1e-9`, so the round-off of the division can
//! never move an integer across the bound. The deductions are thereby weaker
//! than exact arithmetic allows by a sliver, which costs nothing: they
//! decide only which integers remain.

use crate::solver::{bound_tolerance, row_tolerance};
use crate::{ComparisonOp, CsVec, VarDomain};

/// The unit round-off of `f64`.
const UNIT_ROUNDOFF: f64 = f64::EPSILON / 2.0;

/// The relative slack past which a deduced quotient is rounded inward.
const QUOTIENT_SLACK: f64 = 1e-9;

/// One row as propagation reads it, in user units.
#[derive(Clone, Debug)]
pub(crate) struct PropRow {
    vars: Vec<usize>,
    coeffs: Vec<f64>,
    lo: f64,
    hi: f64,
}

/// The rows propagation reads, with each var's rows.
#[derive(Clone, Debug, Default)]
pub(crate) struct PropModel {
    rows: Vec<PropRow>,
    var_rows: Vec<Vec<u32>>,
}

impl PropModel {
    /// The rows of `constraints` that hold an integer var: the only ones
    /// that can tighten a bound.
    pub(crate) fn new(constraints: &[(CsVec, ComparisonOp, f64)], domains: &[VarDomain]) -> Self {
        let rows: Vec<PropRow> = constraints
            .iter()
            .filter(|(coeffs, _, _)| coeffs.indices().iter().any(|&v| is_int(&domains[v])))
            .map(|(coeffs, op, rhs)| {
                let (lo, hi) = match op {
                    ComparisonOp::Le => (f64::NEG_INFINITY, *rhs),
                    ComparisonOp::Ge => (*rhs, f64::INFINITY),
                    ComparisonOp::Eq => (*rhs, *rhs),
                };
                PropRow {
                    vars: coeffs.indices().to_vec(),
                    coeffs: coeffs.data().to_vec(),
                    lo,
                    hi,
                }
            })
            .collect();
        let mut var_rows = vec![Vec::new(); domains.len()];
        for (r, row) in rows.iter().enumerate() {
            for &v in &row.vars {
                var_rows[v].push(r as u32);
            }
        }
        PropModel { rows, var_rows }
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }
}

fn is_int(d: &VarDomain) -> bool {
    matches!(d, VarDomain::Integer | VarDomain::Boolean)
}

/// `a·x` over `[lo, hi]`, as `(min, max)`.
fn term_range(a: f64, lo: f64, hi: f64) -> (f64, f64) {
    if a > 0.0 {
        (a * lo, a * hi)
    } else {
        (a * hi, a * lo)
    }
}

/// A contradiction: the node holds no point (module docs).
#[derive(Debug, PartialEq)]
pub(crate) struct Contradiction;

/// Tighten the integer vars' entries of `bounds` (the node's bounds of
/// every var, in user units) in place, from a worklist of rows: those of the
/// `seeds` vars, or every row for `None`, then the rows of each var a row
/// tightened, at most `rounds` visits per row on average. Returns the vars
/// whose bounds changed, in ascending order, or the contradiction.
///
/// A node inherits its parent's deductions, so only the rows of the var its
/// branch changed can deduce anything new: `seeds` is that var.
pub(crate) fn propagate(
    model: &PropModel,
    domains: &[VarDomain],
    bounds: &mut [(f64, f64)],
    feasibility: f64,
    seeds: Option<&[usize]>,
    rounds: usize,
) -> Result<Vec<usize>, Contradiction> {
    let mut changed = vec![false; bounds.len()];
    let mut queued = vec![false; model.rows.len()];
    let mut queue = std::collections::VecDeque::new();
    let push = |r: u32, queued: &mut [bool], queue: &mut std::collections::VecDeque<u32>| {
        if !queued[r as usize] {
            queued[r as usize] = true;
            queue.push_back(r);
        }
    };
    match seeds {
        None => (0..model.rows.len() as u32).for_each(|r| push(r, &mut queued, &mut queue)),
        Some(vars) => {
            for &v in vars {
                for &r in &model.var_rows[v] {
                    push(r, &mut queued, &mut queue);
                }
            }
        }
    }
    let mut visits_left = rounds.saturating_mul(model.rows.len().max(1));
    let mut tightened = Vec::new();
    while let Some(r) = queue.pop_front() {
        queued[r as usize] = false;
        if visits_left == 0 {
            break;
        }
        visits_left -= 1;
        tightened.clear();
        sweep(
            &model.rows[r as usize],
            domains,
            bounds,
            feasibility,
            &mut tightened,
        )?;
        // Every row of a tightened var, this one too: its sweep read the
        // other terms at their old bounds.
        for &v in &tightened {
            changed[v] = true;
            for &r2 in &model.var_rows[v] {
                push(r2, &mut queued, &mut queue);
            }
        }
    }
    Ok((0..bounds.len()).filter(|&v| changed[v]).collect())
}

/// One row: tighten each integer var in it from the other terms' extremes,
/// pushing each var it tightened onto `tightened`.
fn sweep(
    row: &PropRow,
    domains: &[VarDomain],
    bounds: &mut [(f64, f64)],
    feasibility: f64,
    tightened: &mut Vec<usize>,
) -> Result<(), Contradiction> {
    let n = row.vars.len();
    // Activity extremes over the finite terms, the number of infinite ones,
    // the magnitude, and the room the bound tolerances give.
    let (mut min_act, mut max_act) = (0.0, 0.0);
    let (mut min_inf, mut max_inf) = (0usize, 0usize);
    let mut magnitude = [row.lo, row.hi]
        .iter()
        .filter(|b| b.is_finite())
        .fold(0.0_f64, |m, b| m.max(b.abs()));
    let mut bound_room = 0.0;
    for (&v, &a) in row.vars.iter().zip(&row.coeffs) {
        let (lo, hi) = bounds[v];
        let (tmin, tmax) = term_range(a, lo, hi);
        if tmin.is_finite() {
            min_act += tmin;
        } else {
            min_inf += 1;
        }
        if tmax.is_finite() {
            max_act += tmax;
        } else {
            max_inf += 1;
        }
        let reach = [lo, hi]
            .iter()
            .filter(|b| b.is_finite())
            .fold(0.0_f64, |m, b| m.max(b.abs()));
        magnitude += a.abs() * reach;
        bound_room +=
            a.abs() * bound_tolerance(feasibility, lo).max(bound_tolerance(feasibility, hi));
    }
    if min_inf > 1 && max_inf > 1 {
        return Ok(());
    }
    let room = row_tolerance(feasibility, magnitude)
        + bound_room
        + 4.0 * (n as f64 + 2.0) * UNIT_ROUNDOFF * magnitude;

    for (&v, &a) in row.vars.iter().zip(&row.coeffs) {
        if !is_int(&domains[v]) || a == 0.0 {
            continue;
        }
        let (lo, hi) = bounds[v];
        let (tmin, tmax) = term_range(a, lo, hi);
        // `a x_v ≤ hi_row + room − (min activity of the others)`.
        let mut new = (lo, hi);
        if row.hi.is_finite() {
            let others = if tmin.is_finite() {
                (min_inf == 0).then_some(min_act - tmin)
            } else {
                (min_inf == 1).then_some(min_act)
            };
            if let Some(others) = others {
                let q = (row.hi + room - others) / a;
                if a > 0.0 {
                    new.1 = new.1.min(round_down(q));
                } else {
                    new.0 = new.0.max(round_up(q));
                }
            }
        }
        // `a x_v ≥ lo_row − room − (max activity of the others)`.
        if row.lo.is_finite() {
            let others = if tmax.is_finite() {
                (max_inf == 0).then_some(max_act - tmax)
            } else {
                (max_inf == 1).then_some(max_act)
            };
            if let Some(others) = others {
                let q = (row.lo - room - others) / a;
                if a > 0.0 {
                    new.0 = new.0.max(round_up(q));
                } else {
                    new.1 = new.1.min(round_down(q));
                }
            }
        }
        if new != (lo, hi) {
            if new.0 > new.1 {
                return Err(Contradiction);
            }
            // The activities above use the old bounds: valid, if looser.
            bounds[v] = new;
            tightened.push(v);
        }
    }
    Ok(())
}

fn round_down(q: f64) -> f64 {
    if q.is_finite() {
        (q + QUOTIENT_SLACK * q.abs().max(1.0)).floor()
    } else {
        q
    }
}

fn round_up(q: f64) -> f64 {
    if q.is_finite() {
        (q - QUOTIENT_SLACK * q.abs().max(1.0)).ceil()
    } else {
        q
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(terms: &[(usize, f64)], lo: f64, hi: f64) -> PropRow {
        PropRow {
            vars: terms.iter().map(|t| t.0).collect(),
            coeffs: terms.iter().map(|t| t.1).collect(),
            lo,
            hi,
        }
    }

    fn propagate(
        rows: &[PropRow],
        domains: &[VarDomain],
        bounds: &mut [(f64, f64)],
        feasibility: f64,
        rounds: usize,
    ) -> Result<Vec<usize>, Contradiction> {
        let mut var_rows = vec![Vec::new(); domains.len()];
        for (r, row) in rows.iter().enumerate() {
            for &v in &row.vars {
                var_rows[v].push(r as u32);
            }
        }
        let model = PropModel {
            rows: rows.to_vec(),
            var_rows,
        };
        super::propagate(&model, domains, bounds, feasibility, None, rounds)
    }

    #[test]
    fn a_knapsack_row_caps_its_integers() {
        // 3x + 5y ≤ 11, x, y integer in [0, 10]: x ≤ 3, y ≤ 2.
        let rows = [row(&[(0, 3.0), (1, 5.0)], f64::NEG_INFINITY, 11.0)];
        let domains = [VarDomain::Integer, VarDomain::Integer];
        let mut bounds = [(0.0, 10.0), (0.0, 10.0)];
        let changed = propagate(&rows, &domains, &mut bounds, 1e-7, 5).unwrap();
        assert_eq!(changed, vec![0, 1]);
        assert_eq!(bounds, [(0.0, 3.0), (0.0, 2.0)]);
    }

    #[test]
    fn a_continuous_var_keeps_its_bounds_but_bounds_the_others() {
        // x + z ≥ 7, x integer in [0, 10], z continuous in [0, 2.5]:
        // x ≥ 4.5, so 5; z unchanged.
        let rows = [row(&[(0, 1.0), (1, 1.0)], 7.0, f64::INFINITY)];
        let domains = [VarDomain::Integer, VarDomain::Real];
        let mut bounds = [(0.0, 10.0), (0.0, 2.5)];
        propagate(&rows, &domains, &mut bounds, 1e-7, 5).unwrap();
        assert_eq!(bounds, [(5.0, 10.0), (0.0, 2.5)]);
    }

    #[test]
    fn an_integer_point_on_the_boundary_is_kept() {
        // x ≤ 3 − 1e-8: x = 3 exceeds it by 1e-8, within the contract
        // (1e-7), so 3 stays.
        let rows = [row(&[(0, 1.0)], f64::NEG_INFINITY, 3.0 - 1e-8)];
        let domains = [VarDomain::Integer];
        let mut bounds = [(0.0, 10.0)];
        propagate(&rows, &domains, &mut bounds, 1e-7, 5).unwrap();
        assert_eq!(bounds, [(0.0, 3.0)]);
    }

    #[test]
    fn crossing_bounds_are_a_contradiction() {
        // x + y ≥ 3 with binaries.
        let rows = [row(&[(0, 1.0), (1, 1.0)], 3.0, f64::INFINITY)];
        let domains = [VarDomain::Boolean, VarDomain::Boolean];
        let mut bounds = [(0.0, 1.0), (0.0, 1.0)];
        assert_eq!(
            propagate(&rows, &domains, &mut bounds, 1e-7, 5),
            Err(Contradiction)
        );
    }

    #[test]
    fn rounds_chain_deductions() {
        // x − y ≤ −1 with y ≤ 2 caps x at 1; x + w ≥ 3 with w ≤ 1 lifts x
        // to 2: a contradiction.
        let rows = [
            row(&[(0, 1.0), (1, -1.0)], f64::NEG_INFINITY, -1.0),
            row(&[(0, 1.0), (2, 1.0)], 3.0, f64::INFINITY),
        ];
        let domains = [VarDomain::Integer, VarDomain::Integer, VarDomain::Integer];
        let mut bounds = [(0.0, 10.0), (0.0, 2.0), (0.0, 1.0)];
        assert_eq!(
            propagate(&rows, &domains, &mut bounds, 1e-7, 5),
            Err(Contradiction)
        );
    }

    #[test]
    fn seeds_limit_the_rows_visited() {
        // Two independent knapsack rows; seeding with x visits only x's row.
        let rows = vec![
            row(&[(0, 3.0)], f64::NEG_INFINITY, 11.0),
            row(&[(1, 5.0)], f64::NEG_INFINITY, 11.0),
        ];
        let domains = [VarDomain::Integer, VarDomain::Integer];
        let model = PropModel {
            rows,
            var_rows: vec![vec![0], vec![1]],
        };
        let mut bounds = [(0.0, 10.0), (0.0, 10.0)];
        let changed = super::propagate(&model, &domains, &mut bounds, 1e-7, Some(&[0]), 5).unwrap();
        assert_eq!(changed, vec![0]);
        assert_eq!(bounds, [(0.0, 3.0), (0.0, 10.0)]);
    }
}
