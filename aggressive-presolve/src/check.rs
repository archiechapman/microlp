//! The answer check: a point against every row, bound and integrality of a
//! model as written, at microlp's contract (its ARCHITECTURE §7.1 and
//! `row_tolerance` in `src/solver/tolerances.rs`): a row of magnitude
//! `m = |rhs| + Σ|a_j x_j|` may miss its range by `max(feas, 1e-14·m)`, a
//! bound by the same rule at the bound's magnitude, and an integer var may lie
//! `int_tol` off an integer.

use crate::model::Model;
use crate::sum::Sum;

const ROUNDOFF_FLOOR: f64 = 1e-14;

/// What a check found wrong.
#[derive(Clone, Debug, PartialEq)]
pub enum Violation {
    /// Row `row` (original index) misses its range by `amount` > `tol`.
    Row { row: usize, amount: f64, tol: f64 },
    /// Var `var` leaves its bounds by `amount` > `tol`.
    Bound { var: usize, amount: f64, tol: f64 },
    /// Integer var `var` is `amount` > `int_tol` off an integer.
    Integrality { var: usize, amount: f64 },
}

impl std::fmt::Display for Violation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Violation::Row { row, amount, tol } => {
                write!(f, "row {} violated by {:.3e} (tol {:.3e})", row, amount, tol)
            }
            Violation::Bound { var, amount, tol } => {
                write!(f, "var {} out of bounds by {:.3e} (tol {:.3e})", var, amount, tol)
            }
            Violation::Integrality { var, amount } => {
                write!(f, "integer var {} off by {:.3e}", var, amount)
            }
        }
    }
}

fn tolerance(feas: f64, magnitude: f64) -> f64 {
    feas.max(ROUNDOFF_FLOOR * magnitude)
}

/// Checks `x` against every row, bound and integrality of `model`. An empty
/// result means the point meets the contract.
pub fn check(model: &Model, x: &[f64], feas: f64, int_tol: f64) -> Vec<Violation> {
    let mut out = Vec::new();
    if x.len() != model.vars.len() {
        out.push(Violation::Bound {
            var: x.len().min(model.vars.len()),
            amount: f64::INFINITY,
            tol: 0.0,
        });
        return out;
    }
    for (j, v) in model.vars.iter().enumerate() {
        let xv = x[j];
        if !xv.is_finite() {
            out.push(Violation::Bound {
                var: j,
                amount: f64::INFINITY,
                tol: 0.0,
            });
            continue;
        }
        if v.lo.is_finite() {
            let tol = tolerance(feas, v.lo.abs());
            if xv < v.lo - tol {
                out.push(Violation::Bound {
                    var: j,
                    amount: v.lo - xv,
                    tol,
                });
            }
        }
        if v.hi.is_finite() {
            let tol = tolerance(feas, v.hi.abs());
            if xv > v.hi + tol {
                out.push(Violation::Bound {
                    var: j,
                    amount: xv - v.hi,
                    tol,
                });
            }
        }
        if v.is_integer() {
            let off = (xv - xv.round()).abs();
            if off > int_tol {
                out.push(Violation::Integrality { var: j, amount: off });
            }
        }
    }
    for (i, r) in model.rows.iter().enumerate() {
        let mut s = Sum::new();
        let mut mag = 0.0;
        for &(j, a) in &r.terms {
            s.add_prod(a, x[j]);
            mag += (a * x[j]).abs();
        }
        let act = s.value();
        if !act.is_finite() {
            out.push(Violation::Row {
                row: i,
                amount: f64::INFINITY,
                tol: 0.0,
            });
            continue;
        }
        if r.lo.is_finite() {
            let tol = tolerance(feas, r.lo.abs() + mag);
            if act < r.lo - tol {
                out.push(Violation::Row {
                    row: i,
                    amount: r.lo - act,
                    tol,
                });
                continue;
            }
        }
        if r.hi.is_finite() {
            let tol = tolerance(feas, r.hi.abs() + mag);
            if act > r.hi + tol {
                out.push(Violation::Row {
                    row: i,
                    amount: act - r.hi,
                    tol,
                });
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{Direction, Sense};

    #[test]
    fn flags_rows_bounds_and_integrality() {
        let mut m = Model::new(Direction::Minimize);
        let x = m.integer(1.0, 0.0, 3.0);
        let y = m.continuous(1.0, 0.0, 1.0);
        m.add_row("", [(x, 1.0), (y, 1.0)], Sense::Le, 2.0);
        assert!(check(&m, &[1.0, 1.0], 1e-7, 1e-6).is_empty());
        assert!(check(&m, &[1.0, 1.0 + 5e-8], 1e-7, 1e-6).is_empty());
        let v = check(&m, &[1.5, 1.0], 1e-7, 1e-6);
        assert!(v.iter().any(|v| matches!(v, Violation::Integrality { var: 0, .. })));
        assert!(v.iter().any(|v| matches!(v, Violation::Row { row: 0, .. })));
        let v = check(&m, &[1.0, 1.1], 1e-7, 1e-6);
        assert!(v.iter().any(|v| matches!(v, Violation::Bound { var: 1, .. })));
    }
}
