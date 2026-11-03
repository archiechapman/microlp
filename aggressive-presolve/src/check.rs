//! The answer check: a point against every row and bound of the model as
//! written, at microlp's contract (`row_tolerance` in microlp's
//! `src/solver/tolerances.rs`, ARCHITECTURE §7.1), rows summed in
//! double-double.

use crate::model::Model;
use crate::sum::{two_prod, Dd};

/// The contract the check holds a point to.
#[derive(Clone, Copy, Debug)]
pub struct CheckOptions {
    /// Absolute feasibility tolerance (microlp's default `1e-7`).
    pub feasibility: f64,
    /// Integrality tolerance (microlp's default `1e-6`).
    pub int_tol: f64,
}

impl Default for CheckOptions {
    fn default() -> Self {
        CheckOptions {
            feasibility: 1e-7,
            int_tol: 1e-6,
        }
    }
}

/// One failed test.
#[derive(Clone, Debug, PartialEq)]
pub enum Violation {
    /// Row `row` misses its side by `by`, beyond its tolerance `tol`.
    Row { row: usize, by: f64, tol: f64 },
    /// Var `var` lies outside its bounds by `by`.
    Bound { var: usize, by: f64, tol: f64 },
    /// Integer var `var` lies `by` from the nearest integer.
    Integrality { var: usize, by: f64 },
    /// A value or an activity is not finite.
    NotFinite {
        var: Option<usize>,
        row: Option<usize>,
    },
}

/// What the check found.
#[derive(Clone, Debug, Default)]
pub struct CheckReport {
    pub violations: Vec<Violation>,
    /// Largest row violation relative to its tolerance (`<= 1` passes).
    pub worst_row_ratio: f64,
}

impl CheckReport {
    pub fn ok(&self) -> bool {
        self.violations.is_empty()
    }
}

/// `max(feas, 1e-14·m)`.
pub fn row_tolerance(feas: f64, m: f64) -> f64 {
    feas.max(1e-14 * m)
}

/// Checks `x` against every row, bound and integrality of `model`.
pub fn check(model: &Model, x: &[f64], opts: &CheckOptions) -> CheckReport {
    let mut report = CheckReport::default();
    assert_eq!(x.len(), model.vars.len(), "point has the wrong length");
    for (j, (v, &xj)) in model.vars.iter().zip(x).enumerate() {
        if !xj.is_finite() {
            report.violations.push(Violation::NotFinite {
                var: Some(j),
                row: None,
            });
            continue;
        }
        if v.lo.is_finite() {
            let tol = row_tolerance(opts.feasibility, v.lo.abs());
            if xj < v.lo - tol {
                report.violations.push(Violation::Bound {
                    var: j,
                    by: v.lo - xj,
                    tol,
                });
            }
        }
        if v.hi.is_finite() {
            let tol = row_tolerance(opts.feasibility, v.hi.abs());
            if xj > v.hi + tol {
                report.violations.push(Violation::Bound {
                    var: j,
                    by: xj - v.hi,
                    tol,
                });
            }
        }
        if v.is_integer() && (xj - xj.round()).abs() > opts.int_tol {
            report.violations.push(Violation::Integrality {
                var: j,
                by: (xj - xj.round()).abs(),
            });
        }
    }
    for (i, row) in model.rows.iter().enumerate() {
        let mut act = Dd::default();
        let mut mag = 0.0;
        for &(j, a) in &row.terms {
            let (p, e) = two_prod(a, x[j]);
            act.add(p);
            act.add(e);
            mag += p.abs();
        }
        let act_v = act.value();
        if !act_v.is_finite() || !mag.is_finite() {
            report.violations.push(Violation::NotFinite {
                var: None,
                row: Some(i),
            });
            continue;
        }
        for (side, below) in [(row.lo, true), (row.hi, false)] {
            if !side.is_finite() {
                continue;
            }
            // slack = side - activity, in double-double.
            let mut s = Dd::new(side);
            s.add(-act.hi);
            s.add(-act.lo);
            let slack = s.value();
            let tol = row_tolerance(opts.feasibility, side.abs() + mag);
            let by = if below { slack } else { -slack };
            report.worst_row_ratio = report.worst_row_ratio.max(by / tol);
            if by > tol {
                report.violations.push(Violation::Row { row: i, by, tol });
            }
        }
    }
    report
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::Direction;

    #[test]
    fn flags_rows_bounds_and_integrality() {
        let mut m = Model::new(Direction::Minimize);
        let x = m.add_int_var(1.0, 0.0, 3.0);
        let y = m.add_var(1.0, 0.0, 1.0);
        m.add_le(&[(x, 1.0), (y, 1.0)], 2.0);
        assert!(check(&m, &[1.0, 1.0], &CheckOptions::default()).ok());
        let r = check(&m, &[1.5, 1.2], &CheckOptions::default());
        assert_eq!(r.violations.len(), 3);
    }
}
