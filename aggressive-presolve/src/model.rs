//! The crate's own model: `microlp::Problem` has no getters, so presolve,
//! postsolve and the check work on this, and the reduced model is handed to
//! microlp through its public API ([`Model::to_problem`]).

use crate::sum::{dot, Sum};
use microlp::{ComparisonOp, OptimizationDirection, Problem, Variable};

/// Optimization direction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Direction {
    Minimize,
    Maximize,
}

/// Whether a variable is continuous or integer. A binary is an integer in
/// `[0, 1]`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VarKind {
    Continuous,
    Integer,
}

/// A variable: bounds (either may be infinite), objective coefficient, kind.
#[derive(Clone, Debug, PartialEq)]
pub struct Var {
    pub name: String,
    pub lo: f64,
    pub hi: f64,
    pub obj: f64,
    pub kind: VarKind,
}

impl Var {
    pub fn is_integer(&self) -> bool {
        self.kind == VarKind::Integer
    }
}

/// The sense of a one-sided or equality row, for [`Model::add_row`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Sense {
    Le,
    Ge,
    Eq,
}

/// A row `lo <= Σ a_j x_j <= hi`: an equality when `lo == hi`, `<=` when
/// `lo` is `-inf`, `>=` when `hi` is `+inf`, ranged otherwise. Terms are
/// sorted by variable, with no duplicates and no zeros.
#[derive(Clone, Debug, PartialEq)]
pub struct Row {
    pub name: String,
    pub terms: Vec<(usize, f64)>,
    pub lo: f64,
    pub hi: f64,
}

impl Row {
    /// Compensated activity at `x`.
    pub fn activity(&self, x: &[f64]) -> f64 {
        dot(self.terms.iter().map(|&(j, a)| (a, x[j])))
    }

    /// Largest over smallest |coefficient| (1 for an empty row).
    pub fn coef_ratio(&self) -> f64 {
        coef_ratio(self.terms.iter().map(|t| t.1))
    }
}

pub(crate) fn coef_ratio(coefs: impl Iterator<Item = f64>) -> f64 {
    let (mut lo, mut hi) = (f64::INFINITY, 0.0f64);
    for a in coefs {
        let a = a.abs();
        lo = lo.min(a);
        hi = hi.max(a);
    }
    if hi == 0.0 {
        1.0
    } else {
        hi / lo
    }
}

/// A MILP: `min/max c·x + offset` subject to row ranges, bounds and
/// integrality.
#[derive(Clone, Debug, PartialEq)]
pub struct Model {
    pub name: String,
    pub direction: Direction,
    pub obj_offset: f64,
    pub vars: Vec<Var>,
    pub rows: Vec<Row>,
}

impl Model {
    pub fn new(direction: Direction) -> Self {
        Model {
            name: String::new(),
            direction,
            obj_offset: 0.0,
            vars: Vec::new(),
            rows: Vec::new(),
        }
    }

    pub fn add_var(
        &mut self,
        name: impl Into<String>,
        lo: f64,
        hi: f64,
        obj: f64,
        kind: VarKind,
    ) -> usize {
        self.vars.push(Var {
            name: name.into(),
            lo,
            hi,
            obj,
            kind,
        });
        self.vars.len() - 1
    }

    /// A continuous var named `x<index>`.
    pub fn continuous(&mut self, obj: f64, lo: f64, hi: f64) -> usize {
        let name = format!("x{}", self.vars.len());
        self.add_var(name, lo, hi, obj, VarKind::Continuous)
    }

    /// An integer var named `x<index>`.
    pub fn integer(&mut self, obj: f64, lo: f64, hi: f64) -> usize {
        let name = format!("x{}", self.vars.len());
        self.add_var(name, lo, hi, obj, VarKind::Integer)
    }

    /// A binary var named `x<index>`.
    pub fn binary(&mut self, obj: f64) -> usize {
        self.integer(obj, 0.0, 1.0)
    }

    /// Adds `Σ a_j x_j (sense) rhs`. Duplicate vars are summed, zeros dropped.
    pub fn add_row(
        &mut self,
        name: impl Into<String>,
        terms: impl IntoIterator<Item = (usize, f64)>,
        sense: Sense,
        rhs: f64,
    ) -> usize {
        let (lo, hi) = match sense {
            Sense::Le => (f64::NEG_INFINITY, rhs),
            Sense::Ge => (rhs, f64::INFINITY),
            Sense::Eq => (rhs, rhs),
        };
        self.add_range_row(name, terms, lo, hi)
    }

    /// Adds `lo <= Σ a_j x_j <= hi`.
    pub fn add_range_row(
        &mut self,
        name: impl Into<String>,
        terms: impl IntoIterator<Item = (usize, f64)>,
        lo: f64,
        hi: f64,
    ) -> usize {
        let name = name.into();
        let name = if name.is_empty() {
            format!("r{}", self.rows.len())
        } else {
            name
        };
        self.rows.push(Row {
            name,
            terms: normalize_terms(terms),
            lo,
            hi,
        });
        self.rows.len() - 1
    }

    pub fn num_integer(&self) -> usize {
        self.vars.iter().filter(|v| v.is_integer()).count()
    }

    pub fn nnz(&self) -> usize {
        self.rows.iter().map(|r| r.terms.len()).sum()
    }

    /// The objective (offset included) at `x`, compensated.
    pub fn objective_at(&self, x: &[f64]) -> f64 {
        let mut s = Sum::new();
        s.add(self.obj_offset);
        for (v, &xv) in self.vars.iter().zip(x) {
            if v.obj != 0.0 {
                s.add_prod(v.obj, xv);
            }
        }
        s.value()
    }

    /// Rejects data microlp would reject, or that has no meaning.
    pub fn validate(&self) -> Result<(), String> {
        if !self.obj_offset.is_finite() {
            return Err("objective offset is not finite".into());
        }
        for (j, v) in self.vars.iter().enumerate() {
            if !v.obj.is_finite() {
                return Err(format!("var {} ({}): objective coefficient {}", j, v.name, v.obj));
            }
            if v.lo.is_nan() || v.hi.is_nan() || v.lo == f64::INFINITY || v.hi == f64::NEG_INFINITY
            {
                return Err(format!("var {} ({}): bounds [{}, {}]", j, v.name, v.lo, v.hi));
            }
        }
        for (i, r) in self.rows.iter().enumerate() {
            if r.lo.is_nan() || r.hi.is_nan() || r.lo == f64::INFINITY || r.hi == f64::NEG_INFINITY {
                return Err(format!("row {} ({}): range [{}, {}]", i, r.name, r.lo, r.hi));
            }
            for &(j, a) in &r.terms {
                if j >= self.vars.len() {
                    return Err(format!("row {} ({}): var index {} out of range", i, r.name, j));
                }
                if !a.is_finite() {
                    return Err(format!("row {} ({}): coefficient {} on var {}", i, r.name, a, j));
                }
            }
        }
        Ok(())
    }

    /// Builds the equivalent `microlp::Problem` through microlp's public API.
    ///
    /// A ranged row becomes a `>=` and a `<=` row; a free row (both sides
    /// infinite) is left out. An integer var's bounds go through
    /// `add_integer_var`'s `i32`s: integer bounds are rounded inward (exact
    /// for an integer var), an infinite one becomes `i32::MIN`/`i32::MAX`
    /// (microlp's idiom for "no bound"; it cuts nothing a model with values
    /// below `2^31` uses), and a finite one outside the `i32` range becomes
    /// `i32::MIN`/`MAX` plus a singleton row carrying the real bound. The
    /// objective offset is not part of the problem: add
    /// [`Model::obj_offset`].
    pub fn to_problem(&self) -> (Problem, Vec<Variable>) {
        let dir = match self.direction {
            Direction::Minimize => OptimizationDirection::Minimize,
            Direction::Maximize => OptimizationDirection::Maximize,
        };
        let mut p = Problem::new(dir);
        let mut extra = Vec::new();
        let vars: Vec<Variable> = self
            .vars
            .iter()
            .map(|v| match v.kind {
                VarKind::Continuous => p.add_var(v.obj, (v.lo, v.hi)),
                VarKind::Integer => {
                    let lo = v.lo.ceil();
                    let hi = v.hi.floor();
                    let (lo_i, lo_extra) = to_i32(lo);
                    let (hi_i, hi_extra) = to_i32(hi);
                    let var = p.add_integer_var(v.obj, (lo_i, hi_i));
                    if lo_extra {
                        extra.push((var, ComparisonOp::Ge, lo));
                    }
                    if hi_extra {
                        extra.push((var, ComparisonOp::Le, hi));
                    }
                    var
                }
            })
            .collect();
        for (var, op, b) in extra {
            p.add_constraint([(var, 1.0)], op, b);
        }
        for r in &self.rows {
            let terms: Vec<(Variable, f64)> = r.terms.iter().map(|&(j, a)| (vars[j], a)).collect();
            if r.lo == r.hi {
                p.add_constraint(terms.as_slice(), ComparisonOp::Eq, r.lo);
                continue;
            }
            if r.lo.is_finite() {
                p.add_constraint(terms.as_slice(), ComparisonOp::Ge, r.lo);
            }
            if r.hi.is_finite() {
                p.add_constraint(terms.as_slice(), ComparisonOp::Le, r.hi);
            }
        }
        (p, vars)
    }
}

/// An integral `f64` as an `i32` bound, and whether the real bound must be
/// carried by a row because it is finite but outside the `i32` range.
fn to_i32(x: f64) -> (i32, bool) {
    if x <= i32::MIN as f64 {
        (i32::MIN, x.is_finite() && x < i32::MIN as f64)
    } else if x >= i32::MAX as f64 {
        (i32::MAX, x.is_finite() && x > i32::MAX as f64)
    } else {
        (x as i32, false)
    }
}

/// Sorts by var, sums duplicates (compensated), drops zeros.
pub(crate) fn normalize_terms(terms: impl IntoIterator<Item = (usize, f64)>) -> Vec<(usize, f64)> {
    let mut t: Vec<(usize, f64)> = terms.into_iter().collect();
    t.sort_by_key(|&(j, _)| j);
    let mut out: Vec<(usize, f64)> = Vec::with_capacity(t.len());
    let mut i = 0;
    while i < t.len() {
        let j = t[i].0;
        let mut s = Sum::new();
        while i < t.len() && t[i].0 == j {
            s.add(t[i].1);
            i += 1;
        }
        let a = s.value();
        if a != 0.0 {
            out.push((j, a));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rows_are_normalized() {
        let mut m = Model::new(Direction::Minimize);
        let x = m.continuous(1.0, 0.0, 1.0);
        let y = m.continuous(1.0, 0.0, 1.0);
        m.add_row("", [(y, 1.0), (x, 2.0), (y, -1.0), (x, 0.5)], Sense::Le, 3.0);
        assert_eq!(m.rows[0].terms, vec![(x, 2.5)]);
        assert_eq!(m.rows[0].name, "r0");
    }

    #[test]
    fn to_problem_solves_the_same_model() {
        let mut m = Model::new(Direction::Maximize);
        let x = m.integer(1.0, 0.0, f64::INFINITY);
        let y = m.continuous(1.0, 0.0, f64::INFINITY);
        m.add_range_row("", [(x, 1.0), (y, 1.0)], 1.0, 4.5);
        m.add_row("", [(x, 1.0), (y, -1.0)], Sense::Eq, 0.0);
        m.obj_offset = 10.0;
        let (p, _) = m.to_problem();
        let sol = p.solve().unwrap();
        let sol = sol.solution().unwrap();
        // x = y, x integer, 2x <= 4.5 -> x = 2.
        assert!((sol.objective() + m.obj_offset - 14.0).abs() < 1e-9);
    }
}
