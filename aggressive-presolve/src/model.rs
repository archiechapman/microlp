//! The crate's own model: microlp's `Problem` has no getters, so presolve,
//! postsolve and the check work on this, and the reduced model is handed to
//! microlp through its public API ([`Model::to_problem`]).

use microlp::{ComparisonOp, OptimizationDirection, Problem, Variable};

/// Whether a variable must take an integer value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Domain {
    Real,
    /// Integer; a binary is an integer on `[0, 1]`.
    Integer,
}

/// Optimization direction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Direction {
    Minimize,
    Maximize,
}

/// A variable `lo <= x <= hi`, with its objective coefficient.
#[derive(Clone, Debug, PartialEq)]
pub struct Var {
    pub name: String,
    pub lo: f64,
    pub hi: f64,
    pub cost: f64,
    pub domain: Domain,
}

impl Var {
    pub fn is_integer(&self) -> bool {
        self.domain == Domain::Integer
    }
}

/// A row `lo <= Σ a_j x_j <= hi`: `<=` has `lo = -inf`, `>=` has `hi = +inf`,
/// `=` has `lo == hi`, and a ranged row has both finite.
#[derive(Clone, Debug, PartialEq)]
pub struct Row {
    pub name: String,
    /// `(var index, coefficient)`, at most one entry per var.
    pub terms: Vec<(usize, f64)>,
    pub lo: f64,
    pub hi: f64,
}

/// A MILP: minimize or maximize `obj_offset + Σ cost_j x_j` over the rows and
/// bounds.
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

    /// Adds a continuous variable and returns its index.
    pub fn add_var(&mut self, cost: f64, lo: f64, hi: f64) -> usize {
        self.push_var(cost, lo, hi, Domain::Real)
    }

    /// Adds an integer variable and returns its index.
    pub fn add_int_var(&mut self, cost: f64, lo: f64, hi: f64) -> usize {
        self.push_var(cost, lo, hi, Domain::Integer)
    }

    fn push_var(&mut self, cost: f64, lo: f64, hi: f64, domain: Domain) -> usize {
        let j = self.vars.len();
        self.vars.push(Var {
            name: format!("x{j}"),
            lo,
            hi,
            cost,
            domain,
        });
        j
    }

    /// Adds `lo <= Σ terms <= hi`. Repeated vars are summed and zeros dropped.
    pub fn add_row(&mut self, terms: &[(usize, f64)], lo: f64, hi: f64) -> usize {
        let i = self.rows.len();
        let mut merged: Vec<(usize, f64)> = Vec::with_capacity(terms.len());
        let mut sorted = terms.to_vec();
        sorted.sort_by_key(|t| t.0);
        for (j, a) in sorted {
            match merged.last_mut() {
                Some(last) if last.0 == j => last.1 += a,
                _ => merged.push((j, a)),
            }
        }
        merged.retain(|t| t.1 != 0.0);
        self.rows.push(Row {
            name: format!("r{i}"),
            terms: merged,
            lo,
            hi,
        });
        i
    }

    pub fn add_le(&mut self, terms: &[(usize, f64)], rhs: f64) -> usize {
        self.add_row(terms, f64::NEG_INFINITY, rhs)
    }

    pub fn add_ge(&mut self, terms: &[(usize, f64)], rhs: f64) -> usize {
        self.add_row(terms, rhs, f64::INFINITY)
    }

    pub fn add_eq(&mut self, terms: &[(usize, f64)], rhs: f64) -> usize {
        self.add_row(terms, rhs, rhs)
    }

    pub fn num_integer(&self) -> usize {
        self.vars.iter().filter(|v| v.is_integer()).count()
    }

    pub fn num_nonzeros(&self) -> usize {
        self.rows.iter().map(|r| r.terms.len()).sum()
    }

    /// The objective at `x`, offset included, in compensated arithmetic.
    pub fn objective(&self, x: &[f64]) -> f64 {
        let (v, _) = crate::sum::dot(self.vars.iter().zip(x).map(|(v, &x)| (v.cost, x)));
        v + self.obj_offset
    }

    /// Builds the microlp problem: a ranged row becomes two constraints and
    /// an integer var on `[0, 1]` a binary. Every other integer var takes its
    /// bounds as `i32` (an infinite side becomes `i32::MIN`/`i32::MAX`, as
    /// microlp's docs recommend). The objective offset is not part of the
    /// problem. Errors on a finite integer bound outside `i32`, on a
    /// non-integral integer bound, and on an empty row whose sides exclude 0
    /// is kept for microlp to judge.
    pub fn to_problem(&self) -> Result<(Problem, Vec<Variable>), String> {
        let dir = match self.direction {
            Direction::Minimize => OptimizationDirection::Minimize,
            Direction::Maximize => OptimizationDirection::Maximize,
        };
        let mut problem = Problem::new(dir);
        let mut handles = Vec::with_capacity(self.vars.len());
        for v in &self.vars {
            let h = match v.domain {
                Domain::Real => problem.add_var(v.cost, (v.lo, v.hi)),
                Domain::Integer if v.lo == 0.0 && v.hi == 1.0 => problem.add_binary_var(v.cost),
                Domain::Integer => {
                    let lo = int_bound(v.lo, i32::MIN, &v.name)?;
                    let hi = int_bound(v.hi, i32::MAX, &v.name)?;
                    problem.add_integer_var(v.cost, (lo, hi))
                }
            };
            handles.push(h);
        }
        for r in &self.rows {
            let expr: Vec<(Variable, f64)> =
                r.terms.iter().map(|&(j, a)| (handles[j], a)).collect();
            if r.lo == r.hi {
                problem.add_constraint(expr.as_slice(), ComparisonOp::Eq, r.lo);
                continue;
            }
            if r.lo > f64::NEG_INFINITY {
                problem.add_constraint(expr.as_slice(), ComparisonOp::Ge, r.lo);
            }
            if r.hi < f64::INFINITY {
                problem.add_constraint(expr.as_slice(), ComparisonOp::Le, r.hi);
            }
        }
        Ok((problem, handles))
    }
}

fn int_bound(b: f64, inf: i32, name: &str) -> Result<i32, String> {
    if b.is_infinite() {
        return Ok(inf);
    }
    if b != b.round() {
        return Err(format!("integer var {name} has a fractional bound {b}"));
    }
    if b < i32::MIN as f64 || b > i32::MAX as f64 {
        return Err(format!(
            "integer var {name} has a bound {b} outside i32, which microlp's API cannot take"
        ));
    }
    Ok(b as i32)
}
