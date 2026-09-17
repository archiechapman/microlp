//! Primal presolve for mixed-integer problems.
//!
//! [`presolve`] turns a [`Problem`] into a smaller equivalent one plus a
//! [`Postsolve`] that maps a point of the reduced problem back to the original
//! variables. It runs immediately before `Problem::build_solver` (see
//! `ARCHITECTURE.md` §11) and only for problems with integer variables.
//!
//! Every reduction is primal: it uses bounds and constraint rows only, never the
//! objective, so no feasible point of the original problem is cut off. Points
//! that differ only in an eliminated variable are represented once, by the value
//! postsolve reconstructs. The reductions, iterated to a fixpoint:
//!
//! - empty rows (dropped, or infeasibility proven);
//! - singleton rows (turned into a variable bound, rounded inward for integers);
//! - redundant rows (activity bounds show the row can never bind);
//! - forcing rows (the row only holds with every variable at one bound);
//! - integer bound tightening from row activity;
//! - fixed columns (substituted into the rows and the objective constant);
//! - empty continuous columns with a zero objective coefficient (fixed inside
//!   their bounds; empty integer columns are kept, so distinct integer points stay
//!   distinct);
//! - parallel rows (merged when the result is still a one-sided or equality row);
//! - doubleton equations `a·x + b·y = c` (eliminate `y`, move its bounds to `x`);
//! - substitution of an implied-free column out of an equality row, with bounded
//!   fill-in (this covers free column singletons as the zero fill-in case).
//!
//! An integer column is only eliminated when the equality forces it to be
//! integral, so the reduced problem keeps exactly the original integer points.

use crate::{ComparisonOp, CsVec, Error, OptimizationDirection, Problem, VarDomain};
use std::collections::HashMap;

/// Coefficients at or below this magnitude, relative to the largest magnitude
/// they were computed from, are treated as cancelled to zero.
const DROP_TOL: f64 = 1e-12;
/// A substitution pivot must be at least this fraction of the row's largest
/// coefficient, so reconstruction does not amplify errors in the other terms.
const PIVOT_TOL: f64 = 0.01;
/// Maximum number of nonzeros a single substitution may add.
const MAX_FILL: isize = 10;
/// Bounds this large are not derived, to keep activity sums meaningful.
const HUGE_BOUND: f64 = 1e9;
const MAX_PASSES: usize = 100;

/// Maps a reduced-problem point back to the original variables.
#[derive(Clone, Debug)]
pub(crate) struct Postsolve {
    num_orig: usize,
    /// Original column of each reduced column.
    kept: Vec<usize>,
    /// Reduced column of each original column, if it was kept.
    reduced_of: Vec<Option<usize>>,
    /// Eliminations in the order they were made; postsolve replays them backwards.
    ops: Vec<Op>,
    /// Constant term (internal minimization space) moved out of the objective.
    pub offset: f64,
}

#[derive(Clone, Debug)]
enum Op {
    /// `x[col] = value`.
    Fix { col: usize, value: f64 },
    /// `x[col] = (rhs - Σ a·x[c]) / pivot`.
    Substitute {
        col: usize,
        pivot: f64,
        rhs: f64,
        terms: Vec<(usize, f64)>,
    },
}

impl Postsolve {
    pub(crate) fn reduced_col(&self, orig: usize) -> Option<usize> {
        self.reduced_of.get(orig).copied().flatten()
    }

    /// Original-space values for a point of the reduced problem.
    pub(crate) fn values(&self, reduced: &[f64]) -> Vec<f64> {
        let mut x = vec![0.0; self.num_orig];
        for (r, &o) in self.kept.iter().enumerate() {
            x[o] = reduced[r];
        }
        for op in self.ops.iter().rev() {
            match op {
                Op::Fix { col, value } => x[*col] = *value,
                Op::Substitute {
                    col,
                    pivot,
                    rhs,
                    terms,
                } => {
                    let rest: f64 = terms.iter().map(|&(c, a)| a * x[c]).sum();
                    x[*col] = (rhs - rest) / pivot;
                }
            }
        }
        x
    }
}

pub(crate) struct Presolved {
    pub problem: Problem,
    pub postsolve: Postsolve,
}

struct Row {
    terms: Vec<(usize, f64)>,
    lo: f64,
    hi: f64,
    alive: bool,
}

struct Col {
    lo: f64,
    hi: f64,
    cost: f64,
    domain: VarDomain,
    rows: Vec<usize>,
    alive: bool,
}

impl Col {
    fn is_integer(&self) -> bool {
        matches!(self.domain, VarDomain::Integer | VarDomain::Boolean)
    }
}

/// Minimum and maximum activity of a row: finite parts plus the number of
/// terms whose bound on that side is infinite.
#[derive(Clone, Copy)]
struct Activity {
    min: f64,
    min_inf: usize,
    max: f64,
    max_inf: usize,
    /// Σ|finite contributions|, a scale for rounding error in `min`/`max`.
    mag: f64,
}

/// How often each reduction fired, for the debug log.
#[derive(Debug, Default)]
struct Counts {
    empty_rows: usize,
    singleton_rows: usize,
    redundant_rows: usize,
    forcing_rows: usize,
    fixed_cols: usize,
    empty_cols: usize,
    parallel_rows: usize,
    doubletons: usize,
    substitutions: usize,
}

struct Presolver {
    rows: Vec<Row>,
    cols: Vec<Col>,
    ops: Vec<Op>,
    offset: f64,
    feas_tol: f64,
    int_tol: f64,
    changed: bool,
    counts: Counts,
}

/// Presolve `problem`. Returns [`Error::Infeasible`] when a reduction proves
/// that no feasible point exists.
pub(crate) fn presolve(problem: &Problem, feas_tol: f64, int_tol: f64) -> Result<Presolved, Error> {
    let n = problem.obj_coeffs.len();
    let mut cols: Vec<Col> = (0..n)
        .map(|v| Col {
            lo: problem.var_mins[v],
            hi: problem.var_maxs[v],
            cost: problem.obj_coeffs[v],
            domain: problem.var_domains[v].clone(),
            rows: Vec::new(),
            alive: true,
        })
        .collect();
    let mut rows = Vec::with_capacity(problem.constraints.len());
    for (coeffs, op, rhs) in &problem.constraints {
        let r = rows.len();
        let terms: Vec<(usize, f64)> = coeffs
            .iter()
            .filter(|&(_, &a)| a != 0.0)
            .map(|(c, &a)| (c, a))
            .collect();
        for &(c, _) in &terms {
            cols[c].rows.push(r);
        }
        let (lo, hi) = match op {
            ComparisonOp::Eq => (*rhs, *rhs),
            ComparisonOp::Le => (f64::NEG_INFINITY, *rhs),
            ComparisonOp::Ge => (*rhs, f64::INFINITY),
        };
        rows.push(Row {
            terms,
            lo,
            hi,
            alive: true,
        });
    }
    let mut p = Presolver {
        rows,
        cols,
        ops: Vec::new(),
        offset: 0.0,
        feas_tol,
        int_tol,
        changed: false,
        counts: Counts::default(),
    };
    p.run()?;
    // The pass limit can stop the loop with rows that were never re-checked.
    if p.rows
        .iter()
        .any(|r| r.alive && r.terms.is_empty() && (r.lo > feas_tol || r.hi < -feas_tol))
    {
        return Err(Error::Infeasible);
    }
    let counts = std::mem::take(&mut p.counts);
    let presolved = p.finish(problem.direction);
    debug!(
        "presolve: {} rows x {} cols -> {} x {}, {} eliminations; {:?}",
        problem.constraints.len(),
        n,
        presolved.problem.constraints.len(),
        presolved.problem.obj_coeffs.len(),
        presolved.postsolve.ops.len(),
        counts
    );
    Ok(presolved)
}

fn rel_eps(v: f64) -> f64 {
    DROP_TOL * v.abs().max(1.0)
}

impl Presolver {
    fn run(&mut self) -> Result<(), Error> {
        for c in 0..self.cols.len() {
            self.round_integer_bounds(c)?;
        }
        for _ in 0..MAX_PASSES {
            self.changed = false;
            for r in 0..self.rows.len() {
                if self.rows[r].alive {
                    self.reduce_row(r)?;
                }
            }
            for c in 0..self.cols.len() {
                if self.cols[c].alive {
                    self.reduce_col(c)?;
                }
            }
            if !self.changed {
                self.merge_parallel_rows()?;
            }
            if !self.changed {
                for r in 0..self.rows.len() {
                    if self.rows[r].alive {
                        self.try_substitute(r)?;
                    }
                }
            }
            if !self.changed {
                break;
            }
        }
        Ok(())
    }

    // ---- bookkeeping -------------------------------------------------------

    fn remove_row(&mut self, r: usize) {
        let terms = std::mem::take(&mut self.rows[r].terms);
        for (c, _) in terms {
            self.cols[c].rows.retain(|&x| x != r);
        }
        self.rows[r].alive = false;
        self.changed = true;
    }

    /// Add `delta` to the coefficient of column `c` in row `r`. `scale` is the
    /// magnitude the new value was computed from, for cancellation detection.
    fn add_coef(&mut self, r: usize, c: usize, delta: f64, scale: f64) {
        let row = &mut self.rows[r];
        if let Some(pos) = row.terms.iter().position(|&(x, _)| x == c) {
            let new = row.terms[pos].1 + delta;
            if new.abs() <= DROP_TOL * scale.max(row.terms[pos].1.abs()) {
                row.terms.swap_remove(pos);
                self.cols[c].rows.retain(|&x| x != r);
            } else {
                row.terms[pos].1 = new;
            }
        } else if delta.abs() > DROP_TOL * scale {
            row.terms.push((c, delta));
            self.cols[c].rows.push(r);
        }
    }

    fn coef(&self, r: usize, c: usize) -> f64 {
        self.rows[r]
            .terms
            .iter()
            .find(|&&(x, _)| x == c)
            .map_or(0.0, |&(_, a)| a)
    }

    /// Fix column `c` at `value` and remove it from every row.
    fn fix_col(&mut self, c: usize, value: f64) {
        let rows = std::mem::take(&mut self.cols[c].rows);
        for r in rows {
            let row = &mut self.rows[r];
            let pos = row.terms.iter().position(|&(x, _)| x == c).unwrap();
            let (_, a) = row.terms.swap_remove(pos);
            row.lo -= a * value;
            row.hi -= a * value;
        }
        self.offset += self.cols[c].cost * value;
        self.cols[c].alive = false;
        self.ops.push(Op::Fix { col: c, value });
        self.changed = true;
    }

    fn round_integer_bounds(&mut self, c: usize) -> Result<(), Error> {
        let col = &mut self.cols[c];
        if col.is_integer() {
            let lo = (col.lo - self.int_tol).ceil();
            let hi = (col.hi + self.int_tol).floor();
            if lo != col.lo || hi != col.hi {
                col.lo = lo;
                col.hi = hi;
            }
        }
        if col.lo > col.hi + self.feas_tol {
            return Err(Error::Infeasible);
        }
        if col.lo > col.hi {
            col.hi = col.lo;
        }
        Ok(())
    }

    /// Tighten the bounds of column `c` to `[lo, hi]` (intersected). Integer
    /// bounds are rounded inward.
    fn tighten(&mut self, c: usize, lo: f64, hi: f64) -> Result<(), Error> {
        let col = &mut self.cols[c];
        let (mut lo, mut hi) = (lo, hi);
        if col.is_integer() {
            lo = (lo - self.int_tol).ceil();
            hi = (hi + self.int_tol).floor();
        }
        if lo > col.lo {
            col.lo = lo;
            self.changed = true;
        }
        if hi < col.hi {
            col.hi = hi;
            self.changed = true;
        }
        self.round_integer_bounds(c)
    }

    fn activity(&self, r: usize) -> Activity {
        let mut act = Activity {
            min: 0.0,
            min_inf: 0,
            max: 0.0,
            max_inf: 0,
            mag: 0.0,
        };
        for &(c, a) in &self.rows[r].terms {
            let (lo, hi) = (self.cols[c].lo, self.cols[c].hi);
            let (low_end, high_end) = if a > 0.0 { (lo, hi) } else { (hi, lo) };
            if low_end.is_finite() {
                act.min += a * low_end;
                act.mag += (a * low_end).abs();
            } else {
                act.min_inf += 1;
            }
            if high_end.is_finite() {
                act.max += a * high_end;
                act.mag += (a * high_end).abs();
            } else {
                act.max_inf += 1;
            }
        }
        act
    }

    /// Bounds on `a·x[c]`'s complement in row `r`: the minimum and maximum
    /// activity of every other term, or infinite when unavailable.
    fn activity_without(&self, act: &Activity, c: usize, a: f64) -> (f64, f64) {
        let (lo, hi) = (self.cols[c].lo, self.cols[c].hi);
        let (low_end, high_end) = if a > 0.0 { (lo, hi) } else { (hi, lo) };
        let min = match (act.min_inf, low_end.is_finite()) {
            (0, _) => act.min - a * low_end,
            (1, false) => act.min,
            _ => f64::NEG_INFINITY,
        };
        let max = match (act.max_inf, high_end.is_finite()) {
            (0, _) => act.max - a * high_end,
            (1, false) => act.max,
            _ => f64::INFINITY,
        };
        (min, max)
    }

    /// Bounds on `x[c]` implied by row `r` and the other columns' bounds.
    fn implied_bounds(&self, r: usize, act: &Activity, c: usize, a: f64) -> (f64, f64) {
        let row = &self.rows[r];
        let (rest_min, rest_max) = self.activity_without(act, c, a);
        // lo <= a·x + rest <= hi  =>  lo - rest_max <= a·x <= hi - rest_min
        let ax_lo = row.lo - rest_max;
        let ax_hi = row.hi - rest_min;
        let (mut lo, mut hi) = if a > 0.0 {
            (ax_lo / a, ax_hi / a)
        } else {
            (ax_hi / a, ax_lo / a)
        };
        if lo.is_nan() {
            lo = f64::NEG_INFINITY;
        }
        if hi.is_nan() {
            hi = f64::INFINITY;
        }
        (lo, hi)
    }

    // ---- row reductions ----------------------------------------------------

    fn reduce_row(&mut self, r: usize) -> Result<(), Error> {
        let (lo, hi) = (self.rows[r].lo, self.rows[r].hi);
        if lo > hi + self.feas_tol {
            return Err(Error::Infeasible);
        }
        match self.rows[r].terms.len() {
            0 => {
                if lo > self.feas_tol || hi < -self.feas_tol {
                    return Err(Error::Infeasible);
                }
                self.counts.empty_rows += 1;
                self.remove_row(r);
                return Ok(());
            }
            1 => {
                let (c, a) = self.rows[r].terms[0];
                let (blo, bhi) = if a > 0.0 {
                    (lo / a, hi / a)
                } else {
                    (hi / a, lo / a)
                };
                self.counts.singleton_rows += 1;
                self.remove_row(r);
                return self.tighten(c, blo, bhi);
            }
            _ => {}
        }

        let act = self.activity(r);
        let slack = self.feas_tol + DROP_TOL * act.mag;
        if act.min_inf == 0 && act.min > hi + slack || act.max_inf == 0 && act.max < lo - slack {
            return Err(Error::Infeasible);
        }
        // Forcing: the row can only hold with every variable at the bound that
        // attains the extreme activity.
        let forcing_low = act.min_inf == 0 && hi.is_finite() && act.min >= hi - DROP_TOL * act.mag;
        let forcing_high = act.max_inf == 0 && lo.is_finite() && act.max <= lo + DROP_TOL * act.mag;
        if forcing_low || forcing_high {
            let terms = self.rows[r].terms.clone();
            self.counts.forcing_rows += 1;
            self.remove_row(r);
            for (c, a) in terms {
                let at_lo = (a > 0.0) == forcing_low;
                let value = if at_lo {
                    self.cols[c].lo
                } else {
                    self.cols[c].hi
                };
                self.fix_col(c, value);
            }
            return Ok(());
        }
        let below = lo == f64::NEG_INFINITY || act.min_inf == 0 && act.min >= lo - rel_eps(lo);
        let above = hi == f64::INFINITY || act.max_inf == 0 && act.max <= hi + rel_eps(hi);
        if below && above {
            self.counts.redundant_rows += 1;
            self.remove_row(r);
            return Ok(());
        }

        // Integer bound tightening. Continuous bounds are left alone: implied
        // continuous bounds only add degeneracy to the LP. `act` stays valid
        // through the loop: each column's own bound only changes on its turn,
        // and a looser bound on another column only weakens the deduction.
        let terms = self.rows[r].terms.clone();
        for (c, a) in terms {
            if !self.cols[c].is_integer() {
                continue;
            }
            let (ilo, ihi) = self.implied_bounds(r, &act, c, a);
            let margin = self.int_tol.max(DROP_TOL * act.mag);
            let ilo = if ilo.abs() < HUGE_BOUND {
                ilo - margin
            } else {
                f64::NEG_INFINITY
            };
            let ihi = if ihi.abs() < HUGE_BOUND {
                ihi + margin
            } else {
                f64::INFINITY
            };
            self.tighten(c, ilo, ihi)?;
        }
        Ok(())
    }

    // ---- column reductions -------------------------------------------------

    fn reduce_col(&mut self, c: usize) -> Result<(), Error> {
        let col = &self.cols[c];
        if col.lo == col.hi {
            self.counts.fixed_cols += 1;
            self.fix_col(c, col.lo);
            return Ok(());
        }
        if col.rows.is_empty() && col.cost == 0.0 && !col.is_integer() {
            let value = 0.0f64.clamp(col.lo, col.hi);
            self.counts.empty_cols += 1;
            self.fix_col(c, value);
        }
        Ok(())
    }

    // ---- parallel rows -----------------------------------------------------

    fn merge_parallel_rows(&mut self) -> Result<(), Error> {
        // Normalised support and quantised coefficients -> (row, scale) of each member.
        type Key = Vec<(usize, i64)>;
        let mut groups: HashMap<Key, Vec<(usize, f64)>> = HashMap::new();
        for r in 0..self.rows.len() {
            let row = &mut self.rows[r];
            if !row.alive || row.terms.len() < 2 {
                continue;
            }
            row.terms.sort_by_key(|&(c, _)| c);
            let scale = row.terms[0].1;
            let key = row
                .terms
                .iter()
                .map(|&(c, a)| (c, ((a / scale) * 1e9).round() as i64))
                .collect();
            groups.entry(key).or_default().push((r, scale));
        }
        for group in groups.into_values().filter(|g| g.len() > 1) {
            let (mut keep, mut keep_scale) = group[0];
            for &(other, other_scale) in &group[1..] {
                if !self.same_direction(keep, keep_scale, other, other_scale) {
                    continue;
                }
                let (klo, khi) = self.normalized_range(keep, keep_scale);
                let (olo, ohi) = self.normalized_range(other, other_scale);
                let (lo, hi) = (klo.max(olo), khi.min(ohi));
                if lo > hi + self.feas_tol / keep_scale.abs().min(other_scale.abs()).min(1.0) {
                    return Err(Error::Infeasible);
                }
                let hi = hi.max(lo);
                if (lo, hi) == (klo, khi) {
                    self.counts.parallel_rows += 1;
                    self.remove_row(other);
                    continue;
                }
                if (lo, hi) == (olo, ohi) {
                    self.counts.parallel_rows += 1;
                    self.remove_row(keep);
                    (keep, keep_scale) = (other, other_scale);
                    continue;
                }
                if lo.is_finite() && hi.is_finite() && lo < hi {
                    // The merge would need a ranged row; keep both.
                    continue;
                }
                let row = &mut self.rows[keep];
                if keep_scale > 0.0 {
                    row.lo = lo * keep_scale;
                    row.hi = hi * keep_scale;
                } else {
                    row.lo = hi * keep_scale;
                    row.hi = lo * keep_scale;
                }
                self.counts.parallel_rows += 1;
                self.remove_row(other);
            }
        }
        Ok(())
    }

    /// Whether two rows with the same hash key really are multiples.
    fn same_direction(&self, r: usize, rs: f64, s: usize, ss: f64) -> bool {
        self.rows[r].alive
            && self.rows[s].alive
            && self.rows[r].terms.len() == self.rows[s].terms.len()
            && self.rows[r]
                .terms
                .iter()
                .zip(&self.rows[s].terms)
                .all(|(&(c1, a1), &(c2, a2))| {
                    let (x, y) = (a1 / rs, a2 / ss);
                    c1 == c2 && (x - y).abs() <= 1e-12 * x.abs().max(1.0)
                })
    }

    fn normalized_range(&self, r: usize, scale: f64) -> (f64, f64) {
        let row = &self.rows[r];
        if scale > 0.0 {
            (row.lo / scale, row.hi / scale)
        } else {
            (row.hi / scale, row.lo / scale)
        }
    }

    // ---- substitution ------------------------------------------------------

    /// Eliminate one column of equality row `r` by substitution, if a safe one
    /// exists: doubleton equations always qualify; longer rows need an
    /// implied-free column and bounded fill-in.
    fn try_substitute(&mut self, r: usize) -> Result<(), Error> {
        let row = &self.rows[r];
        if row.lo != row.hi || !row.lo.is_finite() || row.terms.len() < 2 {
            return Ok(());
        }
        let rhs = row.lo;
        let max_abs = row.terms.iter().map(|t| t.1.abs()).fold(0.0, f64::max);
        let doubleton = row.terms.len() == 2;
        let act = self.activity(r);

        let mut best: Option<(usize, f64, isize)> = None;
        for &(j, q) in &row.terms {
            if q.abs() < PIVOT_TOL * max_abs || !self.integrality_preserved(r, j, q, rhs) {
                continue;
            }
            if !doubleton {
                let (ilo, ihi) = self.implied_bounds(r, &act, j, q);
                let col = &self.cols[j];
                let slack = DROP_TOL * act.mag;
                if !(ilo >= col.lo - slack && ihi <= col.hi + slack) {
                    continue;
                }
            }
            let fill = self.fill_in(r, j);
            if fill > MAX_FILL {
                continue;
            }
            let score = fill * 1024 + self.cols[j].rows.len() as isize;
            if best.is_none_or(|(_, _, s)| score < s) {
                best = Some((j, q, score));
            }
        }
        let Some((j, pivot, _)) = best else {
            return Ok(());
        };
        let terms: Vec<(usize, f64)> = self.rows[r]
            .terms
            .iter()
            .copied()
            .filter(|&(c, _)| c != j)
            .collect();

        if doubleton {
            // pivot·x_j = rhs - a·x  with x_j in [l, u]  =>  bound on x.
            let (x, a) = terms[0];
            let (l, u) = (self.cols[j].lo, self.cols[j].hi);
            let (p_lo, p_hi) = if pivot > 0.0 {
                (pivot * l, pivot * u)
            } else {
                (pivot * u, pivot * l)
            };
            let (ax_lo, ax_hi) = (rhs - p_hi, rhs - p_lo);
            let (xlo, xhi) = if a > 0.0 {
                (ax_lo / a, ax_hi / a)
            } else {
                (ax_hi / a, ax_lo / a)
            };
            let xlo = if xlo.is_nan() { f64::NEG_INFINITY } else { xlo };
            let xhi = if xhi.is_nan() { f64::INFINITY } else { xhi };
            self.tighten(x, xlo, xhi)?;
        }

        if doubleton {
            self.counts.doubletons += 1;
        } else {
            self.counts.substitutions += 1;
        }
        self.remove_row(r);
        let other_rows = std::mem::take(&mut self.cols[j].rows);
        for k in other_rows {
            let pos = self.rows[k]
                .terms
                .iter()
                .position(|&(c, _)| c == j)
                .unwrap();
            let (_, akj) = self.rows[k].terms.swap_remove(pos);
            let factor = akj / pivot;
            for &(c, a) in &terms {
                self.add_coef(k, c, -factor * a, (factor * a).abs());
            }
            self.rows[k].lo -= factor * rhs;
            self.rows[k].hi -= factor * rhs;
        }
        let cj = self.cols[j].cost;
        if cj != 0.0 {
            for &(c, a) in &terms {
                self.cols[c].cost -= cj * a / pivot;
            }
            self.offset += cj * rhs / pivot;
        }
        self.cols[j].alive = false;
        self.ops.push(Op::Substitute {
            col: j,
            pivot,
            rhs,
            terms,
        });
        self.changed = true;
        Ok(())
    }

    /// Whether eliminating `j` from row `r` keeps the integer points exactly:
    /// a continuous `j` always does; an integer `j` only when the equality
    /// forces it to be integral.
    fn integrality_preserved(&self, r: usize, j: usize, pivot: f64, rhs: f64) -> bool {
        if !self.cols[j].is_integer() {
            return true;
        }
        let integral = |v: f64| (v - v.round()).abs() <= 1e-9 * v.abs().max(1.0);
        integral(rhs / pivot)
            && self.rows[r]
                .terms
                .iter()
                .filter(|&&(c, _)| c != j)
                .all(|&(c, a)| self.cols[c].is_integer() && integral(a / pivot))
    }

    /// Net nonzeros added by substituting `j` out of row `r`.
    fn fill_in(&self, r: usize, j: usize) -> isize {
        let len = self.rows[r].terms.len() as isize;
        let mut added = 0isize;
        for &k in &self.cols[j].rows {
            if k == r {
                continue;
            }
            for &(c, _) in &self.rows[r].terms {
                if c != j && self.coef(k, c) == 0.0 {
                    added += 1;
                }
            }
            added -= 1; // x_j leaves row k
        }
        added - len
    }

    // ---- output ------------------------------------------------------------

    fn finish(self, direction: OptimizationDirection) -> Presolved {
        let num_orig = self.cols.len();
        let mut kept = Vec::new();
        let mut reduced_of = vec![None; num_orig];
        for (c, col) in self.cols.iter().enumerate() {
            if col.alive {
                reduced_of[c] = Some(kept.len());
                kept.push(c);
            }
        }
        let n = kept.len();
        let mut constraints = Vec::new();
        for row in self.rows.iter().filter(|r| r.alive && !r.terms.is_empty()) {
            let (idx, val): (Vec<usize>, Vec<f64>) = row
                .terms
                .iter()
                .map(|&(c, a)| {
                    (
                        reduced_of[c].expect("live row references a removed column"),
                        a,
                    )
                })
                .unzip();
            let vec = || CsVec::new_from_unsorted(n, idx.clone(), val.clone()).unwrap();
            if row.lo == row.hi {
                constraints.push((vec(), ComparisonOp::Eq, row.lo));
            } else {
                if row.lo.is_finite() {
                    constraints.push((vec(), ComparisonOp::Ge, row.lo));
                }
                if row.hi.is_finite() {
                    constraints.push((vec(), ComparisonOp::Le, row.hi));
                }
            }
        }
        let problem = Problem {
            direction,
            obj_coeffs: kept.iter().map(|&c| self.cols[c].cost).collect(),
            var_mins: kept.iter().map(|&c| self.cols[c].lo).collect(),
            var_maxs: kept.iter().map(|&c| self.cols[c].hi).collect(),
            var_domains: kept.iter().map(|&c| self.cols[c].domain.clone()).collect(),
            constraints,
            time_limit: None,
        };
        Presolved {
            problem,
            postsolve: Postsolve {
                num_orig,
                kept,
                reduced_of,
                ops: self.ops,
                offset: self.offset,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ComparisonOp, OptimizationDirection, Problem};

    fn reduce(problem: &Problem) -> Presolved {
        presolve(problem, 1e-7, 1e-6).expect("feasible")
    }

    fn size(p: &Presolved) -> (usize, usize) {
        (p.problem.constraints.len(), p.problem.obj_coeffs.len())
    }

    #[test]
    fn singleton_row_becomes_a_rounded_bound() {
        let mut problem = Problem::new(OptimizationDirection::Maximize);
        let x = problem.add_integer_var(1.0, (0, 100));
        let y = problem.add_integer_var(1.0, (0, 100));
        problem.add_constraint([(x, 2.0)], ComparisonOp::Le, 7.0);
        problem.add_constraint([(x, 1.0), (y, 3.0)], ComparisonOp::Le, 50.0);
        let p = reduce(&problem);
        assert_eq!(size(&p), (1, 2));
        assert_eq!(p.problem.var_maxs[0], 3.0);
        // y <= 50/3 from the remaining row, rounded down.
        assert_eq!(p.problem.var_maxs[1], 16.0);
    }

    #[test]
    fn fixed_and_zero_cost_empty_columns_go() {
        let mut problem = Problem::new(OptimizationDirection::Minimize);
        let x = problem.add_var(2.0, (3.0, 3.0));
        let y = problem.add_integer_var(1.0, (0, 9));
        let _unused = problem.add_var(0.0, (-4.0, -1.0));
        let z = problem.add_integer_var(1.0, (0, 9));
        problem.add_constraint([(x, 1.0), (y, 1.0), (z, 1.0)], ComparisonOp::Ge, 5.0);
        let p = reduce(&problem);
        assert_eq!(size(&p), (1, 2));
        assert_eq!(p.postsolve.offset, 6.0);
        assert_eq!(p.problem.constraints[0].2, 2.0);
        let values = p.postsolve.values(&[1.0, 1.0]);
        assert_eq!(values, vec![3.0, 1.0, -1.0, 1.0]);
    }

    #[test]
    fn redundant_and_forcing_rows_go() {
        let mut problem = Problem::new(OptimizationDirection::Minimize);
        let a = problem.add_binary_var(1.0);
        let b = problem.add_binary_var(1.0);
        let c = problem.add_binary_var(1.0);
        let d = problem.add_binary_var(-1.0);
        problem.add_constraint([(a, 1.0), (b, 1.0)], ComparisonOp::Le, 2.0); // redundant
        problem.add_constraint([(a, 1.0), (c, -1.0)], ComparisonOp::Ge, 1.0); // forcing: a=1, c=0
        problem.add_constraint([(b, 1.0), (d, 1.0), (c, 1.0)], ComparisonOp::Le, 1.0);
        let p = reduce(&problem);
        assert_eq!(p.problem.obj_coeffs.len(), 2); // b, d
        let values = p.postsolve.values(&[0.0, 1.0]);
        assert_eq!(values, vec![1.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn parallel_rows_merge_only_without_a_range() {
        let mut problem = Problem::new(OptimizationDirection::Minimize);
        let x = problem.add_integer_var(1.0, (0, 10));
        let y = problem.add_integer_var(1.0, (0, 10));
        problem.add_constraint([(x, 1.0), (y, 3.0)], ComparisonOp::Le, 9.0);
        problem.add_constraint([(x, -2.0), (y, -6.0)], ComparisonOp::Ge, -12.0);
        problem.add_constraint([(x, 3.0), (y, 9.0)], ComparisonOp::Ge, 3.0);
        let p = reduce(&problem);
        // The two <= sides merge (x + 3y <= 6); the >= side would make a range.
        assert_eq!(p.problem.constraints.len(), 2);
    }

    #[test]
    fn doubleton_equation_eliminates_the_continuous_side() {
        let mut problem = Problem::new(OptimizationDirection::Minimize);
        let x = problem.add_integer_var(1.0, (0, 10));
        let y = problem.add_var(3.0, (2.0, 5.0));
        let z = problem.add_integer_var(1.0, (0, 10));
        problem.add_constraint([(x, 1.0), (y, -2.0)], ComparisonOp::Eq, 1.0);
        problem.add_constraint([(y, 1.0), (z, 1.0)], ComparisonOp::Le, 6.0);
        let p = reduce(&problem);
        assert_eq!(size(&p), (1, 2));
        // y = (x - 1)/2 in [2, 5]  =>  x in [5, 11] ∩ [0, 10].
        assert_eq!((p.problem.var_mins[0], p.problem.var_maxs[0]), (5.0, 10.0));
        assert_eq!(p.problem.obj_coeffs[0], 1.0 + 1.5);
        assert_eq!(p.postsolve.offset, -1.5);
        let values = p.postsolve.values(&[7.0, 2.0]);
        assert_eq!(values, vec![7.0, 3.0, 2.0]);
    }

    #[test]
    fn integer_doubleton_needs_an_integral_ratio() {
        let mut problem = Problem::new(OptimizationDirection::Minimize);
        let x = problem.add_integer_var(1.0, (0, 10));
        let y = problem.add_integer_var(1.0, (0, 10));
        let z = problem.add_integer_var(1.0, (0, 10));
        problem.add_constraint([(x, 2.0), (y, -3.0)], ComparisonOp::Eq, 1.0);
        problem.add_constraint([(y, 1.0), (z, 1.0)], ComparisonOp::Le, 6.0);
        let p = reduce(&problem);
        // rhs/pivot is 1/2 or -1/3: eliminating either side could admit fractional points.
        assert_eq!(p.problem.obj_coeffs.len(), 3);

        let mut problem = Problem::new(OptimizationDirection::Minimize);
        let x = problem.add_integer_var(1.0, (0, 10));
        let y = problem.add_integer_var(1.0, (0, 10));
        let z = problem.add_integer_var(1.0, (0, 10));
        problem.add_constraint([(x, 3.0), (y, -1.0)], ComparisonOp::Eq, 1.0);
        problem.add_constraint([(y, 1.0), (z, 1.0)], ComparisonOp::Le, 6.0);
        let p = reduce(&problem);
        assert_eq!(p.problem.obj_coeffs.len(), 2);
    }

    #[test]
    fn implied_free_substitution_respects_fill_in() {
        // w = x1 + x2 + x3 appears in one more row; w's bounds are implied.
        let mut problem = Problem::new(OptimizationDirection::Minimize);
        let xs: Vec<_> = (0..3).map(|_| problem.add_binary_var(1.0)).collect();
        let w = problem.add_var(1.0, (0.0, 3.0));
        let v = problem.add_integer_var(1.0, (0, 5));
        let mut row: Vec<_> = xs.iter().map(|&x| (x, 1.0)).collect();
        row.push((w, -1.0));
        problem.add_constraint(row, ComparisonOp::Eq, 0.0);
        problem.add_constraint([(w, 1.0), (v, 1.0)], ComparisonOp::Ge, 2.0);
        let p = reduce(&problem);
        assert_eq!(size(&p), (1, 4));
        assert_eq!(p.problem.obj_coeffs, vec![2.0, 2.0, 2.0, 1.0]);
        let values = p.postsolve.values(&[1.0, 0.0, 1.0, 0.0]);
        assert_eq!(values[3], 2.0);

        // Not implied: w in [0, 1] is tighter than the row's [0, 3].
        let mut problem = Problem::new(OptimizationDirection::Minimize);
        let xs: Vec<_> = (0..3).map(|_| problem.add_binary_var(1.0)).collect();
        let w = problem.add_var(1.0, (0.0, 1.0));
        let v = problem.add_integer_var(1.0, (0, 5));
        let mut row: Vec<_> = xs.iter().map(|&x| (x, 1.0)).collect();
        row.push((w, -1.0));
        problem.add_constraint(row, ComparisonOp::Eq, 0.0);
        problem.add_constraint([(w, 1.0), (v, 1.0)], ComparisonOp::Ge, 2.0);
        let p = reduce(&problem);
        assert_eq!(p.problem.obj_coeffs.len(), 5);
    }

    #[test]
    fn contradictions_are_reported() {
        let mut problem = Problem::new(OptimizationDirection::Minimize);
        let x = problem.add_integer_var(1.0, (0, 3));
        problem.add_constraint([(x, 2.0)], ComparisonOp::Eq, 3.0); // x = 1.5
        assert!(matches!(
            presolve(&problem, 1e-7, 1e-6),
            Err(Error::Infeasible)
        ));
    }
}
