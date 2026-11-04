//! The aggressive presolve: reductions microlp's own presolve refuses, with a
//! postsolve map. Ported from the old in-tree `src/presolve.rs` (branch
//! `enumerate`) onto the crate's [`Model`], with compensated arithmetic and
//! the substitution guards of `DESIGN.md` §4.
//!
//! Every reduction is primal: it uses bounds and rows only, never the
//! objective (which only rides along through a substitution), so no feasible
//! point of the model is cut off, up to round-off and the margins below,
//! which always err toward keeping points and keeping rows. The reductions,
//! iterated to a fixpoint:
//!
//! - empty rows (dropped, or infeasibility found);
//! - singleton rows (turned into a variable bound, continuous vars included;
//!   rounded inward for integers);
//! - redundant rows (activity bounds show the row can never bind, with a
//!   margin for microlp leaving a bound by its tolerance);
//! - forcing rows (the row only holds with every variable at one bound,
//!   continuous vars included);
//! - integer bound tightening from row activity;
//! - fixed columns (folded into the rows and the objective constant);
//! - empty continuous columns with a zero objective coefficient (empty
//!   integer columns are kept, so distinct integer points stay distinct);
//! - doubleton equations `p·y + a·x = b` (eliminate `y`, move its bounds to
//!   `x`);
//! - substitution of an implied-free column out of an equality row, with a
//!   pivot, fill-in and coefficient-ratio guard (free column singletons are
//!   the zero fill-in case).
//!
//! An integer column is only eliminated when its equality forces it to be
//! integral (the other columns integer, `b/p` and every `a/p` integral), so
//! the reduced model keeps exactly the original integer points.
//!
//! Parallel rows are deliberately left alone: merging them was measured
//! slower on the model the old presolve was built for and dropped.

use crate::model::{coef_ratio, Model, Row as ModelRow, Var, VarKind};
use crate::sum::Dd;
use crate::Options;
use std::time::{Duration, Instant};

/// Coefficients at or below this magnitude, relative to the magnitudes they
/// were computed from, are treated as cancelled to zero.
const DROP_TOL: f64 = 1e-12;
/// Relative round-off of a compensated quantity: a few ulps.
const ROUNDOFF: f64 = 4.0 * f64::EPSILON;
/// The contract's relative floor (microlp's `ROUNDOFF_FLOOR`).
const ROUNDOFF_FLOOR: f64 = 1e-14;
/// Bounds this large are not derived from rows.
const HUGE_BOUND: f64 = 1e9;

/// Maps a point of the reduced model back to the original variables.
#[derive(Clone, Debug)]
pub struct Postsolve {
    num_orig: usize,
    /// Original column of each reduced column.
    kept: Vec<usize>,
    /// Eliminations in the order they were made; replayed backwards.
    ops: Vec<Op>,
}

/// One elimination.
#[derive(Clone, Debug)]
pub enum Op {
    /// `x[col] = value`.
    Fix { col: usize, value: f64 },
    /// `x[col] = (rhs − Σ a·x[c]) / pivot`.
    Substitute {
        col: usize,
        pivot: f64,
        rhs: Dd,
        terms: Vec<(usize, f64)>,
    },
}

impl Postsolve {
    /// Original column of each reduced column.
    pub fn kept(&self) -> &[usize] {
        &self.kept
    }

    pub fn ops(&self) -> &[Op] {
        &self.ops
    }

    /// Original-space values for a point of the reduced model: the kept
    /// columns' values placed, then the eliminations replayed in reverse.
    /// Each op only reads columns alive when it was recorded, which are kept
    /// or eliminated later, so already reconstructed.
    pub fn values(&self, reduced: &[f64]) -> Vec<f64> {
        assert_eq!(reduced.len(), self.kept.len(), "reduced point has the wrong length");
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
                    let mut s = *rhs;
                    for &(c, a) in terms {
                        s = s.add_prod(-a, x[c]);
                    }
                    x[*col] = s.value() / pivot;
                }
            }
        }
        x
    }
}

/// How often each reduction fired, and the model's size around presolve.
#[derive(Clone, Debug, Default)]
pub struct Stats {
    pub rows_before: usize,
    pub cols_before: usize,
    pub ints_before: usize,
    pub nnz_before: usize,
    pub rows_after: usize,
    pub cols_after: usize,
    pub ints_after: usize,
    pub nnz_after: usize,
    pub empty_rows: usize,
    pub singleton_rows: usize,
    pub redundant_rows: usize,
    pub forcing_rows: usize,
    pub tightened_bounds: usize,
    pub fixed_cols: usize,
    pub empty_cols: usize,
    pub doubletons: usize,
    pub substitutions: usize,
    /// Substitution candidates rejected by the pivot, fill-in and ratio guards.
    pub rejected_pivot: usize,
    pub rejected_fill: usize,
    pub rejected_ratio: usize,
    pub passes: usize,
    /// Worst row max/min |coefficient| ratio before and after.
    pub max_ratio_before: f64,
    pub max_ratio_after: f64,
    pub elapsed: Duration,
}

impl std::fmt::Display for Stats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "rows {} -> {}, cols {} -> {} (int {} -> {}), nnz {} -> {}; \
             empty rows {}, singleton {}, redundant {}, forcing {}, tightened {}, \
             fixed cols {}, empty cols {}, doubletons {}, substitutions {} \
             (rejected: pivot {}, fill {}, ratio {}); passes {}, max ratio {:.3e} -> {:.3e}, {:.3}s",
            self.rows_before,
            self.rows_after,
            self.cols_before,
            self.cols_after,
            self.ints_before,
            self.ints_after,
            self.nnz_before,
            self.nnz_after,
            self.empty_rows,
            self.singleton_rows,
            self.redundant_rows,
            self.forcing_rows,
            self.tightened_bounds,
            self.fixed_cols,
            self.empty_cols,
            self.doubletons,
            self.substitutions,
            self.rejected_pivot,
            self.rejected_fill,
            self.rejected_ratio,
            self.passes,
            self.max_ratio_before,
            self.max_ratio_after,
            self.elapsed.as_secs_f64()
        )
    }
}

/// A reduced model and the way back.
#[derive(Clone, Debug)]
pub struct Presolved {
    pub reduced: Model,
    pub postsolve: Postsolve,
    pub stats: Stats,
    /// Original row of each reduced row.
    pub row_origin: Vec<usize>,
}

/// Presolve found no point within the contract. Not a proof: the caller
/// solves the original model to decide.
#[derive(Clone, Debug, PartialEq)]
pub struct Infeasible {
    pub reason: String,
}

impl std::fmt::Display for Infeasible {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "presolve found the model infeasible: {}", self.reason)
    }
}

struct Row {
    terms: Vec<(usize, f64)>,
    lo: Dd,
    hi: Dd,
    alive: bool,
    protected: bool,
    /// The row's max/min |coefficient| ratio as written (at least 1).
    ratio0: f64,
    /// Σ|a·v| over the columns fixed out of the row, for its magnitude.
    fixed_mag: f64,
}

impl Row {
    fn is_equality(&self) -> bool {
        self.lo == self.hi && self.lo.is_finite()
    }
}

struct Col {
    lo: f64,
    hi: f64,
    cost: f64,
    kind: VarKind,
    rows: Vec<usize>,
    alive: bool,
    protected: bool,
}

impl Col {
    fn is_integer(&self) -> bool {
        self.kind == VarKind::Integer
    }
}

/// Minimum and maximum activity of a row: finite parts (double-double), the
/// number of terms whose bound on that side is infinite, the magnitudes
/// Σ|a·bound| at each corner and the slack Σ|a|·btol a corner can move by.
struct Activity {
    min: Dd,
    min_inf: usize,
    max: Dd,
    max_inf: usize,
    mag_min: f64,
    mag_max: f64,
    slack_min: f64,
    slack_max: f64,
}

struct Presolver<'a> {
    rows: Vec<Row>,
    cols: Vec<Col>,
    ops: Vec<Op>,
    offset: Dd,
    feas: f64,
    opts: &'a Options,
    changed: bool,
    stats: Stats,
    /// Dense scratch for substitution candidates, zero between uses.
    scratch: Vec<f64>,
    /// Generation marks for `scratch`'s columns met in a row.
    mark: Vec<u64>,
    stamp: u64,
}

/// Presolve `model`. `opts.protect_rows` / `opts.protect_vars` name original
/// rows that must reach microlp as written (less fixed columns) and vars that
/// must not be eliminated.
pub fn presolve(model: &Model, opts: &Options) -> Result<Presolved, Infeasible> {
    let start = Instant::now();
    let n = model.vars.len();
    let mut cols: Vec<Col> = model
        .vars
        .iter()
        .map(|v| Col {
            lo: v.lo,
            hi: v.hi,
            cost: v.obj,
            kind: v.kind,
            rows: Vec::new(),
            alive: true,
            protected: false,
        })
        .collect();
    for &j in &opts.protect_vars {
        if j < n {
            cols[j].protected = true;
        }
    }
    let mut rows = Vec::with_capacity(model.rows.len());
    for (r, row) in model.rows.iter().enumerate() {
        for &(c, _) in &row.terms {
            cols[c].rows.push(r);
        }
        rows.push(Row {
            terms: row.terms.clone(),
            lo: Dd::new(row.lo),
            hi: Dd::new(row.hi),
            alive: true,
            protected: false,
            ratio0: row.coef_ratio().max(1.0),
            fixed_mag: 0.0,
        });
    }
    for &r in &opts.protect_rows {
        if r < rows.len() {
            rows[r].protected = true;
        }
    }
    let stats = Stats {
        rows_before: model.rows.len(),
        cols_before: n,
        ints_before: model.num_integer(),
        nnz_before: model.nnz(),
        max_ratio_before: model.rows.iter().map(ModelRow::coef_ratio).fold(1.0, f64::max),
        ..Stats::default()
    };
    let mut p = Presolver {
        rows,
        cols,
        ops: Vec::new(),
        offset: Dd::new(model.obj_offset),
        feas: opts.feasibility,
        opts,
        changed: false,
        stats,
        scratch: vec![0.0; n],
        mark: vec![0; n],
        stamp: 0,
    };
    p.run()?;
    for r in 0..p.rows.len() {
        if p.rows[r].alive && p.rows[r].terms.is_empty() {
            p.empty_row(r)?;
        }
    }
    let mut presolved = p.finish(model);
    presolved.stats.elapsed = start.elapsed();
    Ok(presolved)
}

fn infeasible(reason: String) -> Infeasible {
    Infeasible { reason }
}

impl Presolver<'_> {
    fn run(&mut self) -> Result<(), Infeasible> {
        for c in 0..self.cols.len() {
            self.round_integer_bounds(c)?;
        }
        for _ in 0..self.opts.max_passes {
            self.stats.passes += 1;
            self.changed = false;
            for r in 0..self.rows.len() {
                if self.rows[r].alive {
                    self.reduce_row(r)?;
                }
            }
            for c in 0..self.cols.len() {
                if self.cols[c].alive {
                    self.reduce_col(c);
                }
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

    // ---- tolerances --------------------------------------------------------

    /// microlp's row contract at magnitude `m`.
    fn rtol(&self, m: f64) -> f64 {
        self.feas.max(ROUNDOFF_FLOOR * m)
    }

    /// microlp's contract on a bound.
    fn btol(&self, b: f64) -> f64 {
        if b.is_finite() {
            self.feas.max(ROUNDOFF_FLOOR * b.abs())
        } else {
            0.0
        }
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

    fn coef_pos(&self, r: usize, c: usize) -> Option<usize> {
        self.rows[r].terms.iter().position(|&(x, _)| x == c)
    }

    /// Fix column `c` at `value` and fold it out of every row.
    fn fix_col(&mut self, c: usize, value: f64) {
        let rows = std::mem::take(&mut self.cols[c].rows);
        for r in rows {
            let pos = self.coef_pos(r, c).expect("column listed in row");
            let row = &mut self.rows[r];
            let (_, a) = row.terms.swap_remove(pos);
            row.lo = row.lo.add_prod(-a, value);
            row.hi = row.hi.add_prod(-a, value);
            row.fixed_mag += (a * value).abs();
        }
        self.offset = self.offset.add_prod(self.cols[c].cost, value);
        self.cols[c].alive = false;
        self.ops.push(Op::Fix { col: c, value });
        self.changed = true;
    }

    fn round_integer_bounds(&mut self, c: usize) -> Result<(), Infeasible> {
        let (lo, hi) = (self.cols[c].lo, self.cols[c].hi);
        if self.cols[c].is_integer() {
            // Keep every integer within the bound's tolerance.
            let lo = (lo - self.btol(lo)).ceil();
            let hi = (hi + self.btol(hi)).floor();
            self.cols[c].lo = lo;
            self.cols[c].hi = hi;
            if lo > hi {
                return Err(infeasible(format!("integer var {} has no integer in its bounds", c)));
            }
            return Ok(());
        }
        if lo > hi {
            if lo - hi > self.btol(lo).max(self.btol(hi)) {
                return Err(infeasible(format!("var {} has bounds [{}, {}]", c, lo, hi)));
            }
            self.cols[c].lo = hi;
        }
        Ok(())
    }

    /// Tighten column `c` to `[lo, hi]` (intersected). An integer column's
    /// bounds are rounded inward after widening by `margin`, so every integer
    /// within `margin` of the range is kept.
    fn tighten(&mut self, c: usize, lo: f64, hi: f64, margin: f64) -> Result<(), Infeasible> {
        let col = &self.cols[c];
        let (mut lo, mut hi) = (lo, hi);
        if col.is_integer() {
            if lo.is_finite() {
                lo = (lo - margin - ROUNDOFF * lo.abs()).ceil();
            }
            if hi.is_finite() {
                hi = (hi + margin + ROUNDOFF * hi.abs()).floor();
            }
        }
        let mut tightened = false;
        if lo > self.cols[c].lo {
            self.cols[c].lo = lo;
            tightened = true;
        }
        if hi < self.cols[c].hi {
            self.cols[c].hi = hi;
            tightened = true;
        }
        if tightened {
            self.changed = true;
            self.stats.tightened_bounds += 1;
        }
        self.round_integer_bounds(c)
    }

    fn activity(&self, r: usize) -> Activity {
        let mut act = Activity {
            min: Dd::new(0.0),
            min_inf: 0,
            max: Dd::new(0.0),
            max_inf: 0,
            mag_min: 0.0,
            mag_max: 0.0,
            slack_min: 0.0,
            slack_max: 0.0,
        };
        for &(c, a) in &self.rows[r].terms {
            let (lo, hi) = (self.cols[c].lo, self.cols[c].hi);
            let (low_end, high_end) = if a > 0.0 { (lo, hi) } else { (hi, lo) };
            if low_end.is_finite() {
                act.min = act.min.add_prod(a, low_end);
                act.mag_min += (a * low_end).abs();
                act.slack_min += a.abs() * self.btol(low_end);
            } else {
                act.min_inf += 1;
            }
            if high_end.is_finite() {
                act.max = act.max.add_prod(a, high_end);
                act.mag_max += (a * high_end).abs();
                act.slack_max += a.abs() * self.btol(high_end);
            } else {
                act.max_inf += 1;
            }
        }
        act
    }

    /// The minimum and maximum activity of every term of row `r` but
    /// `a·x[c]`, infinite when unavailable.
    fn activity_without(&self, act: &Activity, c: usize, a: f64) -> (Dd, Dd) {
        let (lo, hi) = (self.cols[c].lo, self.cols[c].hi);
        let (low_end, high_end) = if a > 0.0 { (lo, hi) } else { (hi, lo) };
        let min = match (act.min_inf, low_end.is_finite()) {
            (0, _) => act.min.add_prod(-a, low_end),
            (1, false) => act.min,
            _ => Dd::new(f64::NEG_INFINITY),
        };
        let max = match (act.max_inf, high_end.is_finite()) {
            (0, _) => act.max.add_prod(-a, high_end),
            (1, false) => act.max,
            _ => Dd::new(f64::INFINITY),
        };
        (min, max)
    }

    /// Bounds on `x[c]` implied by row `r` and the other columns' bounds.
    fn implied_bounds(&self, r: usize, act: &Activity, c: usize, a: f64) -> (f64, f64) {
        let row = &self.rows[r];
        let (rest_min, rest_max) = self.activity_without(act, c, a);
        // lo <= a·x + rest <= hi  =>  lo − rest_max <= a·x <= hi − rest_min
        let ax_lo = if row.lo.is_finite() && rest_max.is_finite() {
            row.lo.add_dd(rest_max.neg()).value()
        } else {
            f64::NEG_INFINITY
        };
        let ax_hi = if row.hi.is_finite() && rest_min.is_finite() {
            row.hi.add_dd(rest_min.neg()).value()
        } else {
            f64::INFINITY
        };
        if a > 0.0 {
            (ax_lo / a, ax_hi / a)
        } else {
            (ax_hi / a, ax_lo / a)
        }
    }

    /// The row's magnitude for its tolerance at a corner of magnitude `mag`.
    fn row_mag(&self, r: usize, mag: f64) -> f64 {
        let row = &self.rows[r];
        let side = |d: Dd| if d.is_finite() { d.value().abs() } else { 0.0 };
        side(row.lo).max(side(row.hi)) + mag + row.fixed_mag
    }

    // ---- row reductions ----------------------------------------------------

    fn empty_row(&mut self, r: usize) -> Result<(), Infeasible> {
        let (lo, hi) = (self.rows[r].lo.value(), self.rows[r].hi.value());
        let tol = self.rtol(self.row_mag(r, 0.0));
        if lo > tol || hi < -tol {
            return Err(infeasible(format!(
                "row {} is empty with range [{}, {}]",
                r, lo, hi
            )));
        }
        self.stats.empty_rows += 1;
        self.remove_row(r);
        Ok(())
    }

    fn reduce_row(&mut self, r: usize) -> Result<(), Infeasible> {
        let (lo, hi) = (self.rows[r].lo.value(), self.rows[r].hi.value());
        let protected = self.rows[r].protected;
        match self.rows[r].terms.len() {
            0 => return self.empty_row(r),
            1 if !protected => {
                let (c, a) = self.rows[r].terms[0];
                let (blo, bhi) = if a > 0.0 {
                    (lo / a, hi / a)
                } else {
                    (hi / a, lo / a)
                };
                // An integer value meeting the row within its contract stays.
                let side = |v: f64| if v.is_finite() { v.abs() } else { 0.0 };
                let m = self.row_mag(r, side(lo).max(side(hi)));
                let margin = self.rtol(m) / a.abs();
                self.stats.singleton_rows += 1;
                self.remove_row(r);
                return self.tighten(c, blo, bhi, margin);
            }
            _ => {}
        }

        let act = self.activity(r);
        let m_min = self.row_mag(r, act.mag_min);
        let m_max = self.row_mag(r, act.mag_max);
        let min = act.min.value();
        let max = act.max.value();
        if act.min_inf == 0 && min > hi + self.rtol(m_min) + ROUNDOFF * m_min
            || act.max_inf == 0 && max < lo - self.rtol(m_max) - ROUNDOFF * m_max
        {
            return Err(infeasible(format!(
                "row {}: activity [{}, {}] misses its range [{}, {}]",
                r, min, max, lo, hi
            )));
        }
        let any_protected_col = || self.rows[r].terms.iter().any(|&(c, _)| self.cols[c].protected);
        if !protected && self.rows[r].terms.len() > 1 {
            // Forcing: the row only holds (within its contract) with every
            // variable at the bound that attains the extreme activity.
            let forcing_low =
                act.min_inf == 0 && hi.is_finite() && min >= hi - ROUNDOFF * m_min;
            let forcing_high =
                act.max_inf == 0 && lo.is_finite() && max <= lo + ROUNDOFF * m_max;
            if (forcing_low || forcing_high) && !any_protected_col() {
                let terms = self.rows[r].terms.clone();
                self.stats.forcing_rows += 1;
                self.remove_row(r);
                for (c, a) in terms {
                    if !self.cols[c].alive {
                        continue;
                    }
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
            // Redundant: the row holds for every point within the bounds,
            // even with each var past its bound by its tolerance, beyond what
            // the row's own contract absorbs.
            let below = lo == f64::NEG_INFINITY
                || act.min_inf == 0
                    && min - act.slack_min + self.rtol(m_min) >= lo + ROUNDOFF * m_min;
            let above = hi == f64::INFINITY
                || act.max_inf == 0
                    && max + act.slack_max - self.rtol(m_max) <= hi - ROUNDOFF * m_max;
            if below && above {
                self.stats.redundant_rows += 1;
                self.remove_row(r);
                return Ok(());
            }
        }

        // Integer bound tightening. Continuous bounds are left alone: implied
        // continuous bounds only add degeneracy to the LP. `act` stays valid
        // through the loop: each column's own bound only changes on its turn,
        // and a tighter bound on another column only makes the deduction
        // weaker than it could be.
        let terms = self.rows[r].terms.clone();
        for (c, a) in terms {
            if !self.cols[c].is_integer() {
                continue;
            }
            let (ilo, ihi) = self.implied_bounds(r, &act, c, a);
            let m = m_min.max(m_max).min(f64::MAX);
            let margin = (self.rtol(m) + ROUNDOFF * m) / a.abs();
            let ilo = if ilo.abs() < HUGE_BOUND {
                ilo
            } else {
                f64::NEG_INFINITY
            };
            let ihi = if ihi.abs() < HUGE_BOUND {
                ihi
            } else {
                f64::INFINITY
            };
            self.tighten(c, ilo, ihi, margin)?;
        }
        Ok(())
    }

    // ---- column reductions -------------------------------------------------

    fn reduce_col(&mut self, c: usize) {
        let col = &self.cols[c];
        if col.protected {
            return;
        }
        if col.lo == col.hi {
            self.stats.fixed_cols += 1;
            self.fix_col(c, col.lo);
            return;
        }
        if col.rows.is_empty() && col.cost == 0.0 && !col.is_integer() {
            let value = 0.0f64.clamp(col.lo, col.hi);
            self.stats.empty_cols += 1;
            self.fix_col(c, value);
        }
    }

    // ---- substitution ------------------------------------------------------

    /// Eliminate one column of equality row `r` by substitution, if a safe
    /// one exists: doubleton equations qualify without the implied-free
    /// test; longer rows need an implied-free column. Every candidate passes
    /// the pivot, integrality, fill-in and ratio guards.
    fn try_substitute(&mut self, r: usize) -> Result<(), Infeasible> {
        let row = &self.rows[r];
        if !row.is_equality() || row.terms.len() < 2 || row.protected {
            return Ok(());
        }
        let rhs = row.lo;
        let max_abs = row.terms.iter().map(|t| t.1.abs()).fold(0.0, f64::max);
        let doubleton = row.terms.len() == 2;
        let act = self.activity(r);

        let mut best: Option<(usize, f64, usize)> = None;
        let candidates = self.rows[r].terms.clone();
        for &(j, q) in &candidates {
            let col = &self.cols[j];
            if col.protected || col.rows.iter().any(|&k| self.rows[k].protected) {
                continue;
            }
            if q.abs() < 0.01 * max_abs {
                self.stats.rejected_pivot += 1;
                continue;
            }
            if !self.integrality_preserved(r, j, q, rhs.value()) {
                continue;
            }
            if !doubleton {
                let (ilo, ihi) = self.implied_bounds(r, &act, j, q);
                // Others leaving their bounds by their tolerance move x_j by
                // Σ|a_i/q|·btol_i; the implied range must hold even then.
                let spread = (act.slack_min.max(act.slack_max)) / q.abs();
                let lo_ok = col.lo == f64::NEG_INFINITY
                    || ilo.is_finite() && ilo - spread >= col.lo - ROUNDOFF * ilo.abs();
                let hi_ok = col.hi == f64::INFINITY
                    || ihi.is_finite() && ihi + spread <= col.hi + ROUNDOFF * ihi.abs();
                if !(lo_ok && hi_ok) {
                    continue;
                }
            }
            match self.substitution_cost(r, j, q) {
                Err(Reject::Fill) => self.stats.rejected_fill += 1,
                Err(Reject::Ratio) => self.stats.rejected_ratio += 1,
                Ok(fill) => {
                    let score = fill * 1024 + self.cols[j].rows.len();
                    if best.is_none_or(|(_, _, s)| score < s) {
                        best = Some((j, q, score));
                    }
                }
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
            // pivot·x_j = rhs − a·x  with x_j in [l, u]  =>  a bound on x.
            let (x, a) = terms[0];
            let (l, u) = (self.cols[j].lo, self.cols[j].hi);
            let p_lo = if pivot > 0.0 { l } else { u };
            let p_hi = if pivot > 0.0 { u } else { l };
            let ax_lo = if p_hi.is_finite() {
                rhs.add_prod(-pivot, p_hi).value()
            } else {
                f64::NEG_INFINITY
            };
            let ax_hi = if p_lo.is_finite() {
                rhs.add_prod(-pivot, p_lo).value()
            } else {
                f64::INFINITY
            };
            let (xlo, xhi) = if a > 0.0 {
                (ax_lo / a, ax_hi / a)
            } else {
                (ax_hi / a, ax_lo / a)
            };
            // An integer x keeps every value whose x_j is within x_j's bound
            // tolerance.
            let margin = self.btol(l).max(self.btol(u)) * (pivot / a).abs();
            self.tighten(x, xlo, xhi, margin)?;
            self.stats.doubletons += 1;
        } else {
            self.stats.substitutions += 1;
        }
        self.remove_row(r);
        let other_rows = std::mem::take(&mut self.cols[j].rows);
        for k in other_rows {
            let pos = self.coef_pos(k, j).expect("column listed in row");
            let (_, akj) = self.rows[k].terms.swap_remove(pos);
            let factor = akj / pivot;
            for &(c, a) in &terms {
                self.add_coef(k, c, -factor, a);
            }
            let row = &mut self.rows[k];
            row.lo = row.lo.add_prod(-factor, rhs.hi).add_prod(-factor, rhs.lo);
            row.hi = row.hi.add_prod(-factor, rhs.hi).add_prod(-factor, rhs.lo);
        }
        let cj = self.cols[j].cost;
        if cj != 0.0 {
            let f = cj / pivot;
            for &(c, a) in &terms {
                self.cols[c].cost = (-f).mul_add(a, self.cols[c].cost);
            }
            self.offset = self.offset.add_prod(f, rhs.hi).add_prod(f, rhs.lo);
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

    /// Add `factor · a` to the coefficient of column `c` in row `k`, dropping
    /// it where it cancels to round-off.
    fn add_coef(&mut self, k: usize, c: usize, factor: f64, a: f64) {
        let delta = factor * a;
        if let Some(pos) = self.coef_pos(k, c) {
            let old = self.rows[k].terms[pos].1;
            let new = factor.mul_add(a, old);
            if new.abs() <= DROP_TOL * delta.abs().max(old.abs()) {
                self.rows[k].terms.swap_remove(pos);
                self.cols[c].rows.retain(|&x| x != k);
            } else {
                self.rows[k].terms[pos].1 = new;
            }
        } else if delta != 0.0 {
            self.rows[k].terms.push((c, delta));
            self.cols[c].rows.push(k);
        }
    }

    /// Whether eliminating `j` from row `r` keeps the integer points
    /// exactly: a continuous `j` always does; an integer `j` only when the
    /// equality forces it to be integral.
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

    /// Net nonzeros added by substituting `j` (pivot `q`) out of row `r`, or
    /// the guard that rejects it: more fill than `max_fill`, or a rewritten
    /// row whose coefficient ratio would exceed `max_ratio_growth` times its
    /// ratio as written (and its current one).
    fn substitution_cost(&mut self, r: usize, j: usize, q: f64) -> Result<usize, Reject> {
        let len = self.rows[r].terms.len() as isize;
        for &(c, a) in &self.rows[r].terms {
            if c != j {
                self.scratch[c] = a;
            }
        }
        let mut added = 0isize;
        let mut verdict = Ok(());
        for &k in &self.cols[j].rows {
            if k == r {
                continue;
            }
            let row = &self.rows[k];
            let akj = row.terms.iter().find(|t| t.0 == j).map_or(0.0, |t| t.1);
            let factor = -akj / q;
            let (mut lo, mut hi) = (f64::INFINITY, 0.0f64);
            let mut seen = 0isize;
            let mut note = |v: f64| {
                let v = v.abs();
                if v > 0.0 {
                    lo = lo.min(v);
                    hi = hi.max(v);
                }
            };
            self.stamp += 1;
            for &(c, a) in &row.terms {
                if c == j {
                    continue;
                }
                let s = self.scratch[c];
                if s != 0.0 {
                    self.mark[c] = self.stamp;
                    seen += 1;
                    let new = factor.mul_add(s, a);
                    if new.abs() > DROP_TOL * a.abs().max((factor * s).abs()) {
                        note(new);
                    }
                } else {
                    note(a);
                }
            }
            for &(c, a) in &self.rows[r].terms {
                if c != j && self.mark[c] != self.stamp {
                    note(factor * a);
                }
            }
            added += (len - 1 - seen) - 1;
            let new_ratio = if hi == 0.0 { 1.0 } else { hi / lo };
            let cur_ratio = coef_ratio(row.terms.iter().map(|t| t.1));
            let cap = self.opts.max_ratio_growth * row.ratio0;
            if new_ratio > cap && new_ratio > cur_ratio {
                verdict = Err(Reject::Ratio);
                break;
            }
        }
        for &(c, _) in &self.rows[r].terms {
            self.scratch[c] = 0.0;
        }
        verdict?;
        let fill = added - len;
        if fill > self.opts.max_fill as isize {
            return Err(Reject::Fill);
        }
        Ok(fill.max(0) as usize)
    }

    // ---- output ------------------------------------------------------------

    fn finish(self, model: &Model) -> Presolved {
        let num_orig = self.cols.len();
        let mut kept = Vec::new();
        let mut reduced_of = vec![usize::MAX; num_orig];
        for (c, col) in self.cols.iter().enumerate() {
            if col.alive {
                reduced_of[c] = kept.len();
                kept.push(c);
            }
        }
        let mut reduced = Model::new(model.direction);
        reduced.name = model.name.clone();
        reduced.obj_offset = self.offset.value();
        reduced.vars = kept
            .iter()
            .map(|&c| Var {
                name: model.vars[c].name.clone(),
                lo: self.cols[c].lo,
                hi: self.cols[c].hi,
                obj: self.cols[c].cost,
                kind: self.cols[c].kind,
            })
            .collect();
        let mut row_origin = Vec::new();
        for (r, row) in self.rows.iter().enumerate() {
            if !row.alive || row.terms.is_empty() {
                continue;
            }
            let terms = row.terms.iter().map(|&(c, a)| (reduced_of[c], a));
            let (lo, hi) = if row.lo == row.hi {
                let v = row.lo.value();
                (v, v)
            } else {
                (row.lo.value(), row.hi.value())
            };
            reduced.add_range_row(model.rows[r].name.clone(), terms, lo, hi);
            row_origin.push(r);
        }
        let mut stats = self.stats;
        stats.rows_after = reduced.rows.len();
        stats.cols_after = reduced.vars.len();
        stats.ints_after = reduced.num_integer();
        stats.nnz_after = reduced.nnz();
        stats.max_ratio_after = reduced.rows.iter().map(ModelRow::coef_ratio).fold(1.0, f64::max);
        Presolved {
            reduced,
            postsolve: Postsolve {
                num_orig,
                kept,
                ops: self.ops,
            },
            stats,
            row_origin,
        }
    }
}

enum Reject {
    Fill,
    Ratio,
}
