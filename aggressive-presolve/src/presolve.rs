//! The aggressive reductions and their postsolve. See `DESIGN.md` for the
//! soundness argument of each reduction.
//!
//! Every reduction is primal: it uses bounds and rows only, never the
//! objective. Bounds a reduction derives are emitted in the reduced model (for
//! continuous vars too, unlike microlp's own presolve), so a row may be dropped
//! on the strength of bounds derived from other rows. Row sides are carried in
//! double-double, so folding many fixed values into a side loses nothing, and
//! every activity is a compensated sum compared with an allowance of
//! `(n + 2)·2^-53·m`.

use crate::model::{Domain, Model, Row, Var};
use crate::sum::{allowance, Dd, Neumaier};
use std::collections::HashSet;
use std::time::{Duration, Instant};

/// Coefficients at or below this magnitude, relative to the magnitudes they
/// were computed from, are treated as cancelled to zero.
const DROP_TOL: f64 = 1e-12;
/// Integer bounds this large are not derived, to keep activity sums meaningful.
const HUGE_BOUND: f64 = 1e9;
const MAX_PASSES: usize = 100;

/// What presolve may do.
#[derive(Clone, Debug)]
pub struct PresolveOptions {
    /// Master switch; off, [`presolve`] returns the model unchanged.
    pub enabled: bool,
    /// The absolute feasibility tolerance of the contract (microlp's
    /// `Tolerances::feasibility`).
    pub feasibility: f64,
    /// How far from an integer an integer var's value may be.
    pub int_tol: f64,
    /// A substitution pivot must be at least this fraction of the row's
    /// largest coefficient.
    pub pivot_tol: f64,
    /// Maximum net nonzeros one substitution may add.
    pub max_fill: isize,
    /// A substitution may leave a row's `max|a|/min|a|` at most this many times
    /// the larger of that row's ratio and the pivot row's ratio as written.
    pub max_ratio_growth: f64,
    /// Turn on the substitutions (doubleton equations and implied-free
    /// columns).
    pub substitute: bool,
    /// Original rows presolve must leave alone: never dropped, never a
    /// substitution source, and none of their columns substituted out.
    pub protected_rows: HashSet<usize>,
    /// Original columns presolve must not eliminate by substitution.
    pub frozen_cols: HashSet<usize>,
}

impl Default for PresolveOptions {
    fn default() -> Self {
        PresolveOptions {
            enabled: true,
            feasibility: 1e-7,
            int_tol: 1e-6,
            pivot_tol: 0.01,
            max_fill: 10,
            max_ratio_growth: 4.0,
            substitute: true,
            protected_rows: HashSet::new(),
            frozen_cols: HashSet::new(),
        }
    }
}

/// Presolve's own proof of infeasibility. It carries no certificate:
/// [`crate::solve`] confirms it on the model as written.
#[derive(Clone, Debug, PartialEq)]
pub struct PresolveInfeasible {
    pub reason: String,
}

/// How often each reduction fired, and the size before and after.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Stats {
    pub rows_before: usize,
    pub cols_before: usize,
    pub nnz_before: usize,
    pub rows_after: usize,
    pub cols_after: usize,
    pub nnz_after: usize,
    pub passes: usize,
    pub empty_rows: usize,
    pub singleton_rows: usize,
    pub redundant_rows: usize,
    pub forcing_rows: usize,
    pub tightened_int_bounds: usize,
    pub fixed_cols: usize,
    pub empty_cols: usize,
    pub doubletons: usize,
    pub substitutions: usize,
    /// Substitutions refused by the fill-in cap or the ratio-growth cap.
    pub refused_fill: usize,
    pub refused_ratio: usize,
    /// Worst row `max|a|/min|a|` before and after.
    pub worst_ratio_before: f64,
    pub worst_ratio_after: f64,
    pub time: Duration,
}

/// Maps a point of the reduced model back to the original variables.
#[derive(Clone, Debug)]
pub struct Postsolve {
    num_orig: usize,
    /// Original column of each reduced column.
    kept: Vec<usize>,
    /// Eliminations in the order they were made; replayed backwards.
    ops: Vec<Op>,
    /// Objective constant moved out by eliminations (the reduced model's
    /// `obj_offset` includes it).
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
        rhs: Dd,
        terms: Vec<(usize, f64)>,
    },
}

impl Postsolve {
    /// The identity map on `n` columns.
    pub fn identity(n: usize) -> Self {
        Postsolve {
            num_orig: n,
            kept: (0..n).collect(),
            ops: Vec::new(),
            offset: 0.0,
        }
    }

    /// Original column of each reduced column.
    pub fn kept(&self) -> &[usize] {
        &self.kept
    }

    /// Number of eliminations.
    pub fn num_ops(&self) -> usize {
        self.ops.len()
    }

    /// The original column each op eliminated, in op order, and whether it was
    /// a substitution.
    pub fn eliminated(&self) -> impl Iterator<Item = (usize, bool)> + '_ {
        self.ops.iter().map(|op| match op {
            Op::Fix { col, .. } => (*col, false),
            Op::Substitute { col, .. } => (*col, true),
        })
    }

    /// Original-space values for a point of the reduced model.
    pub fn values(&self, reduced: &[f64]) -> Vec<f64> {
        assert_eq!(
            reduced.len(),
            self.kept.len(),
            "reduced point has the wrong length"
        );
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
                    let mut acc = *rhs;
                    for &(c, a) in terms {
                        acc.sub_prod(a, x[c]);
                    }
                    x[*col] = acc.value() / pivot;
                }
            }
        }
        x
    }
}

/// The result of [`presolve`].
#[derive(Clone, Debug)]
pub struct Presolved {
    pub reduced: Model,
    pub postsolve: Postsolve,
    pub stats: Stats,
}

struct PRow {
    terms: Vec<(usize, f64)>,
    lo: Dd,
    hi: Dd,
    alive: bool,
    protected: bool,
    /// `max|a|/min|a|` as written.
    base_ratio: f64,
}

impl PRow {
    fn is_eq(&self) -> bool {
        self.lo == self.hi && self.lo.is_finite()
    }
}

struct PCol {
    lo: f64,
    hi: f64,
    cost: f64,
    int: bool,
    rows: Vec<usize>,
    alive: bool,
    /// Never eliminated by substitution.
    pinned: bool,
}

/// Minimum and maximum activity of a row: compensated finite parts plus the
/// number of terms whose bound on that side is infinite.
#[derive(Clone, Copy)]
struct Activity {
    min: f64,
    min_inf: usize,
    max: f64,
    max_inf: usize,
    /// Round-off allowance of `min`/`max` and of removing one term from them.
    eps: f64,
    /// `Σ|finite contributions|`.
    mag: f64,
}

struct Presolver<'a> {
    rows: Vec<PRow>,
    cols: Vec<PCol>,
    ops: Vec<Op>,
    offset: Dd,
    opts: &'a PresolveOptions,
    changed: bool,
    stats: Stats,
    /// Scratch dense row for simulating substitutions.
    scratch: Vec<f64>,
}

fn ratio(terms: &[(usize, f64)]) -> f64 {
    let (mut mx, mut mn) = (0.0f64, f64::INFINITY);
    for &(_, a) in terms {
        mx = mx.max(a.abs());
        mn = mn.min(a.abs());
    }
    if terms.is_empty() {
        1.0
    } else {
        mx / mn
    }
}

fn infeasible(reason: impl Into<String>) -> PresolveInfeasible {
    PresolveInfeasible {
        reason: reason.into(),
    }
}

/// Presolve `model`. Returns [`PresolveInfeasible`] when a reduction shows that
/// no point meets the model (without a certificate).
pub fn presolve(model: &Model, opts: &PresolveOptions) -> Result<Presolved, PresolveInfeasible> {
    let started = Instant::now();
    let n = model.vars.len();
    let mut stats = Stats {
        rows_before: model.rows.len(),
        cols_before: n,
        nnz_before: model.num_nonzeros(),
        worst_ratio_before: model
            .rows
            .iter()
            .map(|r| ratio(&r.terms))
            .fold(1.0, f64::max),
        ..Stats::default()
    };
    if !opts.enabled {
        stats.rows_after = stats.rows_before;
        stats.cols_after = n;
        stats.nnz_after = stats.nnz_before;
        stats.worst_ratio_after = stats.worst_ratio_before;
        return Ok(Presolved {
            reduced: model.clone(),
            postsolve: Postsolve::identity(n),
            stats,
        });
    }
    let mut cols: Vec<PCol> = model
        .vars
        .iter()
        .enumerate()
        .map(|(j, v)| PCol {
            lo: v.lo,
            hi: v.hi,
            cost: v.cost,
            int: v.is_integer(),
            rows: Vec::new(),
            alive: true,
            pinned: opts.frozen_cols.contains(&j),
        })
        .collect();
    let mut rows = Vec::with_capacity(model.rows.len());
    for (i, row) in model.rows.iter().enumerate() {
        let terms: Vec<(usize, f64)> = row.terms.iter().copied().filter(|t| t.1 != 0.0).collect();
        let protected = opts.protected_rows.contains(&i);
        for &(c, _) in &terms {
            cols[c].rows.push(i);
            if protected {
                cols[c].pinned = true;
            }
        }
        rows.push(PRow {
            base_ratio: ratio(&terms),
            terms,
            lo: Dd::new(row.lo),
            hi: Dd::new(row.hi),
            alive: true,
            protected,
        });
    }
    let mut p = Presolver {
        rows,
        cols,
        ops: Vec::new(),
        offset: Dd::default(),
        opts,
        changed: false,
        stats,
        scratch: vec![0.0; n],
    };
    p.run()?;
    for (i, r) in p.rows.iter().enumerate() {
        if r.alive && r.terms.is_empty() {
            let tol = opts.feasibility;
            if r.lo.value() > tol || r.hi.value() < -tol {
                return Err(infeasible(format!("row {i} is empty and excludes 0")));
            }
        }
    }
    let mut presolved = p.finish(model);
    presolved.stats.time = started.elapsed();
    Ok(presolved)
}

impl Presolver<'_> {
    fn run(&mut self) -> Result<(), PresolveInfeasible> {
        for c in 0..self.cols.len() {
            self.round_integer_bounds(c)?;
        }
        for pass in 0..MAX_PASSES {
            self.stats.passes = pass + 1;
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
            if !self.changed && self.opts.substitute {
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

    /// Fix column `c` at `value` and fold it out of every row.
    fn fix_col(&mut self, c: usize, value: f64) {
        let rows = std::mem::take(&mut self.cols[c].rows);
        for r in rows {
            let row = &mut self.rows[r];
            let pos = row.terms.iter().position(|&(x, _)| x == c).unwrap();
            let (_, a) = row.terms.swap_remove(pos);
            row.lo.sub_prod(a, value);
            row.hi.sub_prod(a, value);
        }
        self.offset.add_prod(self.cols[c].cost, value);
        self.cols[c].alive = false;
        self.ops.push(Op::Fix { col: c, value });
        self.changed = true;
    }

    fn round_integer_bounds(&mut self, c: usize) -> Result<(), PresolveInfeasible> {
        let int_tol = self.opts.int_tol;
        let feas = self.opts.feasibility;
        let col = &mut self.cols[c];
        if col.int {
            col.lo = (col.lo - int_tol).ceil();
            col.hi = (col.hi + int_tol).floor();
            if col.lo > col.hi {
                return Err(infeasible(format!(
                    "integer col {c} has no integer in its bounds"
                )));
            }
        }
        if col.lo > col.hi {
            let tol = feas.max(1e-14 * col.lo.abs().max(col.hi.abs()));
            if col.lo > col.hi + tol {
                return Err(infeasible(format!("col {c} has crossing bounds")));
            }
            let mid = 0.5 * (col.lo + col.hi);
            col.lo = mid;
            col.hi = mid;
        }
        Ok(())
    }

    /// Tighten the bounds of column `c` to `[lo, hi]` (intersected). Integer
    /// bounds are rounded inward (after the caller relaxed them outward).
    fn tighten(&mut self, c: usize, lo: f64, hi: f64) -> Result<(), PresolveInfeasible> {
        let int_tol = self.opts.int_tol;
        let col = &mut self.cols[c];
        let (mut lo, mut hi) = (lo, hi);
        if col.int {
            lo = (lo - int_tol).ceil();
            hi = (hi + int_tol).floor();
        }
        let mut tightened = false;
        if lo > col.lo {
            col.lo = lo;
            tightened = true;
        }
        if hi < col.hi {
            col.hi = hi;
            tightened = true;
        }
        if tightened {
            self.changed = true;
            if col.int {
                self.stats.tightened_int_bounds += 1;
            }
        }
        self.round_integer_bounds(c)
    }

    fn activity(&self, r: usize) -> Activity {
        let (mut min, mut max) = (Neumaier::default(), Neumaier::default());
        let (mut min_inf, mut max_inf) = (0, 0);
        let terms = &self.rows[r].terms;
        for &(c, a) in terms {
            let (lo, hi) = (self.cols[c].lo, self.cols[c].hi);
            let (low_end, high_end) = if a > 0.0 { (lo, hi) } else { (hi, lo) };
            if low_end.is_finite() {
                min.add(a * low_end);
            } else {
                min_inf += 1;
            }
            if high_end.is_finite() {
                max.add(a * high_end);
            } else {
                max_inf += 1;
            }
        }
        let (minv, maxv) = (min.value(), max.value());
        let mag_fin = min.mag.max(max.mag);
        let eps = allowance(terms.len() + 2, mag_fin + minv.abs() + maxv.abs());
        Activity {
            min: minv,
            min_inf,
            max: maxv,
            max_inf,
            eps,
            mag: mag_fin,
        }
    }

    /// The minimum and maximum activity of every term of row `r` but `c`'s.
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

    /// Bounds on `x[c]` implied by row `r` and the other columns' bounds, in
    /// exact arithmetic up to the activity's round-off (not relaxed).
    fn implied_bounds(&self, r: usize, act: &Activity, c: usize, a: f64) -> (f64, f64) {
        let row = &self.rows[r];
        let (rest_min, rest_max) = self.activity_without(act, c, a);
        let ax_lo = row.lo.value() - rest_max;
        let ax_hi = row.hi.value() - rest_min;
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

    /// The row's contract at magnitude `m`: `max(feas, 1e-14·m)`.
    fn row_tol(&self, m: f64) -> f64 {
        self.opts.feasibility.max(1e-14 * m)
    }

    // ---- row reductions ----------------------------------------------------

    fn reduce_row(&mut self, r: usize) -> Result<(), PresolveInfeasible> {
        if self.rows[r].protected {
            return Ok(());
        }
        let (lo, hi) = (self.rows[r].lo.value(), self.rows[r].hi.value());
        let side_mag = lo
            .abs()
            .min(f64::MAX)
            .max(if hi.is_finite() { hi.abs() } else { 0.0 });
        let side_mag = if side_mag.is_finite() { side_mag } else { 0.0 };
        if lo > hi + self.row_tol(side_mag) {
            return Err(infeasible(format!("row {r} has crossing sides")));
        }
        match self.rows[r].terms.len() {
            0 => {
                let tol = self.opts.feasibility;
                if lo > tol || hi < -tol {
                    return Err(infeasible(format!("row {r} is empty and excludes 0")));
                }
                self.stats.empty_rows += 1;
                self.remove_row(r);
                return Ok(());
            }
            1 => {
                let (c, a) = self.rows[r].terms[0];
                let (mut blo, mut bhi) = if a > 0.0 {
                    (lo / a, hi / a)
                } else {
                    (hi / a, lo / a)
                };
                if self.cols[c].int {
                    // Keep every integer that meets the row within its contract.
                    if blo.is_finite() {
                        let side = if a > 0.0 { lo } else { hi };
                        blo -= self.row_tol(2.0 * side.abs()) / a.abs();
                    }
                    if bhi.is_finite() {
                        let side = if a > 0.0 { hi } else { lo };
                        bhi += self.row_tol(2.0 * side.abs()) / a.abs();
                    }
                }
                self.stats.singleton_rows += 1;
                self.remove_row(r);
                return self.tighten(c, blo, bhi);
            }
            _ => {}
        }

        let act = self.activity(r);
        let tol = self.row_tol(side_mag + act.mag);
        let slack = tol + act.eps;
        if act.min_inf == 0 && act.min > hi + slack || act.max_inf == 0 && act.max < lo - slack {
            return Err(infeasible(format!(
                "row {r} cannot be met within its bounds"
            )));
        }
        // Forcing: the row can only hold with every var at the bound that
        // attains the extreme activity.
        let forcing_low = act.min_inf == 0 && hi.is_finite() && act.min >= hi - act.eps;
        let forcing_high = act.max_inf == 0 && lo.is_finite() && act.max <= lo + act.eps;
        if forcing_low || forcing_high {
            // An integer var is fixed only where the room the row leaves cannot
            // move it by a whole unit.
            let room = tol + act.eps;
            let ok = self.rows[r]
                .terms
                .iter()
                .all(|&(c, a)| !self.cols[c].int || a.abs() > room);
            if ok {
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
        }
        // Redundant: the row holds at every point of the emitted bounds, even
        // with each var off its bound by its bound tolerance.
        let terms = &self.rows[r].terms;
        let bound_slack: f64 = terms
            .iter()
            .map(|&(c, a)| {
                let col = &self.cols[c];
                let b = col.lo.abs().max(col.hi.abs());
                let b = if b.is_finite() { b } else { 0.0 };
                a.abs() * self.opts.feasibility.max(1e-14 * b)
            })
            .sum();
        let below = lo == f64::NEG_INFINITY
            || act.min_inf == 0 && act.min - bound_slack - act.eps >= lo - self.row_tol(lo.abs());
        let above = hi == f64::INFINITY
            || act.max_inf == 0 && act.max + bound_slack + act.eps <= hi + self.row_tol(hi.abs());
        if below && above {
            self.stats.redundant_rows += 1;
            self.remove_row(r);
            return Ok(());
        }

        // Integer bound tightening. Continuous bounds are left alone: implied
        // continuous bounds from long rows only add degeneracy to the LP.
        let terms = self.rows[r].terms.clone();
        for (c, a) in terms {
            if !self.cols[c].int {
                continue;
            }
            let (ilo, ihi) = self.implied_bounds(r, &act, c, a);
            let margin = (tol + act.eps) / a.abs();
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

    fn reduce_col(&mut self, c: usize) {
        let col = &self.cols[c];
        if col.lo == col.hi {
            self.stats.fixed_cols += 1;
            self.fix_col(c, col.lo);
            return;
        }
        if col.rows.is_empty() && col.cost == 0.0 && !col.int {
            let value = 0.0f64.clamp(col.lo, col.hi);
            self.stats.empty_cols += 1;
            self.fix_col(c, value);
        }
    }

    // ---- substitution ------------------------------------------------------

    /// Eliminate one column of equality row `r` by substitution, if a safe one
    /// exists: doubleton equations qualify outright; longer rows need an
    /// implied-free column. Both are held to the pivot, fill-in and
    /// ratio-growth guards.
    fn try_substitute(&mut self, r: usize) -> Result<(), PresolveInfeasible> {
        let row = &self.rows[r];
        if row.protected || !row.is_eq() || row.terms.len() < 2 {
            return Ok(());
        }
        let rhs = row.lo;
        let max_abs = row.terms.iter().map(|t| t.1.abs()).fold(0.0, f64::max);
        let doubleton = row.terms.len() == 2;
        let act = self.activity(r);

        let mut candidates: Vec<(usize, f64, isize)> = Vec::new();
        for &(j, q) in &row.terms {
            let col = &self.cols[j];
            if col.pinned
                || q.abs() < self.opts.pivot_tol * max_abs
                || !self.integrality_preserved(r, j, q, rhs.value())
            {
                continue;
            }
            if !doubleton {
                let (ilo, ihi) = self.implied_bounds(r, &act, j, q);
                let slack = act.eps / q.abs();
                if !(ilo >= col.lo - slack && ihi <= col.hi + slack) {
                    continue;
                }
            }
            let fill = self.fill_in(r, j);
            if fill > self.opts.max_fill {
                self.stats.refused_fill += 1;
                continue;
            }
            // Prefer few added nonzeros, then the largest pivot, then few rows.
            let small_pivot = (q.abs() < max_abs) as isize;
            let nrows = self.cols[j].rows.len().min((1 << 19) - 1) as isize;
            let score = fill * (1 << 20) + small_pivot * (1 << 19) + nrows;
            candidates.push((j, q, score));
        }
        candidates.sort_by_key(|c| c.2);
        let mut chosen = None;
        for &(j, q, _) in &candidates {
            if self.ratio_ok(r, j, q) {
                chosen = Some((j, q));
                break;
            }
            self.stats.refused_ratio += 1;
        }
        let Some((j, pivot)) = chosen else {
            return Ok(());
        };
        let terms: Vec<(usize, f64)> = self.rows[r]
            .terms
            .iter()
            .copied()
            .filter(|&(c, _)| c != j)
            .collect();

        if doubleton {
            // pivot·x_j = rhs - a·x  with x_j in [l, u]  =>  a bound on x.
            let (x, a) = terms[0];
            let (l, u) = (self.cols[j].lo, self.cols[j].hi);
            let rhs_v = rhs.value();
            let (p_lo, p_hi) = if pivot > 0.0 {
                (pivot * l, pivot * u)
            } else {
                (pivot * u, pivot * l)
            };
            let (ax_lo, ax_hi) = (rhs_v - p_hi, rhs_v - p_lo);
            let (mut xlo, mut xhi) = if a > 0.0 {
                (ax_lo / a, ax_hi / a)
            } else {
                (ax_hi / a, ax_lo / a)
            };
            if xlo.is_nan() {
                xlo = f64::NEG_INFINITY;
            }
            if xhi.is_nan() {
                xhi = f64::INFINITY;
            }
            if self.cols[x].int {
                // Keep every integer x whose x_j meets its bounds within their
                // tolerance.
                let tl = self.opts.feasibility.max(1e-14 * l.abs());
                let tu = self.opts.feasibility.max(1e-14 * u.abs());
                let m = (pivot.abs() * tl.max(tu)
                    + self.row_tol(rhs_v.abs() + p_lo.abs().max(p_hi.abs()).min(f64::MAX)))
                    / a.abs();
                if xlo.is_finite() {
                    xlo -= m;
                }
                if xhi.is_finite() {
                    xhi += m;
                }
            }
            self.tighten(x, xlo, xhi)?;
            self.stats.doubletons += 1;
        } else {
            self.stats.substitutions += 1;
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
                self.add_coef(k, c, -factor * a);
            }
            let row = &mut self.rows[k];
            row.lo.sub_prod_dd(factor, rhs);
            row.hi.sub_prod_dd(factor, rhs);
        }
        let cj = self.cols[j].cost;
        if cj != 0.0 {
            for &(c, a) in &terms {
                let new = self.cols[c].cost - cj * a / pivot;
                self.cols[c].cost = if new.abs() <= DROP_TOL * (cj * a / pivot).abs() {
                    0.0
                } else {
                    new
                };
            }
            let f = cj / pivot;
            self.offset.add_prod(f, rhs.hi);
            self.offset.add_prod(f, rhs.lo);
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

    /// Add `delta` to the coefficient of column `c` in row `r`, dropping it
    /// where it cancels.
    fn add_coef(&mut self, r: usize, c: usize, delta: f64) {
        let row = &mut self.rows[r];
        if let Some(pos) = row.terms.iter().position(|&(x, _)| x == c) {
            let old = row.terms[pos].1;
            let new = old + delta;
            if new.abs() <= DROP_TOL * delta.abs().max(old.abs()) {
                row.terms.swap_remove(pos);
                self.cols[c].rows.retain(|&x| x != r);
            } else {
                row.terms[pos].1 = new;
            }
        } else if delta != 0.0 {
            row.terms.push((c, delta));
            self.cols[c].rows.push(r);
        }
    }

    /// Whether substituting `j` out of row `r` keeps every rewritten row's
    /// coefficient ratio within the cap.
    fn ratio_ok(&mut self, r: usize, j: usize, pivot: f64) -> bool {
        let growth = self.opts.max_ratio_growth;
        let base_r = self.rows[r].base_ratio;
        let mut scratch = std::mem::take(&mut self.scratch);
        let mut ok = true;
        for &k in &self.cols[j].rows {
            if k == r {
                continue;
            }
            let row = &self.rows[k];
            let akj = row.terms.iter().find(|t| t.0 == j).map_or(0.0, |t| t.1);
            let factor = akj / pivot;
            for &(c, a) in &row.terms {
                if c != j {
                    scratch[c] = a;
                }
            }
            let mut touched: Vec<usize> =
                row.terms.iter().map(|t| t.0).filter(|&c| c != j).collect();
            for &(c, a) in &self.rows[r].terms {
                if c == j {
                    continue;
                }
                if scratch[c] == 0.0 && !touched.contains(&c) {
                    touched.push(c);
                }
                let old = scratch[c];
                let delta = -factor * a;
                let new = old + delta;
                scratch[c] = if new.abs() <= DROP_TOL * delta.abs().max(old.abs()) {
                    0.0
                } else {
                    new
                };
            }
            let (mut mx, mut mn) = (0.0f64, f64::INFINITY);
            for &c in &touched {
                let v = scratch[c].abs();
                if v != 0.0 {
                    mx = mx.max(v);
                    mn = mn.min(v);
                }
                scratch[c] = 0.0;
            }
            if mn.is_finite() && mx / mn > growth * row.base_ratio.max(base_r).max(1.0) {
                ok = false;
                break;
            }
        }
        self.scratch = scratch;
        ok
    }

    /// Whether eliminating `j` from row `r` keeps the integer points exactly:
    /// a continuous `j` always does; an integer `j` only when the equality
    /// forces it to be integral.
    fn integrality_preserved(&self, r: usize, j: usize, pivot: f64, rhs: f64) -> bool {
        if !self.cols[j].int {
            return true;
        }
        let integral = |v: f64| (v - v.round()).abs() <= 1e-9 * v.abs().max(1.0);
        integral(rhs / pivot)
            && self.rows[r]
                .terms
                .iter()
                .filter(|&&(c, _)| c != j)
                .all(|&(c, a)| self.cols[c].int && integral(a / pivot))
    }

    /// Net nonzeros added by substituting `j` out of row `r`.
    fn fill_in(&self, r: usize, j: usize) -> isize {
        let len = self.rows[r].terms.len() as isize;
        let mut added = 0isize;
        for &k in &self.cols[j].rows {
            if k == r {
                continue;
            }
            let rk = &self.rows[k].terms;
            for &(c, _) in &self.rows[r].terms {
                if c != j && !rk.iter().any(|t| t.0 == c) {
                    added += 1;
                }
            }
            added -= 1; // x_j leaves row k
        }
        added - len
    }

    // ---- output ------------------------------------------------------------

    fn finish(mut self, model: &Model) -> Presolved {
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
        let offset = self.offset.value();
        reduced.obj_offset = model.obj_offset + offset;
        for &c in &kept {
            let col = &self.cols[c];
            reduced.vars.push(Var {
                name: model.vars[c].name.clone(),
                lo: col.lo,
                hi: col.hi,
                cost: col.cost,
                domain: if col.int {
                    Domain::Integer
                } else {
                    Domain::Real
                },
            });
        }
        for (i, row) in self.rows.iter_mut().enumerate() {
            if !row.alive || row.terms.is_empty() {
                continue;
            }
            let mut terms: Vec<(usize, f64)> =
                row.terms.iter().map(|&(c, a)| (reduced_of[c], a)).collect();
            terms.sort_by_key(|t| t.0);
            reduced.rows.push(Row {
                name: model.rows[i].name.clone(),
                terms,
                lo: row.lo.value(),
                hi: row.hi.value(),
            });
        }
        let mut stats = std::mem::take(&mut self.stats);
        stats.rows_after = reduced.rows.len();
        stats.cols_after = reduced.vars.len();
        stats.nnz_after = reduced.num_nonzeros();
        stats.worst_ratio_after = reduced
            .rows
            .iter()
            .map(|r| ratio(&r.terms))
            .fold(1.0, f64::max);
        Presolved {
            reduced,
            postsolve: Postsolve {
                num_orig,
                kept,
                ops: self.ops,
                offset,
            },
            stats,
        }
    }
}
