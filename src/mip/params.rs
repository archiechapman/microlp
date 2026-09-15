//! Internal, non-configurable numeric constants for the branch & bound driver.
//!
//! These differ from [`crate::Tolerances`] in kind, not just location: every
//! constant here is an implementation-detail knob that keeps one specific
//! piece of solver machinery numerically well-behaved — a denominator that
//! must not be zero, a floor that keeps a score function totally ordered, a
//! slack against float round-trip noise in an already-validated input. None
//! of them carries user-facing semantic meaning, none is expected to need
//! tuning from outside the crate, and changing one is a source edit plus a
//! rebuild, never a per-solve option. If a number here ever needs to become
//! something a caller can reasonably want to tune, promote it to
//! [`crate::Tolerances`] rather than adding a separate top-level
//! [`crate::SolveOptions`] field for it.

/// Floor applied to each side of the pseudocost product-score
/// `max(est_down * f_down, SCORE_EPS) * max(est_up * f_up, SCORE_EPS)`
/// computed by `choose_branch_var`. Without it, a variable whose estimated
/// degradation on one side is still exactly (or numerically near) zero —
/// always true before any observation on a zero-objective-coefficient
/// variable — would score exactly zero and be indistinguishable from a
/// variable that is genuinely useless to branch on; the floor keeps the
/// product rule a well-defined total order from the very first branching
/// decision onward.
pub(crate) const SCORE_EPS: f64 = 1e-6;

/// Offset added to `|obj_coeff|` when seeding a variable's initial
/// (pre-observation) pseudocost estimate in `PseudoCosts::new`. Keeps the
/// seed estimate strictly positive even for a variable with a zero
/// objective coefficient, so [`SCORE_EPS`]'s floor above is never the only
/// thing standing between an unobserved variable and a hard-zero score.
pub(crate) const PSEUDOCOST_INIT_EPS: f64 = 1e-6;

/// Denominator guard applied to `branch_frac` when a node's actual
/// objective degradation is folded back into the pseudocost that predicted
/// it (`PseudoCosts::record`, called from the search loop after each node
/// solve). `branch_frac` is a fractional part in `(0.0, 1.0]` by
/// construction, but the guard is kept defensively so a value that rounds
/// to (numerically) zero can never divide by zero or blow up the recorded
/// per-unit degradation.
pub(crate) const BRANCH_FRAC_GUARD: f64 = 1e-6;

/// Denominator guard in `relative_gap`'s `incumbent_obj.abs().max(..)`: an
/// incumbent objective of exactly (or numerically near) zero must not make
/// the reported relative gap divide by zero or explode to `inf`.
pub(crate) const GAP_DENOM_GUARD: f64 = 1e-10;

/// Slack allowed on each side of a warm-start hint's bound check in
/// `try_warm_start`. A hint that misses its variable's bounds by more than
/// float round-trip noise is a genuinely invalid hint and is dropped — the
/// whole mechanism is advisory, per [`crate::SolveOptions::warm_start`] —
/// this constant only forgives noise in the hint value itself, never a real
/// out-of-bounds input.
pub(crate) const HINT_BOUNDS_SLACK: f64 = 1e-9;

/// Most Gomory cuts added per root round (the most fractional rows first): one per
/// `GOMORY_ROWS_PER_CUT_ROUND` model rows, clamped to `[GOMORY_MIN_CUTS_PER_ROUND,
/// GOMORY_MAX_CUTS_PER_ROUND]`. Every cut is a new row in every later node LP, so the
/// budget scales with the model instead of taking one cut per fractional row.
pub(crate) const GOMORY_MAX_CUTS_PER_ROUND: usize = 100;
pub(crate) const GOMORY_MIN_CUTS_PER_ROUND: usize = 10;
pub(crate) const GOMORY_ROWS_PER_CUT_ROUND: usize = 10;

/// All root rounds together add at most one cut per this many model rows (at least
/// `GOMORY_MIN_CUTS_PER_ROUND`).
pub(crate) const GOMORY_ROWS_PER_CUT_TOTAL: usize = 2;

/// Most candidates strong-branched at one node (the best by pseudocost score first).
/// Probing every fractional variable would cost more than the branching decision is
/// worth; the classic reliability-branching implementations cap it the same way.
pub(crate) const SB_MAX_CANDIDATES: usize = 8;

/// Simplex iterations one strong-branching probe may use. A probe is a warm-started
/// dual solve of a child that differs from the node by a single bound, so it normally
/// finishes well inside this; the cap bounds the pathological ones, whose result is
/// then simply not recorded.
pub(crate) const SB_MAX_ITERS_PER_LP: u64 = 100;

/// Total strong-branching probes allowed in one search, per integer variable. Probing
/// stops once pseudocosts are reliable, so this is a backstop for models where they
/// never settle.
pub(crate) const SB_BUDGET_PER_INT_VAR: u64 = 20;

/// A root cut round "stalls" when it improves the bound by less than this fraction of
/// `max(1, |bound|)`; two stalled rounds in a row end the loop.
pub(crate) const GOMORY_STALL_REL: f64 = 1e-5;
