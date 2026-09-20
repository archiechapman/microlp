//! The engine's tolerance model: the constants that define it and the
//! functions that derive a tolerance for one comparison from them.
//!
//! Every tolerance here is *relative to a magnitude* — the size of the
//! quantities the compared value was computed from — floored at what `f64`
//! can resolve for that magnitude. See ARCHITECTURE §7 for the contract these
//! implement and `super::Solver::var_tols` / `super::Solver::dual_tols` for
//! the per-variable and per-row instances.

/// The engine's base absolute tolerance, in its scaled units. It is the
/// absolute part of every tolerance below and the absolute floor of the pivot
/// tolerances; every comparison that involves a quantity with a magnitude
/// uses a tolerance floored at that magnitude's round-off instead of `EPS`
/// alone (see [`super::Solver::var_tols`] and [`super::Solver::dual_tols`]).
///
/// Deliberately tight for structural bounds because the big-M correctness
/// models rely on node LPs resolving basic integer values sharply onto their
/// bounds: a basic binary sitting `1e-8` off its bound is rounded by the MIP
/// layer, and a 1e9-scale big-M row then amplifies the rounding past the
/// rounded-incumbent feasibility guard. Row (slack) tolerances instead carry
/// the user's feasibility contract, see [`slack_tol`].
pub const EPS: f64 = 1e-10;

/// Relative round-off allowance: about a hundred ulps of `f64`. A quantity
/// computed from terms of magnitude `m` cannot be trusted below
/// `ROUNDOFF_FLOOR * m`, so no tolerance on it is ever tighter than that.
/// This is the same allowance the crate's own contract tests grant a row.
pub(crate) const ROUNDOFF_FLOOR: f64 = 1e-14;

/// Round-off of a single tableau entry computed through the factorization,
/// relative to the largest entry of its row or column: about ten ulps of
/// `f64`. An entry below this fraction of its neighbours is indistinguishable
/// from the round-off of computing it, and two computations of the same
/// entry (through the row and through the column) agree to about this much.
/// Tighter than [`ROUNDOFF_FLOOR`], which allows for a sum over a row's
/// many terms.
pub(crate) const ENTRY_ROUNDOFF: f64 = 1e-15;

/// Relative pivot tolerance of both ratio tests: an entry of the pivot row
/// (dual simplex) or pivot column (primal simplex) is a candidate pivot only
/// if its magnitude is at least this fraction of the largest entry scanned.
/// A pivot seven orders of magnitude below its neighbours amplifies the
/// round-off of the update by the same factor; with the matrix scaled, an
/// entry that small relative to its row or column is noise or as good as
/// noise. The largest entry is always a candidate, so this can never turn a
/// feasible model infeasible or a bounded one unbounded on its own.
pub(crate) const PIVOT_REL_TOL: f64 = 1e-7;

/// Relative disagreement between the pivot element computed through the
/// pivot row (`row_coeffs[col]`) and through the pivot column
/// (`col_coeffs[row]`) above which the factorization is not trusted for the
/// pivot: the two are the same number in exact arithmetic and agree to
/// ~1e-12 at every legitimate pivot ever measured, while a pivot on round-off
/// noise disagrees by tens of percent.
pub(crate) const PIVOT_AGREEMENT_TOL: f64 = 1e-7;

/// Iterative refinement steps [`super::Solver::verify_primal`] may apply to values
/// recomputed through a fresh factorization before giving up on the basis.
pub(crate) const REFINEMENT_STEPS: usize = 3;

/// Upper bound on the number of primal/dual phase alternations
/// [`super::Solver::run_phases`] performs before giving up loudly. A phase ends on
/// values recomputed from the factorization and verified against the rows,
/// so a second round is already rare; hitting this bound means the basis is
/// numerically unusable, which must not be reported as an optimum.
pub(crate) const MAX_PHASE_ROUNDS: usize = 8;

/// The tolerance a row of magnitude `magnitude` (`|b| + sum |a_j x_j|`) is
/// held to under an absolute tolerance `tol`: `tol` itself, but never finer
/// than the round-off of evaluating the row. Shared by the engine's slack
/// tolerances and every row check at the boundary, so all of them agree.
pub(crate) fn row_tolerance(tol: f64, magnitude: f64) -> f64 {
    tol.max(ROUNDOFF_FLOOR * magnitude)
}

/// Tolerance within which a structural var (column factor `scale`, scaled
/// bounds `min..max`) counts as at or within its bounds, floored at the
/// round-off of the bound magnitude. For a continuous var it is the user's
/// feasibility tolerance in user units. For an integer var it is `EPS` in
/// user units, and no looser than `EPS` in the engine's scaled units:
/// branch & bound closes a node only on exactly integral values, repairs a
/// near-integral one by a bound change the dual simplex must act on, and
/// rounds the value it adopts, which changes every row the var is in by the
/// coefficient times the rounding; with the matrix scaled to entries of
/// order one, `EPS` in scaled units keeps that change below the rows'
/// tolerances whatever the user-unit coefficient is.
///
/// `rounding_budget` bounds an integer var's tolerance by what its rounding
/// may change any row it is in: the row tolerance not used by the engine
/// (`(1 - ROW_BUDGET_SHARE)` of the contract) divided by the var's scaled
/// coefficient, minimised over its rows (infinite for a continuous var).
pub(crate) fn structural_tol(
    scale: f64,
    min: f64,
    max: f64,
    contract: f64,
    integer: bool,
    rounding_budget: f64,
) -> f64 {
    let magnitude = [min.abs(), max.abs()]
        .into_iter()
        .filter(|m| m.is_finite())
        .fold(0.0, f64::max);
    let base = if integer {
        (contract / scale.max(1.0)).min(rounding_budget)
    } else {
        contract / scale
    };
    base.max(ROUNDOFF_FLOOR * magnitude)
}

/// Tolerance within which a row's slack counts as at or within its bounds:
/// the user's absolute `feasibility` tolerance expressed in the row's
/// equilibrated units, floored at the round-off of evaluating the row, whose
/// `magnitude` is `|b| + sum |a_j x_j|` at the current values. This is the
/// engine's side of [`crate::Tolerances::feasibility`]: a row is held to
/// what the contract promises ([`row_tolerance`]) minus the round-off of
/// evaluating it, so that whoever re-evaluates the row from the reported
/// values in their own order of operations still finds the contract met;
/// half the round-off allowance always remains, since the engine's own
/// arithmetic needs room too. Never tighter than `f64` can resolve.
pub(crate) fn slack_tol(feasibility: f64, row_scale: f64, magnitude: f64) -> f64 {
    let round_off = ROUNDOFF_FLOOR * magnitude;
    (row_tolerance(ROW_BUDGET_SHARE * feasibility * row_scale, magnitude) - round_off)
        .max(round_off / 2.0)
}

/// The share of a row's contract tolerance the engine may use for the row's
/// own violation; the rest is reserved for the rounding of the integer vars
/// in the row (see [`structural_tol`]), so that the point the MIP layer
/// adopts, with its integer values rounded, still meets the contract.
pub(crate) const ROW_BUDGET_SHARE: f64 = 0.5;

/// Tolerance within which a reduced cost counts as zero, given the magnitude
/// `|c_j| + sum |a_ij y_i|` of the terms it was computed from.
pub(crate) fn dual_tol(magnitude: f64) -> f64 {
    EPS.max(ROUNDOFF_FLOOR * magnitude)
}

/// How often (in simplex iterations) the primal/dual loops in `optimize` and
/// `restore_feasibility` check the deadline and emit a progress `debug!` log.
/// Checking every iteration would make the deadline check itself a
/// significant fraction of the per-iteration cost on easy problems; checking
/// too rarely would make a time limit overshoot by a visible amount on hard
/// ones. 1000 keeps the check overhead negligible while still bounding the
/// worst-case overshoot to about a thousand pivots.
pub(crate) const DEADLINE_CHECK_INTERVAL: u64 = 1000;

/// Threshold-pivoting stability coefficient passed to [`crate::lu::lu_factorize`] for
/// every LU (re)factorization the simplex performs: a candidate pivot is
/// accepted only if its magnitude is at least this fraction of the column's
/// largest eligible entry. 0.1 is the standard textbook default for
/// Gilbert-Peierls sparse LU (see `lu_factorize`'s doc reference) — it
/// balances numerical stability (higher would refuse more marginal pivots,
/// at the cost of extra fill-in) against sparsity (lower risks amplifying
/// rounding error through a poorly-conditioned pivot).
pub(crate) const LU_STABILITY_THRESHOLD: f64 = 0.1;

pub(crate) fn float_eq(a: f64, b: f64) -> bool {
    (a - b).abs() < EPS
}

/// Whether `val` sits at `bound` within `tol` (the var's own tolerance, see
/// [`super::Solver::var_tols`]).
#[inline]
pub(crate) fn at_bound(val: f64, bound: f64, tol: f64) -> bool {
    (val - bound).abs() <= tol
}
