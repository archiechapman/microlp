//! Branch & bound driver for mixed-integer problems.
//!
//! Owns exactly one [`Solver`] per search. Branching changes variable bounds in
//! place.

pub(crate) mod branching;
pub(crate) mod node;
pub(crate) mod params;

use crate::presolve::{presolve, Postsolve};
use crate::solver::{check_deadline, Deadline, Solver};
use crate::{ComparisonOp, Error, OptimizationDirection, Problem, StopReason, VarDomain, Variable};
use core::time::Duration;
use node::{node_bounds, Deductions, Node};
use std::collections::BTreeMap;
use web_time::Instant;

/// Whether a usable solution is proven optimal.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SolutionStatus {
    /// The solver completed the optimality proof.
    Optimal,
    /// A valid incumbent is available, but exact optimality was not proven.
    Feasible,
}

/// Why a solve or resume call returned.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum TerminationReason {
    /// The LP or branch-and-bound optimality proof completed.
    ProvenOptimal,
    /// The configured relative MIP gap was reached before exact proof completed.
    MipGap,
    /// The wall-clock budget for this call was exhausted.
    TimeLimit,
    /// The branch-and-bound node budget for this call was exhausted.
    NodeLimit,
}

/// Options controlling a solve. Construct with [`SolveOptions::default`] and
/// mutate the fields you need.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct SolveOptions {
    /// Wall-clock budget for this call (`None` = unlimited). On expiry the search
    /// and can be resumed.
    pub time_limit: Option<Duration>,
    /// Maximum number of branch & bound nodes to solve in this call
    /// (`None` = unlimited). Deterministic alternative to `time_limit`; the
    /// budget applies per call, so [`crate::SolveOutcome::resume`] reapplies it
    /// as a fresh budget. The root relaxation does not count as a node.
    pub node_limit: Option<u64>,
    /// Relative MIP gap at which the search may stop with a feasible incumbent.
    /// Such a stop reports [`SolutionStatus::Feasible`] and
    /// [`TerminationReason::MipGap`]. Must be finite and non-negative. Default
    /// `0.0` (prove exact optimality).
    pub mip_gap: f64,
    /// Integrality tolerance: a value within this distance of an integer counts
    /// as integral. Default `1e-6`. A very loose `int_tol` mainly causes extra
    /// exact-fixing branching rather than admitting an infeasible point.
    /// Must be finite and in the half-open range `[0, 0.5)`.
    pub int_tol: f64,
    /// Optional (partial) starting assignment used to seed the incumbent.
    /// An infeasible or incomplete hint is ignored. Default `None`.
    pub warm_start: Option<Vec<(Variable, f64)>>,
    /// Edit the tolerances used by the solver, most callers
    /// should leave this at [`Tolerances::default`]; override an individual
    /// field only once you understand the correctness/permissiveness
    /// trade-off documented on it.
    pub tolerances: Tolerances,
    /// Rounds of Gomory mixed-integer cuts at the root of a MILP search. Each round
    /// derives cuts from the fractional rows of the optimal root tableau, adds them as
    /// rows and re-solves; the loop also stops when no cut is found or the bound stalls.
    /// Cuts tighten the relaxation (fewer nodes) at the cost of larger node LPs.
    /// Default `0` (no cuts).
    pub gomory_rounds: u32,
    /// Reliability threshold for strong branching. A branching candidate whose
    /// pseudocosts rest on fewer than this many observations on either side is probed
    /// by solving both of its children before the branching variable is chosen, and the
    /// probes' outcomes become pseudocost observations. Probing costs LP solves at the
    /// top of the tree and buys better branching decisions there; it stops by itself as
    /// pseudocosts become reliable. Default `0` (pure pseudocost branching).
    pub strong_branch_reliability: u32,
    /// Bound-propagation sweeps run at the root and at every node before its LP. Each
    /// sweep deduces tighter variable bounds from the rows alone (no LP), rounding
    /// integer bounds inward; deduced bounds are inherited by the node's children, and a
    /// node whose bounds cross is pruned without an LP solve. Cheap relative to an LP,
    /// but wasted on models where the rows imply nothing. Default `0` (no propagation).
    ///
    /// Note: like any change to the search order, this can change *which* of several
    /// equally optimal solutions is returned, including between an uninterrupted solve
    /// and the same solve resumed from a node limit. The objective is unaffected.
    pub propagate_rounds: u32,
    /// Stop a node's LP as soon as its bound passes the pruning cutoff, instead of
    /// solving on to an optimum the node will be pruned against anyway. Default `true`;
    /// turning it off only costs time (it exists for measurement).
    pub cutoff_prune: bool,
    /// Run a primal presolve before a mixed-integer search (see
    /// `src/presolve.rs`). Reductions use bounds and rows only, never the
    /// objective, so every feasible point survives; values and objectives are
    /// always reported for the original variables. Pure LPs ignore this
    /// option. Default `false`.
    pub presolve: bool,
}

impl Default for SolveOptions {
    fn default() -> Self {
        Self {
            time_limit: None,
            node_limit: None,
            mip_gap: 0.0,
            int_tol: 1e-6,
            warm_start: None,
            tolerances: Tolerances::default(),
            gomory_rounds: 0,
            strong_branch_reliability: 0,
            propagate_rounds: 0,
            cutoff_prune: true,
            presolve: false,
        }
    }
}

impl SolveOptions {
    pub(crate) fn validate(&self) -> Result<(), Error> {
        if !self.mip_gap.is_finite() || self.mip_gap < 0.0 {
            return Err(Error::InvalidOptions(
                "invalid SolveOptions.mip_gap: expected a finite non-negative value".to_string(),
            ));
        }
        if !self.int_tol.is_finite() || !(0.0..0.5).contains(&self.int_tol) {
            return Err(Error::InvalidOptions(
                "invalid SolveOptions.int_tol: expected a finite value in [0, 0.5)".to_string(),
            ));
        }
        if !self.tolerances.feasibility.is_finite() || self.tolerances.feasibility < 0.0 {
            return Err(Error::InvalidOptions(
                "invalid SolveOptions.tolerances.feasibility: expected a finite non-negative value"
                    .to_string(),
            ));
        }
        if !self.tolerances.integrality_rounding.is_finite()
            || !(0.0..0.5).contains(&self.tolerances.integrality_rounding)
        {
            return Err(Error::InvalidOptions(
                "invalid SolveOptions.tolerances.integrality_rounding: expected a finite value in [0, 0.5)"
                    .to_string(),
            ));
        }
        if !self.tolerances.prune_epsilon.is_finite() || self.tolerances.prune_epsilon < 0.0 {
            return Err(Error::InvalidOptions(
                "invalid SolveOptions.tolerances.prune_epsilon: expected a finite non-negative value"
                    .to_string(),
            ));
        }
        Ok(())
    }
}

/// Overrides for the solver settings used for a subsequent search/resume call.
///
/// These fields override the ones defined in the previous call to
/// [`Problem::solve`] or [`Problem::resume`].
#[derive(Clone, Debug, Default, PartialEq)]
#[non_exhaustive]
pub struct ResumeOptions {
    /// New wall-clock budget (`None` = unlimited).
    pub time_limit: Option<Duration>,
    /// New branch-and-bound node budget (`None` = unlimited).
    pub node_limit: Option<u64>,
    /// New relative MIP gap (`None` = no MIP gap / exact optimality `0.0`).
    pub mip_gap: Option<f64>,
}

impl ResumeOptions {
    pub(crate) fn validate(&self) -> Result<(), Error> {
        if let Some(mip_gap) = self.mip_gap {
            if !mip_gap.is_finite() || mip_gap < 0.0 {
                return Err(Error::InvalidOptions(
                    "invalid ResumeOptions.mip_gap: expected a finite non-negative value"
                        .to_string(),
                ));
            }
        }
        Ok(())
    }
}

/// Numeric tolerances for a solve (see [`SolveOptions::tolerances`]).
///
/// This options override the solver's tolerances when solving the problem.
/// Only edit those if you are sure of the impact of changing those values.
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub struct Tolerances {
    /// Uused to validate a rounded-to-integer
    /// candidate solution before it is accepted as the incumbent.
    /// Applied to each variable's distance
    /// outside its bounds and to each row's distance outside its feasible
    /// range. Also used, identically, by the post-edit warm-start
    /// pre-filter that decides whether a previous incumbent survives a
    /// [`crate::Solution`] edit.
    /// Must be finite and non-negative. Default `1e-7`.
    pub feasibility: f64,
    /// Distance from the nearest integer within which an integer/boolean
    /// variable's value is still treated as exactly that integer. Used by
    /// the post-edit warm-start pre-filter's integrality check.
    /// Must be finite and in the half-open range `[0, 0.5)`. Default `1e-5`.
    pub integrality_rounding: f64,
    /// Relative slack subtracted from the incumbent objective to form the
    /// branch & bound pruning cutoff: a node whose bound is not strictly
    /// better than `incumbent - max(prune_epsilon, prune_epsilon *
    /// |incumbent|)` is pruned. Guards against continuing to explore or
    /// retain nodes that could only ever match the incumbent to within
    /// float noise.
    ///
    /// Must be finite and non-negative. Default `1e-9`.
    pub prune_epsilon: f64,
}

impl Default for Tolerances {
    fn default() -> Self {
        Self {
            feasibility: 1e-7,
            integrality_rounding: 1e-5,
            prune_epsilon: 1e-9,
        }
    }
}

/// Statistics of a solve, available via [`crate::Solution::stats`].
#[derive(Clone, Copy, Debug, Default)]
#[non_exhaustive]
pub struct Stats {
    /// Branch & bound nodes whose LP was solved (0 for pure-LP problems).
    pub nodes_solved: u64,
    /// Total simplex pivots across the whole solve (including the root LP).
    pub lp_iterations: u64,
    /// Wall-clock time spent inside the solver, accumulated across resumes.
    pub elapsed: Duration,
    /// Best proven bound on the objective, in user space. `None` until an
    /// incumbent or an open node exists to derive one from.
    pub best_bound: Option<f64>,
    /// Relative gap between incumbent and best bound. `None` until both are
    /// known; `Some(0.0)` once optimality is proven.
    pub gap: Option<f64>,
    /// Cutting planes added at the root (see [`SolveOptions::gomory_rounds`]).
    pub root_cuts: u64,
    /// Root relaxation bound before any cuts, in user space (`None` for a pure LP or
    /// before the root is solved).
    pub root_lp_bound: Option<f64>,
    /// Child LPs solved for strong branching (see
    /// [`SolveOptions::strong_branch_reliability`]).
    pub strong_branch_lps: u64,
    /// Variable bounds tightened by propagation (see
    /// [`SolveOptions::propagate_rounds`]).
    pub bounds_tightened: u64,
    /// Nodes propagation proved infeasible before any LP was solved.
    pub propagation_prunes: u64,
    /// Nodes whose LP was abandoned once its bound passed the pruning cutoff.
    pub cutoff_prunes: u64,
}

/// A feasible integer assignment, in internal (minimize) objective space.
#[derive(Clone, Debug)]
pub(crate) struct Incumbent {
    /// Values of the structural variables (length = Problem var count).
    pub values: Vec<f64>,
    pub objective: f64,
}

/// The complete, resumable state of a branch & bound search.
#[derive(Clone)]
pub(crate) struct MipState {
    pub solver: Solver,
    /// Original bounds of the structural vars (to reset when jumping between nodes).
    pub root_bounds: Vec<(f64, f64)>,
    /// Bound changes currently applied to `solver` (collapsed, sorted by var).
    pub applied: Vec<(usize, f64, f64)>,
    pub open: Vec<Node>,
    /// Pop policy toggle for `pop_node`: `true` while the most recently processed
    /// node pushed children (keep plunging via LIFO pop); `false` once a dive dies
    /// out with no children pushed, or once a node is requeued unsolved by an
    /// interruption — the next pop then jumps to the open node with the best
    /// (lowest) bound instead of blindly continuing the old dive.
    pub diving: bool,
    pub incumbent: Option<Incumbent>,
    /// Sequence counter for branchings; children carry it as `parent_id`.
    pub node_seq: u64,
    /// `Some(id)` iff `solver` currently holds the optimal basis + bounds of the
    /// branching with that id — its children can skip the basis load (warm dive).
    pub last_solved_id: Option<u64>,
    pub root_solved: bool,
    pub stats: Stats,
    pub options: SolveOptions,
    pub deadline: Deadline,
    /// Consumed by `fill_bound_stats` to report `best_bound`/`gap` in user space.
    pub direction: OptimizationDirection,
    /// Learned per-variable branching degradation estimates, updated after each
    /// node LP solve and consulted by `branching::choose_branch_var`.
    pub pseudocosts: branching::PseudoCosts,
    /// Clean copy of the user's problem, including post-solve edits — never
    /// contains branching artifacts. Post-solve edits re-solve from this.
    pub base: Problem,
    /// User-level fix_var overlay on `base` (var → fixed value).
    pub fixed: BTreeMap<usize, f64>,
    /// True while a zero-objective search classifies an unbounded LP
    /// relaxation as either integer-feasible (the original MILP is unbounded)
    /// or integer-infeasible.
    pub classifying_unbounded: bool,
    /// `Some` for a [`crate::Problem::solve_enumerate`] run: candidates go to a
    /// callback instead of becoming incumbents, and nodes are pruned against a
    /// fixed objective cutoff.
    pub enumeration: Option<Enumeration>,
    /// Strong-branching probes still allowed in this search (see
    /// [`SolveOptions::strong_branch_reliability`] and `params::SB_BUDGET_PER_INT_VAR`).
    pub sb_budget: u64,
    /// Present when `solver` holds a presolved problem: maps its points back to
    /// `base`'s variables. Incumbent values are always in `base`'s space;
    /// objectives and bounds held internally exclude `postsolve.offset`.
    pub postsolve: Option<Postsolve>,
}

/// State of an enumeration run (see [`crate::Problem::solve_enumerate`]).
#[derive(Clone, Debug)]
pub(crate) struct Enumeration {
    /// Internal (minimize) space: a node or candidate strictly above this is pruned.
    /// The cutoff is fixed, so a prune is permanent, unlike incumbent pruning.
    pub cutoff: f64,
    /// Rows added through [`CandidateAction::Reject`]. They live only in the solver;
    /// `base` stays the clean user model.
    pub lazy_rows: usize,
    /// Candidates handed to the callback.
    pub candidates: u64,
    /// The callback asked to stop.
    pub stopped: bool,
}

/// What a [`crate::Problem::solve_enumerate`] callback wants done with an
/// integer candidate.
#[derive(Clone, Debug)]
pub enum CandidateAction {
    /// Add these rows and keep enumerating. They must be globally valid for the
    /// enumeration (they remove solutions everywhere in the tree, not just below
    /// the current node) and must cut off the candidate; the node is then
    /// re-solved with them in place.
    Reject(Vec<(crate::LinearExpr, ComparisonOp, f64)>),
    /// End the enumeration now.
    Stop,
}

/// Why a [`crate::Problem::solve_enumerate`] run ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EnumerateReason {
    /// The tree is exhausted: no solution within the cutoff satisfies the added
    /// rows. This is a certificate, as strong as an infeasible solve.
    Exhausted,
    /// The callback returned [`CandidateAction::Stop`].
    Stopped,
    /// The wall-clock budget ran out; the enumeration is incomplete.
    TimeLimit,
    /// The node budget ran out; the enumeration is incomplete.
    NodeLimit,
}

/// Result of a [`crate::Problem::solve_enumerate`] run.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct EnumerateOutcome {
    /// Why the run ended.
    pub reason: EnumerateReason,
    /// Search statistics (`best_bound`/`gap` are not meaningful here).
    pub stats: Stats,
    /// Rows added through [`CandidateAction::Reject`].
    pub lazy_rows: usize,
    /// Candidates passed to the callback.
    pub candidates: u64,
}

/// Callback type threaded through the search in enumeration mode.
pub(crate) type CandidateFn<'a> = dyn FnMut(&[f64], f64) -> CandidateAction + 'a;

/// Run a single-tree enumeration: see [`crate::Problem::solve_enumerate`].
pub(crate) fn run_enumerate(
    problem: &Problem,
    options: SolveOptions,
    objective_cutoff: f64,
    on_candidate: &mut CandidateFn<'_>,
) -> Result<EnumerateOutcome, Error> {
    if options.warm_start.is_some() {
        return Err(Error::InvalidOptions(
            "SolveOptions.warm_start is not supported by solve_enumerate".to_string(),
        ));
    }
    if !objective_cutoff.is_finite() {
        return Err(Error::InvalidOptions(
            "solve_enumerate: objective_cutoff must be finite".to_string(),
        ));
    }
    let mut state = match build_state(problem, options) {
        // Presolve proved there is nothing to enumerate: the same certificate an
        // exhausted tree gives.
        Err(Error::Infeasible) => {
            return Ok(EnumerateOutcome {
                reason: EnumerateReason::Exhausted,
                stats: Stats::default(),
                lazy_rows: 0,
                candidates: 0,
            })
        }
        result => result?,
    };
    let internal = match problem.direction {
        OptimizationDirection::Minimize => objective_cutoff,
        OptimizationDirection::Maximize => -objective_cutoff,
    };
    // Float noise must never prune a solution that sits exactly on the cutoff.
    let slack = state.options.tolerances.prune_epsilon * internal.abs().max(1.0);
    state.enumeration = Some(Enumeration {
        // The solver's objective excludes the constant presolve moved out.
        cutoff: internal - state.objective_offset() + slack,
        lazy_rows: 0,
        candidates: 0,
        stopped: false,
    });

    let started = Instant::now();
    let result = search_loop(&mut state, Some(on_candidate));
    state.stats.elapsed += started.elapsed();
    state.stats.lp_iterations = state.solver.lp_iterations;
    fill_bound_stats(&mut state);

    let e = state.enumeration.as_ref().expect("enumeration state");
    let reason = match result {
        _ if e.stopped => EnumerateReason::Stopped,
        Ok(TerminationReason::TimeLimit) => EnumerateReason::TimeLimit,
        Ok(TerminationReason::NodeLimit) => EnumerateReason::NodeLimit,
        // No incumbent is ever adopted, so a finished tree reports Infeasible.
        Err(Error::Infeasible) => EnumerateReason::Exhausted,
        Ok(other) => {
            return Err(Error::InternalError(format!(
                "enumeration search ended with unexpected reason {:?}",
                other
            )))
        }
        Err(e) => return Err(e),
    };
    Ok(EnumerateOutcome {
        reason,
        stats: state.stats,
        lazy_rows: e.lazy_rows,
        candidates: e.candidates,
    })
}

impl std::fmt::Debug for MipState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MipState")
            .field("open_nodes", &self.open.len())
            .field("has_incumbent", &self.incumbent.is_some())
            .field("stats", &self.stats)
            .field("diving", &self.diving)
            .field(
                "pseudocost_observations",
                &self.pseudocosts.observation_count(),
            )
            .field("classifying_unbounded", &self.classifying_unbounded)
            .finish()
    }
}

impl MipState {
    /// Objective of the solution exposed through the public value accessors.
    /// Unboundedness classification uses a zero-objective solver, so a working
    /// point without an incumbent is evaluated against the original model.
    pub(crate) fn current_objective(&self) -> f64 {
        if let Some(incumbent) = &self.incumbent {
            return incumbent.objective + self.objective_offset();
        }
        let values = self.original_values(
            &(0..self.solver.num_vars)
                .map(|v| *self.solver.get_value(v))
                .collect::<Vec<_>>(),
        );
        self.base
            .obj_coeffs
            .iter()
            .zip(&values)
            .map(|(&coefficient, &value)| coefficient * value)
            .sum()
    }

    /// Internal objective constant removed by presolve.
    pub(crate) fn objective_offset(&self) -> f64 {
        self.postsolve.as_ref().map_or(0.0, |p| p.offset)
    }

    /// `base`-space values for a point of the solver's problem.
    pub(crate) fn original_values(&self, values: &[f64]) -> Vec<f64> {
        match &self.postsolve {
            Some(postsolve) => postsolve.values(values),
            None => values.to_vec(),
        }
    }
}

#[derive(Debug)]
pub(crate) struct MipRun {
    pub reason: TerminationReason,
    pub state: MipState,
}

fn build_state(problem: &Problem, mut options: SolveOptions) -> Result<MipState, Error> {
    let deadline = options.time_limit.map(|d| Instant::now() + d);
    let base = problem.clone();
    let (problem, postsolve) = if options.presolve {
        let presolved = presolve(problem, options.tolerances.feasibility, options.int_tol)?;
        // Hints name original variables; keep the ones that survived.
        if let Some(hints) = options.warm_start.take() {
            let p = &presolved.postsolve;
            options.warm_start = Some(
                hints
                    .into_iter()
                    .filter_map(|(var, val)| p.reduced_col(var.idx()).map(|r| (Variable(r), val)))
                    .collect(),
            );
        }
        (presolved.problem, Some(presolved.postsolve))
    } else {
        (base.clone(), None)
    };
    let problem = &problem;
    let solver = problem.build_solver(deadline)?;
    let root_bounds = problem
        .var_mins
        .iter()
        .zip(&problem.var_maxs)
        .map(|(&lo, &hi)| (lo, hi))
        .collect();
    let pseudocosts = branching::PseudoCosts::new(&problem.obj_coeffs, problem.obj_coeffs.len());
    Ok(MipState {
        solver,
        root_bounds,
        applied: Vec::new(),
        open: Vec::new(),
        diving: false,
        incumbent: None,
        node_seq: 0,
        last_solved_id: None,
        root_solved: false,
        stats: Stats::default(),
        options,
        deadline,
        direction: problem.direction,
        pseudocosts,
        base,
        fixed: BTreeMap::new(),
        classifying_unbounded: false,
        enumeration: None,
        sb_budget: params::SB_BUDGET_PER_INT_VAR
            * problem
                .var_domains
                .iter()
                .filter(|d| matches!(d, VarDomain::Integer | VarDomain::Boolean))
                .count() as u64,
        postsolve,
    })
}

/// Replace a state whose original relaxation is unbounded with a
/// zero-objective integer-feasibility search. For a rational MILP, one
/// integer-feasible point plus the original relaxation ray proves
/// unboundedness; exhausting this search proves integer infeasibility.
fn begin_unbounded_classification(state: &mut MipState) -> Result<(), Error> {
    let base = state.base.clone();
    let fixed = state.fixed.clone();
    let mut feasibility = effective_problem(&base, &fixed);
    feasibility.obj_coeffs.fill(0.0);

    let deadline = state.deadline;
    let elapsed = state.stats.elapsed;
    let lp_iterations = state.solver.lp_iterations;
    let mut replacement = build_state(&feasibility, state.options.clone())?;
    replacement.deadline = deadline;
    replacement.solver.deadline = deadline;
    replacement.solver.lp_iterations = lp_iterations;
    replacement.stats.elapsed = elapsed;
    replacement.base = base;
    replacement.fixed = fixed;
    replacement.classifying_unbounded = true;
    *state = replacement;
    Ok(())
}

fn resume_or_classify(state: &mut MipState) -> Result<TerminationReason, Error> {
    match resume_run_with_deadline(state) {
        Err(Error::Unbounded) if !state.classifying_unbounded => {
            begin_unbounded_classification(state)?;
            resume_run_with_deadline(state)
        }
        result => result,
    }
}

/// Build the search state for `problem` and run it under `options`.
pub(crate) fn run(problem: &Problem, options: SolveOptions) -> Result<MipRun, Error> {
    let mut state = build_state(problem, options)?;
    let reason = resume_or_classify(&mut state)?;
    Ok(MipRun { reason, state })
}

/// `base` with the fix_var overlay applied to the variable bounds.
pub(crate) fn effective_problem(base: &Problem, fixed: &BTreeMap<usize, f64>) -> Problem {
    let mut p = base.clone();
    for (&v, &val) in fixed {
        p.var_mins[v] = val;
        p.var_maxs[v] = val;
    }
    p
}

fn candidate_variables_feasible(
    values: &[f64],
    domains: &[VarDomain],
    tolerances: &Tolerances,
    mut bounds: impl FnMut(usize) -> (f64, f64),
) -> bool {
    if values.len() != domains.len() {
        return false;
    }
    values.iter().enumerate().all(|(v, &value)| {
        let (lo, hi) = bounds(v);
        value.is_finite()
            && !lo.is_nan()
            && !hi.is_nan()
            && lo <= hi
            && value >= lo - tolerances.feasibility
            && value <= hi + tolerances.feasibility
            && (!matches!(domains[v], VarDomain::Integer | VarDomain::Boolean)
                || (value - value.round()).abs() <= tolerances.integrality_rounding)
    })
}

/// Cheap feasibility check of a value vector against base + fixes: bounds,
/// domains, and every user-scale constraint row. This is a warm-start prefilter;
/// adoption re-validates the candidate against the active solver's scaled rows.
pub(crate) fn incumbent_feasible(
    base: &Problem,
    fixed: &BTreeMap<usize, f64>,
    values: &[f64],
    tolerances: &Tolerances,
) -> bool {
    if !candidate_variables_feasible(values, &base.var_domains, tolerances, |v| {
        fixed
            .get(&v)
            .map_or((base.var_mins[v], base.var_maxs[v]), |&value| {
                (value, value)
            })
    }) {
        return false;
    }
    for (coeffs, op, rhs) in &base.constraints {
        let lhs: f64 = coeffs.iter().map(|(i, c)| c * values[i]).sum();
        if !lhs.is_finite() {
            return false;
        }
        let tol = tolerances.feasibility;
        let ok = match op {
            ComparisonOp::Eq => (lhs - rhs).abs() <= tol,
            ComparisonOp::Le => lhs <= rhs + tol,
            ComparisonOp::Ge => lhs >= *rhs - tol,
        };
        if !ok {
            return false;
        }
    }
    true
}

/// After a user edit: drop the open tree, carry the incumbent as a warm-start
/// hint when it survives the edit, and re-run the search on base + fixes.
/// The fresh run gets the state's original options (incl. a fresh time budget).
pub(crate) fn reedit_and_resolve(state: Box<MipState>) -> Result<MipRun, Error> {
    let MipState {
        base,
        fixed,
        incumbent,
        mut options,
        ..
    } = *state;

    options.warm_start = incumbent
        .filter(|inc| incumbent_feasible(&base, &fixed, &inc.values, &options.tolerances))
        .map(|inc| {
            inc.values
                .iter()
                .enumerate()
                .map(|(v, &val)| (Variable(v), val))
                .collect()
        });

    let effective = effective_problem(&base, &fixed);
    let mut run = run(&effective, options)?;
    // `run` cloned `effective` as its base; restore the true base/fixed split so
    // later edits keep composing against the user's problem.
    run.state.base = base;
    run.state.fixed = fixed;
    Ok(run)
}

/// Continue a paused search with fresh per-call budgets and options.
pub(crate) fn resume_run(
    state: &mut MipState,
    options: ResumeOptions,
) -> Result<TerminationReason, Error> {
    options.validate()?;
    state.deadline = options.time_limit.map(|d| Instant::now() + d);
    state.options.time_limit = options.time_limit;
    state.options.node_limit = options.node_limit;
    state.options.mip_gap = options.mip_gap.unwrap_or(0.0);
    resume_or_classify(state)
}

fn resume_run_with_deadline(state: &mut MipState) -> Result<TerminationReason, Error> {
    let started = Instant::now();
    let res = search_loop(state, None);
    state.stats.elapsed += started.elapsed();
    state.stats.lp_iterations = state.solver.lp_iterations;
    fill_bound_stats(state);
    res
}

/// Best proven lower bound (internal space) on the optimum: the min over open-node
/// bounds and the incumbent. `None` while nothing is known (no nodes, no incumbent).
/// Only valid BETWEEN nodes (a popped node's subtree is otherwise unaccounted).
fn global_bound_internal(state: &MipState) -> Option<f64> {
    if !state.root_solved {
        return None;
    }
    let open_min = state
        .open
        .iter()
        .map(|n| n.lp_bound)
        .fold(f64::INFINITY, f64::min);
    match (&state.incumbent, state.open.is_empty()) {
        (Some(inc), true) => Some(inc.objective), // proof complete
        (Some(inc), false) => Some(open_min.min(inc.objective)),
        (None, false) => Some(open_min),
        (None, true) => None,
    }
}

/// Relative gap between incumbent and bound, internal space (0.0 when they meet).
fn relative_gap(incumbent_obj: f64, bound: f64) -> f64 {
    (incumbent_obj - bound).max(0.0) / incumbent_obj.abs().max(params::GAP_DENOM_GUARD)
}

fn to_user_space(direction: OptimizationDirection, internal: f64) -> f64 {
    match direction {
        OptimizationDirection::Minimize => internal,
        OptimizationDirection::Maximize => -internal,
    }
}

fn fill_bound_stats(state: &mut MipState) {
    if state.classifying_unbounded {
        state.stats.best_bound = None;
        state.stats.gap = None;
        return;
    }
    let bound = global_bound_internal(state);
    let offset = state.objective_offset();
    state.stats.best_bound = bound.map(|b| to_user_space(state.direction, b + offset));
    state.stats.gap = match (&state.incumbent, bound) {
        (Some(inc), Some(b)) => Some(relative_gap(inc.objective + offset, b + offset)),
        _ => None,
    };
}

/// Prune threshold: a node whose lower bound is ≥ this cannot improve the incumbent.
fn cutoff(incumbent_obj: f64, prune_epsilon: f64) -> f64 {
    incumbent_obj - f64::max(prune_epsilon, prune_epsilon * incumbent_obj.abs())
}

/// Validate and adopt the solver's current solution using integer-rounded
/// values. `Ok(false)` means rounding produced an invalid candidate and the
/// caller must branch. A valid candidate completes unboundedness classification.
fn try_adopt_incumbent(state: &mut MipState) -> Result<bool, Error> {
    let tolerances = &state.options.tolerances;
    let solver = &state.solver;
    let n = solver.num_vars;
    let domains = &solver.orig_var_domains;
    let mut values: Vec<f64> = (0..n).map(|v| *solver.get_value(v)).collect();
    for (val, dom) in values.iter_mut().zip(domains.iter()) {
        if matches!(dom, VarDomain::Integer | VarDomain::Boolean) {
            *val = val.round();
        }
    }
    if !candidate_variables_feasible(&values, domains, tolerances, |v| state.root_bounds[v])
        || !solver.check_constraints(&values, tolerances.feasibility)
    {
        debug!("integral-within-tol solution rejected: rounded values infeasible");
        return Ok(false);
    }
    let objective = solver.objective_of(&values);
    if !objective.is_finite() {
        debug!("integral-within-tol solution rejected: objective is non-finite");
        return Ok(false);
    }
    let values = match &state.postsolve {
        None => values,
        Some(postsolve) => {
            let mut original = postsolve.values(&values);
            for (val, dom) in original.iter_mut().zip(&state.base.var_domains) {
                if matches!(dom, VarDomain::Integer | VarDomain::Boolean)
                    && (*val - val.round()).abs() <= tolerances.integrality_rounding
                {
                    *val = val.round();
                }
            }
            if !incumbent_feasible(&state.base, &state.fixed, &original, tolerances) {
                debug!("integral-within-tol solution rejected: postsolved values infeasible");
                return Ok(false);
            }
            original
        }
    };
    let better = match &state.incumbent {
        Some(inc) => objective < inc.objective,
        None => true,
    };
    if better {
        debug!("new incumbent, internal obj: {:.6}", objective);
        state.incumbent = Some(Incumbent { values, objective });
    }
    if state.classifying_unbounded {
        Err(Error::Unbounded)
    } else {
        Ok(true)
    }
}

enum IntegralCandidate {
    Closed,
    Branch(usize),
    Limit,
    /// Enumeration only: the callback asked to stop.
    Stop,
}

/// The enumeration cutoff (internal space), if this is an enumeration run.
fn enumeration_cutoff(state: &MipState) -> Option<f64> {
    state.enumeration.as_ref().map(|e| e.cutoff)
}

/// Enumeration mode's counterpart of [`process_integral_candidate`]: the solver
/// holds the current node's optimal LP point, integral within `int_tol`.
///
/// The rounded point goes through the same validation funnel as an incumbent and
/// is then handed to the callback, never adopted. On `Reject`, the rows are added
/// to the live solver and the SAME node is re-solved, repeatedly while the
/// re-solved point stays integral: it may become fractional (branch), infeasible
/// or worse than the cutoff (close), or yield the next candidate. Rows are added
/// before any branching, so a rejected point can never reappear in a child.
fn enumerate_candidate(
    state: &mut MipState,
    domains: &[VarDomain],
    int_tol: f64,
    on_candidate: &mut CandidateFn<'_>,
) -> Result<IntegralCandidate, Error> {
    let cutoff = enumeration_cutoff(state).expect("enumeration run");
    let mut retried = false;
    loop {
        let feasibility = state.options.tolerances.feasibility;
        let solver = &state.solver;
        let mut values: Vec<f64> = (0..solver.num_vars).map(|v| *solver.get_value(v)).collect();
        for (val, dom) in values.iter_mut().zip(&solver.orig_var_domains) {
            if matches!(dom, VarDomain::Integer | VarDomain::Boolean) {
                *val = val.round();
            }
        }
        let valid = candidate_variables_feasible(
            &values,
            &solver.orig_var_domains,
            &state.options.tolerances,
            |v| state.root_bounds[v],
        ) && solver.check_constraints(&values, feasibility);
        let objective = solver.objective_of(&values);
        // The callback sees original variables; they must pass the same validation.
        let original = match (&state.postsolve, valid) {
            (Some(postsolve), true) => {
                let mut original = postsolve.values(&values);
                for (val, dom) in original.iter_mut().zip(&state.base.var_domains) {
                    if matches!(dom, VarDomain::Integer | VarDomain::Boolean)
                        && (*val - val.round()).abs() <= state.options.tolerances.integrality_rounding
                    {
                        *val = val.round();
                    }
                }
                incumbent_feasible(&state.base, &state.fixed, &original, &state.options.tolerances)
                    .then_some(original)
            }
            _ => valid.then(|| values.clone()),
        };
        let valid = original.is_some();
        if !valid || !objective.is_finite() || objective > cutoff {
            // Rounding moved the point (infeasible or beyond the cutoff): resolve the
            // below-tolerance fractionality by branching, as the normal search does.
            if let Some(var) =
                branching::choose_branch_var(solver, domains, 0.0, &state.pseudocosts)
            {
                return Ok(IntegralCandidate::Branch(var));
            }
            if valid && objective.is_finite() {
                // Exactly integral and feasible, but above the cutoff by float noise
                // (the node's LP value was within it): not a candidate.
                return Ok(IntegralCandidate::Closed);
            }
            // Exactly integral yet failing the guard: eta drift, as in
            // `process_integral_candidate`. Closing the node here could lose
            // solutions, so re-solve it once from the slack basis, then fail loudly.
            if retried {
                return Err(Error::InternalError(
                    "enumeration: exactly integral candidate failed validation after slack-basis retry"
                        .to_string(),
                ));
            }
            retried = true;
            debug!("enumeration candidate failed guard; retrying from slack basis");
            let slack = state.solver.slack_basis();
            state
                .solver
                .load_basis(&slack)
                .map_err(|e| Error::InternalError(format!("slack basis load failed: {}", e)))?;
            match solve_node_lp(state)? {
                NodeLp::Solved => {}
                NodeLp::Infeasible => {
                    return Err(Error::InternalError(
                        "enumeration: integral candidate became infeasible after slack-basis retry"
                            .to_string(),
                    ))
                }
                NodeLp::Limit => return Ok(IntegralCandidate::Limit),
            }
            if state.solver.cur_obj_val > cutoff {
                return Ok(IntegralCandidate::Closed);
            }
            if !branching::is_integral(&state.solver, domains, int_tol) {
                return Ok(
                    match branching::choose_branch_var(
                        &state.solver,
                        domains,
                        int_tol,
                        &state.pseudocosts,
                    ) {
                        Some(var) => IntegralCandidate::Branch(var),
                        None => IntegralCandidate::Closed,
                    },
                );
            }
            continue;
        }

        let user_objective = to_user_space(state.direction, objective + state.objective_offset());
        let enumeration = state.enumeration.as_mut().expect("enumeration run");
        enumeration.candidates += 1;
        let rows = match on_candidate(original.as_deref().expect("validated"), user_objective) {
            CandidateAction::Stop => {
                enumeration.stopped = true;
                return Ok(IntegralCandidate::Stop);
            }
            CandidateAction::Reject(rows) => rows,
        };

        let num_vars = state.solver.num_vars;
        let mut prepared = Vec::with_capacity(rows.len());
        for (expr, op, rhs) in rows {
            let (vars, coeffs, rhs) = match &state.postsolve {
                None => (expr.vars, expr.coeffs, rhs),
                Some(postsolve) => {
                    let terms: Vec<(usize, f64)> =
                        expr.vars.iter().copied().zip(expr.coeffs.iter().copied()).collect();
                    if terms.iter().any(|&(v, _)| v >= state.base.obj_coeffs.len()) {
                        return Err(Error::InvalidOperation(
                            "solve_enumerate: row references an unknown variable".to_string(),
                        ));
                    }
                    let (terms, rhs) = postsolve.forward_row(&terms, rhs);
                    let (vars, coeffs) = terms.into_iter().unzip();
                    (vars, coeffs, rhs)
                }
            };
            let coeffs = crate::CsVec::new_from_unsorted(num_vars, vars, coeffs)
                .map_err(|error| Error::InvalidOperation(error.2.to_string()))?;
            prepared.push((coeffs, op, rhs));
        }
        let added = state.solver.append_rows(prepared, false)?;
        state.enumeration.as_mut().expect("enumeration run").lazy_rows += added;
        if state.solver.check_constraints(&values, feasibility) {
            return Err(Error::InvalidOperation(
                "solve_enumerate: the rows of a Reject must cut off the candidate".to_string(),
            ));
        }

        let (node_lp, cut_off) = solve_node_lp_bounded(state)?;
        if cut_off {
            state.stats.cutoff_prunes += 1;
            return Ok(IntegralCandidate::Closed);
        }
        match node_lp {
            NodeLp::Solved => {}
            NodeLp::Infeasible => return Ok(IntegralCandidate::Closed),
            NodeLp::Limit => return Ok(IntegralCandidate::Limit),
        }
        if state.solver.cur_obj_val > cutoff {
            return Ok(IntegralCandidate::Closed);
        }
        if !branching::is_integral(&state.solver, domains, int_tol) {
            return Ok(
                match branching::choose_branch_var(&state.solver, domains, int_tol, &state.pseudocosts)
                {
                    Some(var) => IntegralCandidate::Branch(var),
                    None => IntegralCandidate::Closed,
                },
            );
        }
    }
}

/// Adopt a feasible rounded candidate, but close the current subtree only when
/// the LP point itself is exactly integral. If an exactly integral point fails
/// the independent feasibility guard, retry once from the all-slack basis:
/// large coefficients can leave an eta-updated continuous value just outside
/// the absolute guard even though a clean factorization recovers the vertex.
fn process_integral_candidate(
    state: &mut MipState,
    domains: &[VarDomain],
    int_tol: f64,
) -> Result<IntegralCandidate, Error> {
    let adopted = try_adopt_incumbent(state)?;
    if let Some(var) = branching::choose_branch_var(&state.solver, domains, 0.0, &state.pseudocosts)
    {
        return Ok(IntegralCandidate::Branch(var));
    }
    if adopted {
        return Ok(IntegralCandidate::Closed);
    }

    debug!("exactly integral candidate failed guard; retrying from slack basis");
    let slack = state.solver.slack_basis();
    state
        .solver
        .load_basis(&slack)
        .map_err(|e| Error::InternalError(format!("slack basis load failed: {}", e)))?;
    match solve_node_lp(state)? {
        NodeLp::Limit => return Ok(IntegralCandidate::Limit),
        NodeLp::Infeasible => {
            return Err(Error::InternalError(
                "integral candidate became infeasible after slack-basis retry".to_string(),
            ))
        }
        NodeLp::Solved => {}
    }

    if !branching::is_integral(&state.solver, domains, int_tol) {
        return branching::choose_branch_var(&state.solver, domains, int_tol, &state.pseudocosts)
            .map(IntegralCandidate::Branch)
            .ok_or_else(|| {
                Error::InternalError(
                    "slack-basis retry produced a non-integral point with no branchable variable"
                        .to_string(),
                )
            });
    }

    let adopted = try_adopt_incumbent(state)?;
    if let Some(var) = branching::choose_branch_var(&state.solver, domains, 0.0, &state.pseudocosts)
    {
        Ok(IntegralCandidate::Branch(var))
    } else if adopted {
        Ok(IntegralCandidate::Closed)
    } else {
        Err(Error::InternalError(
            "exactly integral solution failed feasibility validation after slack-basis retry"
                .to_string(),
        ))
    }
}

/// Record bounds deduced by propagation at `node`: chain them onto the node so its
/// children inherit them (see [`Deductions`]), and mirror them in `state.applied`,
/// which is what the next node diffs against to reset them.
fn record_propagated(state: &mut MipState, node: &mut Node, changes: Vec<(usize, f64, f64)>) {
    state.stats.bounds_tightened += changes.len() as u64;
    for &(var, lo, hi) in &changes {
        match state.applied.binary_search_by_key(&var, |t| t.0) {
            Ok(i) => state.applied[i] = (var, lo, hi),
            Err(i) => state.applied.insert(i, (var, lo, hi)),
        }
    }
    node.deduced = Deductions::chain(node.deduced.take(), changes);
}

/// Apply `node`'s bounds to the solver, diffing against what is currently applied.
/// Returns false (node pruned, solver untouched) if the node's bounds cross.
fn apply_node_bounds(state: &mut MipState, node: &Node) -> bool {
    let target = node_bounds(&node.bound_changes, &node.deduced);
    if target.iter().any(|&(_, lo, hi)| lo > hi) {
        return false;
    }
    // Reset vars that are currently changed but absent from the target.
    for &(v, _, _) in &state.applied {
        if target.binary_search_by_key(&v, |t| t.0).is_err() {
            let (rlo, rhi) = state.root_bounds[v];
            state
                .solver
                .set_var_bounds(v, rlo, rhi)
                .expect("root bounds cannot cross");
        }
    }
    // Apply the target bounds (validated above, cannot fail).
    for &(v, lo, hi) in &target {
        state
            .solver
            .set_var_bounds(v, lo, hi)
            .expect("validated bounds cannot cross");
    }
    state.applied = target;
    true
}

/// Branch on `var` at the solver's current (just solved) optimum: push the two
/// children carrying the parent's basis and objective bound.
fn branch(state: &mut MipState, parent: &Node, var: usize) {
    let z = state.solver.cur_obj_val;
    let val = *state.solver.get_value(var);
    let (lo, hi) = state.solver.get_var_bounds(var);
    // The split point k (children: x ≤ k and x ≥ k + 1) must be
    // noise-robust: a raw `val.floor()` of a within-tolerance-integral value
    // is catastrophic — floor(−8e-16) = −1 makes the up child
    // (max(0, lo), hi) reproduce the parent VERBATIM and the search
    // descends forever. Reachable through the rounding-rejected re-branch
    // path (`choose_branch_var` at int_tol = 0) whenever LP noise puts an
    // integer var a hair below an integer. Snap near-integral values to
    // their integer first, then clamp k into [lo, hi − 1] so BOTH children
    // strictly tighten the parent's [lo, hi] whenever hi − lo ≥ 1
    // (integral bounds — guaranteed for branchable vars).
    let floor = {
        let near = val.round();
        let k = if (val - near).abs() <= state.options.int_tol {
            near
        } else {
            val.floor()
        };
        k.clamp(lo, (hi - 1.0).max(lo))
    };
    let f_down = (val - floor).clamp(0.0, 1.0);

    state.node_seq += 1;
    let id = state.node_seq;
    state.last_solved_id = Some(id);
    let basis = state.solver.snapshot_basis();

    let mut down_changes = parent.bound_changes.clone();
    down_changes.push((var, lo, floor));
    let mut up_changes = parent.bound_changes.clone();
    up_changes.push((var, floor + 1.0, hi));

    let down_node = Node {
        bound_changes: down_changes,
        basis: basis.clone(),
        lp_bound: z,
        depth: parent.depth + 1,
        parent_id: id,
        branch_var: Some(var),
        branch_up: false,
        branch_frac: f_down,
        deduced: parent.deduced.clone(),
        propagated: false,
    };
    let up_node = Node {
        bound_changes: up_changes,
        basis,
        lp_bound: z,
        depth: parent.depth + 1,
        parent_id: id,
        branch_var: Some(var),
        branch_up: true,
        branch_frac: 1.0 - f_down,
        deduced: parent.deduced.clone(),
        propagated: false,
    };

    // Estimate-ordered dive: push the child with the LARGER estimated degradation
    // first, so the cheaper (more promising) direction is popped/dived first.
    let est_down = state.pseudocosts.estimate(var, false) * f_down;
    let est_up = state.pseudocosts.estimate(var, true) * (1.0 - f_down);
    if est_down > est_up {
        // up is cheaper → push it last so it is dived first
        state.open.push(down_node);
        state.open.push(up_node);
    } else {
        state.open.push(up_node);
        state.open.push(down_node);
    }
    // Children were pushed: keep plunging (LIFO pop) into this subtree.
    state.diving = true;
}

/// Outcome of solving one branch & bound node's LP relaxation.
enum NodeLp {
    /// The solver now holds this node's optimal basis and objective.
    Solved,
    /// The node's LP is infeasible under its current bounds.
    Infeasible,
    /// A limit interrupted the solve (possibly during the slack retry); the
    /// solver's unfinished, non-optimal state must not be used as a node result.
    Limit,
}

/// Solve the current node's LP relaxation. Bounds and (if needed) the warm basis
/// are already loaded into the solver by the caller.
///
/// Robustness valve: if the first `reoptimize` fails with an internal error class
/// — anything that is neither [`Error::Infeasible`] nor [`Error::Unbounded`], e.g.
/// a singular LU produced by numerical degradation during pivoting — fall back
/// ONCE to the all-slack basis (which is documented to always load) and re-solve
/// the node from scratch, then take that retry's outcome as final. The retry
/// cannot loop: it is attempted at most once and its own internal error is
/// propagated rather than retried again.
/// The objective beyond which this node cannot hold anything useful: the incumbent's
/// pruning cutoff, an enumeration's fixed cutoff, or the tighter of the two.
fn node_cutoff(state: &MipState) -> Option<f64> {
    if !state.options.cutoff_prune {
        return None;
    }
    let incumbent = state
        .incumbent
        .as_ref()
        .map(|inc| cutoff(inc.objective, state.options.tolerances.prune_epsilon));
    match (incumbent, enumeration_cutoff(state)) {
        (Some(a), Some(b)) => Some(a.min(b)),
        (a, b) => a.or(b),
    }
}

/// [`solve_node_lp`] with the node's cutoff armed: the dual simplex may stop as soon as
/// the bound passes it (see [`Solver::set_objective_cutoff`]). The flag says the solve
/// stopped that way, so the node is pruned and its objective is not a solved LP value.
fn solve_node_lp_bounded(state: &mut MipState) -> Result<(NodeLp, bool), Error> {
    state.solver.set_objective_cutoff(node_cutoff(state));
    let result = solve_node_lp(state);
    let cut_off = state.solver.cutoff_reached();
    state.solver.set_objective_cutoff(None);
    Ok((result?, cut_off))
}

fn solve_node_lp(state: &mut MipState) -> Result<NodeLp, Error> {
    state.solver.deadline = state.deadline;
    let err = match state.solver.reoptimize() {
        Ok(StopReason::Finished) => return Ok(NodeLp::Solved),
        Ok(StopReason::Limit) => return Ok(NodeLp::Limit),
        Err(Error::Infeasible) => return Ok(NodeLp::Infeasible),
        Err(Error::Unbounded) => {
            return Err(Error::InternalError(
                "bounded B&B node reported unbounded".to_string(),
            ))
        }
        // Internal/singular error class: fall through to the one-shot slack retry.
        Err(e) => e,
    };

    debug!(
        "node LP reoptimize failed ({}); retrying from slack basis",
        err
    );
    let slack = state.solver.slack_basis();
    state
        .solver
        .load_basis(&slack)
        .map_err(|e| Error::InternalError(format!("slack basis load failed: {}", e)))?;
    match state.solver.reoptimize() {
        Ok(StopReason::Finished) => Ok(NodeLp::Solved),
        Ok(StopReason::Limit) => Ok(NodeLp::Limit),
        Err(Error::Infeasible) => Ok(NodeLp::Infeasible),
        Err(Error::Unbounded) => Err(Error::InternalError(
            "bounded B&B node reported unbounded".to_string(),
        )),
        // The retry also failed internally: propagate the error this time.
        Err(e) => Err(e),
    }
}

/// Pop policy: keep diving (DFS) while the last processed node produced children;
/// when a dive dies out, jump to the open node with the best (lowest) bound. Ties
/// in `lp_bound` resolve to the first (lowest-index) such node in `open` — the
/// scan uses strict `<`, so `best` only moves for a strictly smaller bound.
fn pop_node(state: &mut MipState) -> Option<Node> {
    if state.open.is_empty() {
        return None;
    }
    if state.diving {
        state.open.pop()
    } else {
        let mut best = 0;
        for (i, n) in state.open.iter().enumerate() {
            if n.lp_bound < state.open[best].lp_bound {
                best = i;
            }
        }
        Some(state.open.swap_remove(best))
    }
}

/// Evaluate a warm-start hint: fix hinted vars, LP-complete the rest, and if the
/// completion is feasible and integral adopt it as the initial incumbent.
/// Advisory by design — every failure path just drops the hint. Always restores
/// the solver to the root optimum before returning.
///
/// Returns `Ok(None)` on the normal path (the caller proceeds to build the root
/// node). Returns `Ok(Some(TerminationReason::TimeLimit))` only when the restore
/// of the root basis fails AND the deadline strikes mid-restore: rather than let the
/// caller read an unfinished `cur_obj_val` as the root bound, it un-sets
/// `root_solved` so a resume re-enters `initial_solve` and continues honestly from
/// the solver's feasibility flags.
fn try_warm_start(
    state: &mut MipState,
    hints: &[(crate::Variable, f64)],
) -> Result<Option<TerminationReason>, Error> {
    let domains = state.solver.orig_var_domains.clone();
    let root_basis = state.solver.snapshot_basis();
    let mut applied: Vec<usize> = Vec::new();
    let mut ok = true;
    let mut pending_error = None;

    for &(var, val) in hints {
        let v = var.idx();
        if v >= state.solver.num_vars || !val.is_finite() {
            ok = false;
            break;
        }
        let val = if matches!(
            domains.get(v),
            Some(VarDomain::Integer | VarDomain::Boolean)
        ) {
            val.round()
        } else {
            val
        };
        let (lo, hi) = state.root_bounds[v];
        if val < lo - params::HINT_BOUNDS_SLACK || val > hi + params::HINT_BOUNDS_SLACK {
            ok = false;
            break;
        }
        state
            .solver
            .set_var_bounds(v, val, val)
            .expect("fixing to [val, val] cannot cross");
        applied.push(v);
    }

    if ok {
        match state.solver.reoptimize() {
            Ok(StopReason::Finished) => {
                if branching::is_integral(&state.solver, &domains, state.options.int_tol) {
                    // Rounded-incumbent feasibility guard: never bypass it for a hint.
                    // If it rejects the completion, drop the hint — do not branch
                    // below-tolerance vars here, that fallback is only for the main
                    // search loop.
                    match try_adopt_incumbent(state) {
                        Ok(true) => {}
                        Ok(false) => {
                            debug!("warm-start hint rejected by feasibility guard; ignored");
                        }
                        Err(error) => pending_error = Some(error),
                    }
                } else {
                    debug!("warm-start hint LP-completed fractionally; ignored");
                }
            }
            Ok(StopReason::Limit) | Err(Error::Infeasible) => {
                debug!("warm-start hint infeasible or out of time; ignored");
            }
            Err(error) => pending_error = Some(error),
        }
    } else {
        debug!(
            "warm-start hint invalid (unknown variable, non-finite value, or out of bounds); ignored"
        );
    }

    // Restore the root state exactly: bounds back, then the optimal root basis
    // (load_basis recomputes everything, discarding the hint solve).
    for v in applied {
        let (lo, hi) = state.root_bounds[v];
        state
            .solver
            .set_var_bounds(v, lo, hi)
            .expect("root bounds cannot cross");
    }
    if state.solver.load_basis(&root_basis).is_err() {
        let slack = state.solver.slack_basis();
        state
            .solver
            .load_basis(&slack)
            .map_err(|e| Error::InternalError(format!("slack basis load failed: {}", e)))?;
        if state.solver.reoptimize()? == StopReason::Limit {
            // A limit during the root re-solve leaves `cur_obj_val` unsuitable
            // as a root bound. Mark the root unsolved and return Interrupted;
            // `initialize_root` continues `initial_solve` from the solver's
            // feasibility flags on resume.
            state.root_solved = false;
            if let Some(error) = pending_error {
                return Err(error);
            }
            return Ok(Some(TerminationReason::TimeLimit));
        }
    }
    if let Some(error) = pending_error {
        return Err(error);
    }
    Ok(None)
}

/// Root cutting-plane loop: up to `options.gomory_rounds` rounds of Gomory mixed-integer
/// cuts ([`Solver::gomory_cuts`]), each added as rows and re-optimised. Stops early when
/// the root point is integral, no cut is found, or the bound stalls. The cuts are derived
/// at the root bounds, so they are globally valid and stay in the solver for the whole
/// search (they are not part of `base`, and candidate validation ignores them).
///
/// Returns `Limit` if the deadline interrupted a re-solve.
///
/// If a round makes the relaxation infeasible, all cuts are discarded (the solver is
/// rebuilt from the model and the root re-solved). Valid cuts can only do that when no
/// integer point exists, which the search then proves without them; but with the
/// engine's tight tolerances it can also be round-off, and an unverified `Infeasible`
/// must never stand as a proof.
fn root_cuts(state: &mut MipState, domains: &[VarDomain]) -> Result<StopReason, Error> {
    match root_cut_rounds(state, domains) {
        Err(Error::Infeasible) => {
            debug!("root cuts made the relaxation infeasible; discarding them");
            let lp_iterations = state.solver.lp_iterations;
            let problem = effective_problem(&state.base, &state.fixed);
            state.solver = problem.build_solver(state.deadline)?;
            state.solver.lp_iterations = lp_iterations;
            state.stats.root_cuts = 0;
            state.solver.initial_solve()
        }
        result => result,
    }
}

fn root_cut_rounds(state: &mut MipState, domains: &[VarDomain]) -> Result<StopReason, Error> {
    let int_tol = state.options.int_tol;
    let rows = state.solver.num_constraints();
    let per_round = (rows / params::GOMORY_ROWS_PER_CUT_ROUND).clamp(
        params::GOMORY_MIN_CUTS_PER_ROUND,
        params::GOMORY_MAX_CUTS_PER_ROUND,
    );
    let mut budget = (rows / params::GOMORY_ROWS_PER_CUT_TOTAL)
        .max(params::GOMORY_MIN_CUTS_PER_ROUND);
    let mut stalled = 0;
    for round in 0..state.options.gomory_rounds {
        if budget == 0 || branching::is_integral(&state.solver, domains, int_tol) {
            break;
        }
        let before = state.solver.cur_obj_val;
        let cuts = state.solver.gomory_cuts(per_round.min(budget));
        if cuts.is_empty() {
            break;
        }
        budget -= cuts.len();
        let rows = cuts
            .into_iter()
            .map(|(coeffs, rhs)| (coeffs, ComparisonOp::Ge, rhs))
            .collect();
        let added = state.solver.append_rows(rows, true)?;
        state.stats.root_cuts += added as u64;
        if state.solver.reoptimize()? == StopReason::Limit {
            return Ok(StopReason::Limit);
        }
        let gain = state.solver.cur_obj_val - before;
        debug!(
            "root cut round {}: {} cuts, bound {:.6} (+{:.3e})",
            round + 1,
            added,
            state.solver.cur_obj_val,
            gain
        );
        if gain <= params::GOMORY_STALL_REL * before.abs().max(1.0) {
            stalled += 1;
            if stalled >= 2 {
                break;
            }
        } else {
            stalled = 0;
        }
    }
    Ok(StopReason::Finished)
}

/// Solve or resume the root relaxation, restore any advisory warm start, and
/// either close the problem or seed the open tree. `None` means node processing
/// can begin; `Some` is a completed or interrupted root outcome.
fn initialize_root(
    state: &mut MipState,
    domains: &[VarDomain],
    on_candidate: Option<&mut CandidateFn<'_>>,
) -> Result<Option<TerminationReason>, Error> {
    if state.root_solved {
        return Ok(None);
    }

    if state.solver.initial_solve()? == StopReason::Limit {
        return Ok(Some(TerminationReason::TimeLimit));
    }
    state.root_solved = true;

    if let Some(hints) = state.options.warm_start.take() {
        if let Some(outcome) = try_warm_start(state, &hints)? {
            return Ok(Some(outcome));
        }
    }

    state.stats.root_lp_bound = Some(to_user_space(
        state.direction,
        state.solver.cur_obj_val + state.objective_offset(),
    ));

    // Root propagation runs before the cuts, so they are derived from the tighter model.
    if state.options.propagate_rounds > 0 && !state.classifying_unbounded {
        let propagation = state
            .solver
            .propagate_bounds(state.options.propagate_rounds)?;
        state.stats.bounds_tightened += propagation.changes.len() as u64;
        // Deductions at the root hold everywhere, so they become the root bounds every
        // node resets to — no node has to carry them.
        let tightened = !propagation.changes.is_empty();
        for (var, lo, hi) in propagation.changes {
            state.root_bounds[var] = (lo, hi);
        }
        if !propagation.feasible {
            return Err(Error::Infeasible);
        }
        if tightened && state.solver.reoptimize()? == StopReason::Limit {
            state.root_solved = false;
            return Ok(Some(TerminationReason::TimeLimit));
        }
    }

    if state.options.gomory_rounds > 0 && !state.classifying_unbounded {
        match root_cuts(state, domains)? {
            StopReason::Finished => {}
            StopReason::Limit => {
                // The re-solve after a round was interrupted: no usable root optimum.
                // A resume re-enters `initial_solve` and runs the cut loop again.
                state.root_solved = false;
                return Ok(Some(TerminationReason::TimeLimit));
            }
        }
    }

    let root = Node {
        bound_changes: Vec::new(),
        basis: state.solver.snapshot_basis(),
        lp_bound: state.solver.cur_obj_val,
        depth: 0,
        parent_id: 0,
        branch_var: None,
        branch_up: false,
        branch_frac: 1.0,
        // Root deductions became `state.root_bounds`, which every node resets to.
        deduced: None,
        propagated: true,
    };
    let int_tol = state.options.int_tol;
    if let Some(cutoff) = enumeration_cutoff(state) {
        if root.lp_bound > cutoff {
            // Nothing within the cutoff at all: the enumeration is exhausted.
            return Err(Error::Infeasible);
        }
    }
    if branching::is_integral(&state.solver, domains, int_tol) {
        let outcome = match on_candidate {
            Some(on_candidate) => enumerate_candidate(state, domains, int_tol, on_candidate)?,
            None => process_integral_candidate(state, domains, int_tol)?,
        };
        match outcome {
            IntegralCandidate::Branch(var) => branch(state, &root, var),
            // Enumeration: the root closed, so the tree is exhausted.
            IntegralCandidate::Closed if state.enumeration.is_some() => {
                return Err(Error::Infeasible);
            }
            IntegralCandidate::Closed => {
                return Ok(Some(TerminationReason::ProvenOptimal));
            }
            IntegralCandidate::Limit => {
                state.root_solved = false;
                return Ok(Some(TerminationReason::TimeLimit));
            }
            IntegralCandidate::Stop => return Ok(Some(TerminationReason::ProvenOptimal)),
        }
    } else {
        match branching::select_branch_var(state, domains, int_tol)? {
            Some(var) => branch(state, &root, var),
            // Fractional integer variables fixed to a non-integer value cannot
            // produce an integer point or a useful branch. Strong branching can also
            // prove both children infeasible, which says the same about the root.
            None => return Err(Error::Infeasible),
        }
    }

    Ok(None)
}

enum NodeVisit {
    /// The node was discarded before an LP solve, so it consumes no node budget.
    Pruned,
    /// One node LP completed, including an infeasible relaxation.
    Solved,
    /// The node must be restored to the frontier. `lp_solved` distinguishes an
    /// interrupted initial LP from an interrupted retry after a completed LP.
    Interrupted { node: Node, lp_solved: bool },
    /// Enumeration only: the callback asked to stop (the node's LP was solved).
    Stopped,
}

/// Fold the objective gain of the branch that created `node` into its pseudocost:
/// `objective` over the parent's bound, per unit of the parent's fractionality.
fn record_branch_gain(state: &mut MipState, node: &Node, objective: f64) {
    if let Some(var) = node.branch_var {
        state.pseudocosts.record(
            var,
            node.branch_up,
            (objective - node.lp_bound).max(0.0) / node.branch_frac.max(params::BRANCH_FRAC_GUARD),
        );
    }
}

/// Reconstruct and process one node selected by the outer search policy.
fn visit_node(
    state: &mut MipState,
    mut node: Node,
    domains: &[VarDomain],
    on_candidate: Option<&mut CandidateFn<'_>>,
) -> Result<NodeVisit, Error> {
    if !apply_node_bounds(state, &node) {
        state.diving = false;
        return Ok(NodeVisit::Pruned);
    }

    // Propagation before the LP: deduced bounds go onto the node (its children inherit
    // them), and a contradiction prunes the node without solving anything.
    if state.options.propagate_rounds > 0 && !node.propagated {
        node.propagated = true;
        let propagation = state
            .solver
            .propagate_bounds(state.options.propagate_rounds)?;
        // Record even when infeasible: those bounds are in the solver already, and
        // `state.applied` is what tells the next node to reset them.
        record_propagated(state, &mut node, propagation.changes);
        if !propagation.feasible {
            state.stats.propagation_prunes += 1;
            state.last_solved_id = None;
            state.diving = false;
            return Ok(NodeVisit::Pruned);
        }
    }

    let warm = state.last_solved_id == Some(node.parent_id);
    // Rows added lazily after this node's basis was stored have their slacks at the
    // end of the variable list; they enter as basic (see `Solver::append_rows`).
    let total = state.solver.num_total_vars();
    if !warm && node.basis.0.len() < total {
        node.basis.0.resize(total, crate::solver::VarStatus::Basic);
    }
    if !warm && state.solver.load_basis(&node.basis).is_err() {
        debug!("basis load failed; falling back to slack basis");
        let slack = state.solver.slack_basis();
        state
            .solver
            .load_basis(&slack)
            .map_err(|e| Error::InternalError(format!("slack basis load failed: {}", e)))?;
    }

    let (node_lp, cut_off) = solve_node_lp_bounded(state)?;
    if cut_off {
        // The bound passed the cutoff mid-solve: nothing here can improve on it. The
        // objective where the dual simplex stopped is a valid lower bound on this child's
        // LP value (the stop is decided on recomputed values, and the dual objective only
        // rises), so it still says how much the branch gained, conservatively. Leaving it
        // out starves the pseudocosts of exactly the branches that close subtrees: once the
        // strong-branching budget is spent, a variable whose children are always cut off
        // is never observed again, and the product rule keeps preferring variables that
        // gain nothing (case118 "- doubleton": 33,000 nodes on one flat bound).
        let objective = state.solver.cur_obj_val;
        record_branch_gain(state, &node, objective);
        state.stats.cutoff_prunes += 1;
        state.last_solved_id = None;
        state.diving = false;
        return Ok(NodeVisit::Solved);
    }
    match node_lp {
        NodeLp::Solved => {}
        NodeLp::Infeasible => {
            state.last_solved_id = None;
            state.diving = false;
            return Ok(NodeVisit::Solved);
        }
        NodeLp::Limit => {
            return Ok(NodeVisit::Interrupted {
                node,
                lp_solved: false,
            })
        }
    }

    let objective = state.solver.cur_obj_val;
    record_branch_gain(state, &node, objective);
    if let Some(incumbent) = &state.incumbent {
        if objective >= cutoff(incumbent.objective, state.options.tolerances.prune_epsilon) {
            state.last_solved_id = None;
            state.diving = false;
            return Ok(NodeVisit::Solved);
        }
    }
    if let Some(cutoff) = enumeration_cutoff(state) {
        // Strictly worse than the fixed cutoff: a permanent prune. Ties survive; in an
        // enumeration they are the alternative solutions.
        if objective > cutoff {
            state.last_solved_id = None;
            state.diving = false;
            return Ok(NodeVisit::Solved);
        }
    }

    let int_tol = state.options.int_tol;
    if branching::is_integral(&state.solver, domains, int_tol) {
        let outcome = match on_candidate {
            Some(on_candidate) => enumerate_candidate(state, domains, int_tol, on_candidate)?,
            None => process_integral_candidate(state, domains, int_tol)?,
        };
        match outcome {
            IntegralCandidate::Branch(var) => branch(state, &node, var),
            IntegralCandidate::Closed => {
                state.last_solved_id = None;
                state.diving = false;
            }
            IntegralCandidate::Limit => {
                return Ok(NodeVisit::Interrupted {
                    node,
                    lp_solved: true,
                })
            }
            IntegralCandidate::Stop => return Ok(NodeVisit::Stopped),
        }
        return Ok(NodeVisit::Solved);
    }

    match branching::select_branch_var(state, domains, int_tol)? {
        Some(var) => branch(state, &node, var),
        None => {
            state.last_solved_id = None;
            state.diving = false;
        }
    }
    Ok(NodeVisit::Solved)
}

/// The branch & bound loop. `on_candidate` is `Some` exactly for enumeration runs
/// (`state.enumeration` set); a callback stop is reported as `ProvenOptimal` with
/// `Enumeration::stopped` raised, and an exhausted enumeration as `Err(Infeasible)`.
fn search_loop(
    state: &mut MipState,
    mut on_candidate: Option<&mut CandidateFn<'_>>,
) -> Result<TerminationReason, Error> {
    let domains = state.solver.orig_var_domains.clone();
    state.solver.deadline = state.deadline;

    if let Some(outcome) = initialize_root(state, &domains, on_candidate.as_deref_mut())? {
        return Ok(outcome);
    }

    let mut nodes_this_run: u64 = 0;

    loop {
        // Tree exhausted → the proof is COMPLETE: fall through to the post-loop
        // incumbent-vs-Infeasible verdict. Checked before the gap/deadline/node
        // limit tests so a limit that lands on the exact iteration the tree empties
        // never masks a finished proof as Interrupted/Feasible. (The loop is only
        // entered with `root_solved` true; the bottom `pop_node → None → break`
        // remains a safety net for any other path that empties `open` mid-body.)
        if state.open.is_empty() {
            break;
        }

        // Proof-quality stops: exact incumbent/bound equality wins; otherwise
        // an incumbent within `mip_gap` is a valid feasible result. Checked
        // first so proof quality takes priority over limit interruptions.
        if state.options.mip_gap > 0.0 {
            if let (Some(inc), Some(bound)) = (&state.incumbent, global_bound_internal(state)) {
                // The open tree may still contain nodes whose stored bounds
                // equal the incumbent. Equality of the incumbent and global
                // bound is nevertheless a complete proof, so it must not be
                // weakened to a gap-satisfied feasible result.
                if bound >= inc.objective {
                    state.open.clear();
                    break;
                }
                if relative_gap(inc.objective, bound) <= state.options.mip_gap {
                    return Ok(TerminationReason::MipGap);
                }
            }
        }

        // Global limits are checked between nodes; no unfinished node result is consulted.
        if check_deadline(&state.deadline) == StopReason::Limit {
            return Ok(TerminationReason::TimeLimit);
        }
        let node = match pop_node(state) {
            Some(n) => n,
            None => break,
        };

        // Prune with the stored parent bound before any LP work.
        if let Some(inc) = &state.incumbent {
            if node.lp_bound >= cutoff(inc.objective, state.options.tolerances.prune_epsilon) {
                state.diving = false;
                continue;
            }
        }
        if let Some(cutoff) = enumeration_cutoff(state) {
            if node.lp_bound > cutoff {
                state.diving = false;
                continue;
            }
        }

        // A node budget limits LP solves, not free bookkeeping. Pop and apply
        // the stored-bound prune first so hitting the exact solve count can
        // still finish a proof whose remaining nodes are already dominated by
        // the incumbent.
        if let Some(nl) = state.options.node_limit {
            if nodes_this_run >= nl {
                state.open.push(node);
                state.diving = false;
                return Ok(TerminationReason::NodeLimit);
            }
        }

        match visit_node(state, node, &domains, on_candidate.as_deref_mut())? {
            NodeVisit::Pruned => continue,
            NodeVisit::Stopped => {
                state.stats.nodes_solved += 1;
                return Ok(TerminationReason::ProvenOptimal);
            }
            NodeVisit::Solved => {
                state.stats.nodes_solved += 1;
                nodes_this_run += 1;
            }
            NodeVisit::Interrupted { node, lp_solved } => {
                if lp_solved {
                    state.stats.nodes_solved += 1;
                }
                state.open.push(node);
                state.last_solved_id = None;
                state.diving = false;
                return Ok(TerminationReason::TimeLimit);
            }
        }
    }

    if state.incumbent.is_some() {
        Ok(TerminationReason::ProvenOptimal)
    } else {
        Err(Error::Infeasible)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ComparisonOp, OptimizationDirection, Problem};

    fn int_2var_problem() -> Problem {
        // minimize 3a + 4b s.t. a + 2b >= 5, 3a + b >= 4; a,b integer in [0,10].
        // LP relaxation: a=0.6, b=2.2, obj 10.6. Integer optimum: a=1, b=2, obj 11.
        let mut p = Problem::new(OptimizationDirection::Minimize);
        let a = p.add_integer_var(3.0, (0, 10));
        let b = p.add_integer_var(4.0, (0, 10));
        p.add_constraint(&[(a, 1.0), (b, 2.0)], ComparisonOp::Ge, 5.0);
        p.add_constraint(&[(a, 3.0), (b, 1.0)], ComparisonOp::Ge, 4.0);
        p
    }

    fn binary_knapsack() -> Problem {
        // maximize 8x + 11y + 6z + 4w s.t. 5x + 7y + 4z + 3w <= 14, binaries.
        // Optimum: y + z + w = 21 (weight 14).
        let mut p = Problem::new(OptimizationDirection::Maximize);
        let x = p.add_binary_var(8.0);
        let y = p.add_binary_var(11.0);
        let z = p.add_binary_var(6.0);
        let w = p.add_binary_var(4.0);
        p.add_constraint(
            &[(x, 5.0), (y, 7.0), (z, 4.0), (w, 3.0)],
            ComparisonOp::Le,
            14.0,
        );
        p
    }

    fn incumbent_obj(state: &MipState) -> f64 {
        state.incumbent.as_ref().unwrap().objective
    }

    #[test]
    fn driver_finds_integer_optimum() {
        let run = run(&int_2var_problem(), SolveOptions::default()).unwrap();
        assert_eq!(run.reason, TerminationReason::ProvenOptimal);
        // Internal space == user space for Minimize.
        assert!((incumbent_obj(&run.state) - 11.0).abs() < 1e-6);
        let inc = run.state.incumbent.as_ref().unwrap();
        assert!((inc.values[0] - 1.0).abs() < 1e-6);
        assert!((inc.values[1] - 2.0).abs() < 1e-6);
        assert!(run.state.stats.nodes_solved > 0);
    }

    #[test]
    fn driver_binary_knapsack_maximize() {
        let run = run(&binary_knapsack(), SolveOptions::default()).unwrap();
        assert_eq!(run.reason, TerminationReason::ProvenOptimal);
        // Maximize is negated internally: internal optimum is -21.
        assert!((incumbent_obj(&run.state) + 21.0).abs() < 1e-6);
    }

    #[test]
    fn driver_no_integer_point_is_infeasible() {
        // 2x == 1 with x integer in [0,10]: LP-feasible (x=0.5), integer-infeasible.
        let mut p = Problem::new(OptimizationDirection::Minimize);
        let x = p.add_integer_var(1.0, (0, 10));
        p.add_constraint(&[(x, 2.0)], ComparisonOp::Eq, 1.0);
        assert_eq!(
            run(&p, SolveOptions::default()).unwrap_err(),
            crate::Error::Infeasible
        );
    }

    #[test]
    fn driver_exact_node_exhaustion_reports_infeasible_not_interrupted() {
        // Same infeasible fixture (2x == 1, x int in [0,10]): the root LP is
        // fractional (x=0.5) and branches into x<=0 and x>=1, both LP-infeasible.
        // The tree is therefore exactly two nodes: node_limit=1 interrupts and
        // node_limit=2 exhausts it. At the exact exhaustion count, the empty-open
        // check precedes the node-limit check and must report Infeasible.
        let mut p = Problem::new(OptimizationDirection::Minimize);
        let x = p.add_integer_var(1.0, (0, 10));
        p.add_constraint(&[(x, 2.0)], ComparisonOp::Eq, 1.0);
        let mut options = SolveOptions::default();
        options.node_limit = Some(2);
        assert_eq!(run(&p, options).unwrap_err(), crate::Error::Infeasible);
    }

    #[test]
    fn driver_node_limit_equal_to_exhaustion_count_reports_optimal() {
        // int_2var_problem is proven optimal in exactly 2 B&B nodes (deterministic:
        // the unlimited solve reports nodes_solved == 2; node_limit=1 -> Interrupted,
        // node_limit=2 -> Optimal). Setting the node limit to that exact count must
        // report Optimal, not Interrupted: when the tree empties on the same
        // iteration the limit would fire, the empty-`open` check at the loop top
        // wins and the finished proof is reported honestly.
        assert_eq!(
            run(&int_2var_problem(), SolveOptions::default())
                .unwrap()
                .state
                .stats
                .nodes_solved,
            2,
            "node count must stay deterministic; update the limit below if it changes"
        );
        let mut options = SolveOptions::default();
        options.node_limit = Some(2);
        let r = run(&int_2var_problem(), options).unwrap();
        assert_eq!(r.reason, TerminationReason::ProvenOptimal);
        assert!((incumbent_obj(&r.state) - 11.0).abs() < 1e-6);
    }

    #[test]
    fn driver_node_limit_interrupts_and_resumes_to_same_optimum() {
        let mut options = SolveOptions::default();
        options.node_limit = Some(1);
        let mut r = run(&int_2var_problem(), options).unwrap();
        let mut guard = 0;
        while r.reason != TerminationReason::ProvenOptimal {
            guard += 1;
            assert!(guard < 10_000, "resume loop did not terminate");
            r.reason = resume_run(&mut r.state, ResumeOptions::default()).unwrap();
        }
        assert!(guard >= 1, "node_limit=1 should interrupt at least once");
        assert!((incumbent_obj(&r.state) - 11.0).abs() < 1e-6);
    }

    #[test]
    fn open_children_have_branch_metadata() {
        let mut options = SolveOptions::default();
        options.node_limit = Some(0);
        let run = run(&int_2var_problem(), options).unwrap();

        assert_eq!(run.reason, TerminationReason::NodeLimit);
        assert_eq!(run.state.open.len(), 2);
        assert!(run.state.open.iter().all(|node| node.branch_var.is_some()));
    }

    #[test]
    fn driver_zero_time_limit_interrupts_cleanly_then_resumes() {
        let mut options = SolveOptions::default();
        options.time_limit = Some(Duration::ZERO);
        let mut r = run(&binary_knapsack(), options).unwrap();
        assert_eq!(r.reason, TerminationReason::TimeLimit);
        assert!(r.state.incumbent.is_none());
        let resume_options = ResumeOptions {
            time_limit: Some(Duration::from_secs(10)),
            ..ResumeOptions::default()
        };
        let reason = resume_run(&mut r.state, resume_options).unwrap();
        assert_eq!(reason, TerminationReason::ProvenOptimal);
        assert!((incumbent_obj(&r.state) + 21.0).abs() < 1e-6);
    }

    #[test]
    fn optimal_solve_reports_zero_gap_and_matching_bound() {
        let r = run(&int_2var_problem(), SolveOptions::default()).unwrap();
        assert_eq!(r.reason, TerminationReason::ProvenOptimal);
        assert_eq!(r.state.stats.gap, Some(0.0));
        // User space == internal for Minimize.
        assert!((r.state.stats.best_bound.unwrap() - 11.0).abs() < 1e-6);
    }

    #[test]
    fn unsolved_root_with_incumbent_has_no_proven_bound_or_gap() {
        let problem = binary_knapsack();
        let mut state = build_state(&problem, SolveOptions::default()).unwrap();
        state.incumbent = Some(Incumbent {
            values: vec![0.0; problem.obj_coeffs.len()],
            objective: 0.0,
        });
        state.root_solved = false;
        state.open.clear();

        fill_bound_stats(&mut state);

        assert_eq!(state.stats.best_bound, None);
        assert_eq!(state.stats.gap, None);
    }

    #[test]
    fn a_node_stopped_at_the_cutoff_still_feeds_the_pseudocosts() {
        // minimize x s.t. 2x >= 3, x integer: the root LP is x = 1.5 (objective 1.5).
        // With an incumbent at 2, the up child x >= 2 has LP value 2, past the cutoff, so
        // its dual simplex stops there and the node is pruned.
        let mut p = Problem::new(OptimizationDirection::Minimize);
        let x = p.add_integer_var(1.0, (0, 10));
        p.add_constraint(&[(x, 2.0)], ComparisonOp::Ge, 3.0);
        let mut state = build_state(&p, SolveOptions::default()).unwrap();
        state.solver.initial_solve().unwrap();
        let root_obj = state.solver.cur_obj_val;
        assert!((root_obj - 1.5).abs() < 1e-9, "{root_obj}");
        state.incumbent = Some(Incumbent {
            values: vec![2.0],
            objective: 2.0,
        });
        state.node_seq = 1;
        state.last_solved_id = Some(1); // warm: the solver sits at the parent's optimum
        let up = Node {
            bound_changes: vec![(x.idx(), 2.0, 10.0)],
            basis: state.solver.snapshot_basis(),
            lp_bound: root_obj,
            depth: 1,
            parent_id: 1,
            branch_var: Some(x.idx()),
            branch_up: true,
            branch_frac: 0.5,
            deduced: None,
            propagated: false,
        };
        let domains = state.solver.orig_var_domains.clone();

        let visit = visit_node(&mut state, up, &domains, None).unwrap();

        assert!(matches!(visit, NodeVisit::Solved));
        assert_eq!(state.stats.cutoff_prunes, 1, "the child must stop at the cutoff");
        assert_eq!(
            state.pseudocosts.observations(x.idx()),
            (0, 1),
            "the up branch's gain was dropped"
        );
        // The gain is taken where the dual simplex stopped: 2 - 1.5 = 0.5 over a
        // fractionality of 0.5, a lower bound on the true 1.0 per unit.
        let est = state.pseudocosts.estimate(x.idx(), true);
        assert!(est > 0.0 && est <= 1.0 + 1e-9, "{est}");
    }

    #[test]
    fn maximize_bound_is_in_user_space() {
        let r = run(&binary_knapsack(), SolveOptions::default()).unwrap();
        assert_eq!(r.reason, TerminationReason::ProvenOptimal);
        // Internally -21; user-facing bound must be +21.
        assert!((r.state.stats.best_bound.unwrap() - 21.0).abs() < 1e-6);
    }

    #[test]
    fn mip_gap_stops_early_with_consistent_bound() {
        let mut options = SolveOptions::default();
        options.mip_gap = 0.5;
        let r = run(&binary_knapsack(), options).unwrap();
        assert_eq!(r.reason, TerminationReason::MipGap);
        let inc = -incumbent_obj(&r.state); // user space (Maximize)
        let bound = r.state.stats.best_bound.unwrap();
        // Incumbent within 50% of the proven bound, and never better than it.
        assert!(inc <= bound + 1e-9);
        assert!((bound - inc) / bound.abs().max(1e-10) <= 0.5 + 1e-9);
    }

    #[test]
    fn feasible_interrupt_reports_gap() {
        let mut options = SolveOptions::default();
        options.node_limit = Some(2);
        let mut r = run(&binary_knapsack(), options).unwrap();
        // Resume with node budget until an incumbent exists but the search isn't done.
        let mut guard = 0;
        while r.reason == TerminationReason::NodeLimit && r.state.incumbent.is_none() {
            guard += 1;
            assert!(guard < 10_000);
            r.reason = resume_run(
                &mut r.state,
                ResumeOptions {
                    node_limit: Some(2),
                    ..ResumeOptions::default()
                },
            )
            .unwrap();
        }
        if r.reason == TerminationReason::NodeLimit {
            // Feasible-but-unproven: a gap must be reported.
            assert!(r.state.stats.gap.unwrap() >= 0.0);
            assert!(r.state.stats.best_bound.is_some());
        }
    }

    #[test]
    fn plunge_and_jump_selection_preserves_optima() {
        // Same optima as plain DFS on all driver test problems, plus interrupt/resume.
        let r = run(&int_2var_problem(), SolveOptions::default()).unwrap();
        assert!((incumbent_obj(&r.state) - 11.0).abs() < 1e-6);

        let r = run(&binary_knapsack(), SolveOptions::default()).unwrap();
        assert!((incumbent_obj(&r.state) + 21.0).abs() < 1e-6);

        let mut options = SolveOptions::default();
        options.node_limit = Some(1);
        let mut r = run(&binary_knapsack(), options).unwrap();
        let mut guard = 0;
        while r.reason != TerminationReason::ProvenOptimal {
            guard += 1;
            assert!(guard < 10_000);
            r.reason = resume_run(&mut r.state, ResumeOptions::default()).unwrap();
        }
        assert!((incumbent_obj(&r.state) + 21.0).abs() < 1e-6);
        // After a best-bound jump the pop is NOT the last-pushed node at least once
        // on this instance; correctness above is the real assertion.
    }

    #[test]
    fn tolerances_default_matches_documented_values() {
        let t = Tolerances::default();
        assert_eq!(
            t.feasibility, 1e-7,
            "see Tolerances::feasibility's doc default"
        );
        assert_eq!(
            t.integrality_rounding, 1e-5,
            "see Tolerances::integrality_rounding's doc default"
        );
        assert_eq!(
            t.prune_epsilon, 1e-9,
            "see Tolerances::prune_epsilon's doc default"
        );
    }

    #[test]
    fn try_adopt_incumbent_respects_custom_feasibility_tolerance() {
        // Derived from the big-M fixture `tests_general::solve_big_m` (same
        // m = 1e9 shape: `x - m*b == 10`, minimize x). Pin b to a value that
        // is integral-within-`int_tol` (5e-7, well inside the default 1e-6)
        // but not exactly 0; the rounded-incumbent guard then re-checks the
        // ROUNDED point (b -> 0) against the ORIGINAL row, which is off by
        // exactly m * 5e-7 = 500 — precisely the "big-M trap"
        // `tolerances.feasibility` exists to catch, in absolute terms.
        let m = 1.0e9;
        let mut p = Problem::new(OptimizationDirection::Minimize);
        let x = p.add_var(1.0, (0.0, f64::INFINITY));
        let b = p.add_binary_var(0.0);
        p.add_constraint(&[(x, 1.0), (b, -m)], ComparisonOp::Eq, 10.0);

        let mut solved = run(&p, SolveOptions::default()).unwrap();
        let state = &mut solved.state;

        // Force the relaxation to a specific near-zero fractional b: pin its
        // bounds to [5e-7, 5e-7] and re-solve. The equality row then forces x
        // to exactly 10 + m*5e-7 = 510, deterministically — no dependence on
        // which vertex the simplex would otherwise have picked.
        state.solver.set_var_bounds(b.idx(), 5e-7, 5e-7).unwrap();
        assert_eq!(
            state.solver.reoptimize().unwrap(),
            crate::StopReason::Finished
        );
        assert!((*state.solver.get_value(x.idx()) - 510.0).abs() < 1e-6);

        // Default tolerance (1e-7): the rounded point (x=510, b=0) misses the
        // original `x - m*b == 10` row by 500 — must be rejected.
        state.options.tolerances.feasibility = Tolerances::default().feasibility;
        assert!(
            !try_adopt_incumbent(state).unwrap(),
            "a 500-unit rounding-induced violation must be rejected at the default feasibility tolerance"
        );

        // Absurdly loosened tolerance: the same 500-unit violation is now
        // within bounds — the guard must accept it.
        state.options.tolerances.feasibility = 1e6;
        assert!(
            try_adopt_incumbent(state).unwrap(),
            "the same violation must be accepted once tolerances.feasibility is loosened past it"
        );
    }

    #[test]
    fn incumbent_feasible_row_tolerance_is_absolute_not_relative_to_rhs() {
        // `incumbent_feasible` uses the same absolute feasibility tolerance
        // as the rounded-incumbent guard, regardless of row magnitude.
        let mut p = Problem::new(OptimizationDirection::Minimize);
        let x = p.add_var(1.0, (0.0, f64::INFINITY));
        p.add_constraint(&[(x, 1.0)], ComparisonOp::Le, 1000.0);
        let fixed = std::collections::BTreeMap::new();
        let tolerances = Tolerances::default();

        // Within the absolute tolerance (5e-8 < 1e-7): accepted.
        assert!(incumbent_feasible(
            &p,
            &fixed,
            &[1000.0 + 5e-8],
            &tolerances
        ));

        // A `5e-5` violation exceeds the `1e-7` absolute tolerance even though
        // it is small relative to this row's right-hand side.
        assert!(!incumbent_feasible(
            &p,
            &fixed,
            &[1000.0 + 5e-5],
            &tolerances
        ));
    }

    #[test]
    fn candidate_validation_rejects_non_finite_and_malformed_values() {
        let mut problem = Problem::new(OptimizationDirection::Minimize);
        problem.add_integer_var(1.0, (0, 10));
        let fixed = BTreeMap::new();
        let tolerances = Tolerances::default();

        assert!(!incumbent_feasible(&problem, &fixed, &[], &tolerances));
        assert!(!incumbent_feasible(
            &problem,
            &fixed,
            &[f64::NAN],
            &tolerances
        ));
        assert!(!incumbent_feasible(
            &problem,
            &fixed,
            &[f64::INFINITY],
            &tolerances
        ));

        let mut overflowing_row = Problem::new(OptimizationDirection::Minimize);
        let x = overflowing_row.add_var(0.0, (0.0, f64::INFINITY));
        overflowing_row.add_constraint(&[(x, 1.0e308)], ComparisonOp::Le, 1.0e308);
        assert!(!incumbent_feasible(
            &overflowing_row,
            &fixed,
            &[1.0e308],
            &tolerances
        ));
    }

    #[test]
    fn valid_candidate_completes_unbounded_classification() {
        let mut problem = Problem::new(OptimizationDirection::Minimize);
        problem.add_integer_var(1.0, (0, 10));
        let mut state = build_state(&problem, SolveOptions::default()).unwrap();
        assert_eq!(state.solver.initial_solve().unwrap(), StopReason::Finished);
        state.classifying_unbounded = true;

        assert_eq!(try_adopt_incumbent(&mut state), Err(Error::Unbounded));
    }

    #[test]
    fn warm_start_restores_bounds_before_unbounded_verdict() {
        let mut problem = Problem::new(OptimizationDirection::Minimize);
        let x = problem.add_integer_var(0.0, (0, 10));
        let mut state = build_state(&problem, SolveOptions::default()).unwrap();
        assert_eq!(state.solver.initial_solve().unwrap(), StopReason::Finished);
        state.classifying_unbounded = true;

        assert_eq!(
            try_warm_start(&mut state, &[(x, 5.0)]),
            Err(Error::Unbounded)
        );
        assert_eq!(state.solver.get_var_bounds(x.idx()), (0.0, 10.0));
    }
}
