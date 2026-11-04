//! Aggressive presolve for MILPs, around [microlp].
//!
//! microlp's own presolve keeps continuous vars' bounds as written, never
//! turns rows into bounds, never changes the variable set and has no
//! postsolve. This crate adds the reductions that rules out (implied-free
//! substitution, doubleton equations, singleton rows as bounds of continuous
//! vars, forcing rows on continuous vars; see [`presolve`](mod@presolve)),
//! solves the reduced model with microlp, maps the answer back, and
//! **checks it against the model as written** at microlp's contract
//! ([`check`](mod@check)). A point that fails the check costs a re-solve with
//! the failing rows and vars protected, then a solve of the original model:
//! never a wrong answer.
//!
//! ```
//! use aggressive_presolve::{solve, Model, Direction, Sense, Options, Status};
//! let mut m = Model::new(Direction::Maximize);
//! let x = m.integer(1.0, 0.0, 10.0);
//! let y = m.continuous(2.0, 0.0, f64::INFINITY);
//! m.add_row("c1", [(x, 1.0), (y, 1.0)], Sense::Le, 4.5);
//! m.add_row("c2", [(x, 1.0), (y, -1.0)], Sense::Eq, 0.0);
//! let answer = solve(&m, &Options::default()).unwrap();
//! assert_eq!(answer.status, Status::Optimal);
//! assert!((answer.objective.unwrap() - 6.75).abs() < 1e-6);
//! ```
//!
//! # What version 1 leaves out
//!
//! * **Duals**: `Solution::dual` is not on microlp master, and no dual
//!   postsolve is implemented.
//! * **Post-solve edits** (`Solution::add_constraint`, `fix_var`), **warm
//!   starts**, **resume** of an interrupted solve and **solution
//!   enumeration**: an [`Answer`] is plain data in original indices, with
//!   nothing live behind it.
//! * The check guarantees **feasibility**, not optimality: the reductions are
//!   exact up to round-off and their margins err toward keeping points, but a
//!   reduction that cut the optimum through round-off would give a worse
//!   answer the check cannot see.

pub mod check;
pub mod lp;
pub mod model;
pub mod mps;
pub mod presolve;
pub mod sum;

pub use check::{check, Violation};
pub use model::{Direction, Model, Row, Sense, Var, VarKind};
pub use presolve::{presolve, Infeasible, Postsolve, Presolved, Stats};

use std::time::{Duration, Instant};

/// Options for [`presolve`](presolve()) and [`solve`].
#[derive(Clone, Debug)]
pub struct Options {
    /// Run the aggressive presolve (off: microlp solves the model as written).
    pub presolve: bool,
    /// microlp's own presolve on the (reduced) model.
    pub microlp_presolve: bool,
    /// Wall-clock limit for the whole [`solve`], retries included.
    pub time_limit: Option<Duration>,
    /// microlp's node limit, per microlp solve.
    pub node_limit: Option<u64>,
    /// microlp's relative MIP gap.
    pub mip_gap: f64,
    /// Integrality tolerance (microlp's `int_tol`, default `1e-6`).
    pub int_tol: f64,
    /// Feasibility tolerance (microlp's `tolerances.feasibility`, `1e-7`).
    pub feasibility: f64,
    /// Most nonzeros a substitution may add net.
    pub max_fill: usize,
    /// A substitution may grow a row's max/min |coefficient| ratio to at most
    /// this times its ratio as written.
    pub max_ratio_growth: f64,
    /// Presolve passes.
    pub max_passes: usize,
    /// Protected re-solves after a failed check, before the original model.
    pub max_retries: usize,
    /// Confirm `Infeasible`/`Unbounded` of a reduced model on the original.
    pub confirm_infeasible: bool,
    /// Original rows presolve must leave as written (less fixed columns).
    pub protect_rows: Vec<usize>,
    /// Original vars presolve must not eliminate.
    pub protect_vars: Vec<usize>,
}

impl Default for Options {
    fn default() -> Self {
        Options {
            presolve: true,
            microlp_presolve: true,
            time_limit: None,
            node_limit: None,
            mip_gap: 0.0,
            int_tol: 1e-6,
            feasibility: 1e-7,
            max_fill: 10,
            max_ratio_growth: 4.0,
            max_passes: 100,
            max_retries: 2,
            confirm_infeasible: true,
            protect_rows: Vec::new(),
            protect_vars: Vec::new(),
        }
    }
}

/// How a solve ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Status {
    /// Proven optimal (to `mip_gap`).
    Optimal,
    /// A checked feasible point, the limit reached before a proof.
    Feasible,
    Infeasible,
    Unbounded,
    /// A limit was reached with no point.
    NoSolution,
}

/// Which model the answer came from.
#[derive(Clone, Debug, PartialEq)]
pub enum Path {
    /// The reduced model, after `retries` protected re-solves.
    Reduced { retries: usize },
    /// The model as written.
    Original { reason: FallbackReason },
}

#[derive(Clone, Debug, PartialEq)]
pub enum FallbackReason {
    /// [`Options::presolve`] was off.
    Disabled,
    /// Presolve found the model infeasible (not a proof).
    PresolveInfeasible(String),
    /// microlp found the reduced model infeasible or unbounded; confirmed.
    Confirm(Status),
    /// The postsolved point failed the check, retries exhausted.
    CheckFailed,
}

/// One microlp solve, for the record.
#[derive(Clone, Debug)]
pub struct Attempt {
    /// `None` for the original model.
    pub presolve: Option<Stats>,
    pub status: Status,
    pub nodes: u64,
    pub lp_iterations: u64,
    pub time: Duration,
    pub violations: Vec<Violation>,
}

/// The answer to a [`solve`]: values in original indices, checked.
#[derive(Clone, Debug)]
pub struct Answer {
    pub status: Status,
    /// The original objective (offset included) at `values`.
    pub objective: Option<f64>,
    /// microlp's objective of the model that answered, plus its offset.
    pub reported_objective: Option<f64>,
    pub values: Vec<f64>,
    pub path: Path,
    /// Every microlp solve made, in order.
    pub attempts: Vec<Attempt>,
    pub time: Duration,
}

impl Answer {
    /// Stats of the presolve that produced the answer, if it was reduced.
    pub fn presolve_stats(&self) -> Option<&Stats> {
        match self.path {
            Path::Reduced { .. } => self.attempts.last().and_then(|a| a.presolve.as_ref()),
            Path::Original { .. } => None,
        }
    }

    /// Branch-and-bound nodes over every attempt.
    pub fn nodes(&self) -> u64 {
        self.attempts.iter().map(|a| a.nodes).sum()
    }
}

/// Errors that are not answers.
#[derive(Clone, Debug, PartialEq)]
pub enum Error {
    InvalidModel(String),
    Microlp(microlp::Error),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::InvalidModel(m) => write!(f, "invalid model: {}", m),
            Error::Microlp(e) => write!(f, "microlp: {}", e),
        }
    }
}

impl std::error::Error for Error {}

struct Run {
    status: Status,
    values: Vec<f64>,
    objective: Option<f64>,
    nodes: u64,
    lp_iterations: u64,
    time: Duration,
}

fn remaining(opts: &Options, start: Instant) -> Option<Duration> {
    opts.time_limit
        .map(|t| t.saturating_sub(start.elapsed()).max(Duration::from_millis(1)))
}

/// Solves `model` with microlp as given.
fn run_microlp(model: &Model, opts: &Options, limit: Option<Duration>) -> Result<Run, Error> {
    let started = Instant::now();
    if model.vars.is_empty() {
        let infeasible = !check(model, &[], opts.feasibility, opts.int_tol).is_empty();
        return Ok(Run {
            status: if infeasible {
                Status::Infeasible
            } else {
                Status::Optimal
            },
            values: Vec::new(),
            objective: (!infeasible).then_some(model.obj_offset),
            nodes: 0,
            lp_iterations: 0,
            time: started.elapsed(),
        });
    }
    let (problem, vars) = model.to_problem();
    let mut so = microlp::SolveOptions::default();
    so.time_limit = limit;
    so.node_limit = opts.node_limit;
    so.mip_gap = opts.mip_gap;
    so.int_tol = opts.int_tol;
    so.presolve = opts.microlp_presolve;
    so.tolerances.feasibility = opts.feasibility;
    let mut run = Run {
        status: Status::NoSolution,
        values: Vec::new(),
        objective: None,
        nodes: 0,
        lp_iterations: 0,
        time: Duration::ZERO,
    };
    match problem.solve_with(so) {
        Ok(outcome) => {
            let st = outcome.stats();
            run.nodes = st.nodes_solved;
            run.lp_iterations = st.lp_iterations;
            if let Some(sol) = outcome.solution() {
                run.status = match sol.status() {
                    microlp::SolutionStatus::Optimal => Status::Optimal,
                    _ => Status::Feasible,
                };
                run.values = vars.iter().map(|&v| sol.var_value(v)).collect();
                run.objective = Some(sol.objective() + model.obj_offset);
            }
        }
        Err(microlp::Error::Infeasible) => run.status = Status::Infeasible,
        Err(microlp::Error::Unbounded) => run.status = Status::Unbounded,
        Err(e) => return Err(Error::Microlp(e)),
    }
    run.time = started.elapsed();
    Ok(run)
}

/// Presolves, solves the reduced model with microlp, postsolves, checks the
/// point against `model` as written and falls back when the check fails
/// (see the crate docs). Statuses (including infeasible and unbounded) are
/// in [`Answer::status`]; `Err` is for invalid input and microlp errors.
pub fn solve(model: &Model, opts: &Options) -> Result<Answer, Error> {
    model.validate().map_err(Error::InvalidModel)?;
    let start = Instant::now();
    let mut attempts = Vec::new();
    if !opts.presolve {
        return solve_original(model, opts, start, attempts, FallbackReason::Disabled);
    }
    let mut popts = opts.clone();
    for retry in 0..=opts.max_retries {
        let pre = match presolve(model, &popts) {
            Ok(p) => p,
            Err(inf) => {
                return solve_original(
                    model,
                    opts,
                    start,
                    attempts,
                    FallbackReason::PresolveInfeasible(inf.reason),
                )
            }
        };
        let run = run_microlp(&pre.reduced, opts, remaining(opts, start))?;
        let mut attempt = Attempt {
            presolve: Some(pre.stats.clone()),
            status: run.status,
            nodes: run.nodes,
            lp_iterations: run.lp_iterations,
            time: run.time,
            violations: Vec::new(),
        };
        match run.status {
            Status::Optimal | Status::Feasible => {
                let x = pre.postsolve.values(&run.values);
                let violations = check(model, &x, opts.feasibility, opts.int_tol);
                if violations.is_empty() {
                    attempts.push(attempt);
                    return Ok(Answer {
                        status: run.status,
                        objective: Some(model.objective_at(&x)),
                        reported_objective: run.objective,
                        values: x,
                        path: Path::Reduced { retries: retry },
                        attempts,
                        time: start.elapsed(),
                    });
                }
                // Protect what failed and try again.
                let before = popts.protect_rows.len() + popts.protect_vars.len();
                for v in &violations {
                    match *v {
                        Violation::Row { row, .. } => popts.protect_rows.push(row),
                        Violation::Bound { var, .. } | Violation::Integrality { var, .. } => {
                            popts.protect_vars.push(var)
                        }
                    }
                }
                popts.protect_rows.sort_unstable();
                popts.protect_rows.dedup();
                popts.protect_vars.sort_unstable();
                popts.protect_vars.dedup();
                let grew = popts.protect_rows.len() + popts.protect_vars.len() > before;
                attempt.violations = violations;
                attempts.push(attempt);
                if !grew {
                    break;
                }
            }
            Status::Infeasible | Status::Unbounded => {
                attempts.push(attempt);
                if !opts.confirm_infeasible {
                    return Ok(Answer {
                        status: run.status,
                        objective: None,
                        reported_objective: None,
                        values: Vec::new(),
                        path: Path::Reduced { retries: retry },
                        attempts,
                        time: start.elapsed(),
                    });
                }
                return solve_original(
                    model,
                    opts,
                    start,
                    attempts,
                    FallbackReason::Confirm(run.status),
                );
            }
            Status::NoSolution => {
                attempts.push(attempt);
                return Ok(Answer {
                    status: Status::NoSolution,
                    objective: None,
                    reported_objective: None,
                    values: Vec::new(),
                    path: Path::Reduced { retries: retry },
                    attempts,
                    time: start.elapsed(),
                });
            }
        }
        if opts.time_limit.is_some_and(|t| start.elapsed() >= t) {
            break;
        }
    }
    solve_original(model, opts, start, attempts, FallbackReason::CheckFailed)
}

fn solve_original(
    model: &Model,
    opts: &Options,
    start: Instant,
    mut attempts: Vec<Attempt>,
    reason: FallbackReason,
) -> Result<Answer, Error> {
    let run = run_microlp(model, opts, remaining(opts, start))?;
    let violations = if run.values.is_empty() {
        Vec::new()
    } else {
        check(model, &run.values, opts.feasibility, opts.int_tol)
    };
    let objective = (!run.values.is_empty()).then(|| model.objective_at(&run.values));
    attempts.push(Attempt {
        presolve: None,
        status: run.status,
        nodes: run.nodes,
        lp_iterations: run.lp_iterations,
        time: run.time,
        violations,
    });
    Ok(Answer {
        status: run.status,
        objective,
        reported_objective: run.objective,
        values: run.values,
        path: Path::Original { reason },
        attempts,
        time: start.elapsed(),
    })
}
