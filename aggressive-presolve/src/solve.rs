//! Presolve, solve with microlp, postsolve, check, and fall back.

use crate::check::{check, CheckOptions, CheckReport, Violation};
use crate::model::Model;
use crate::presolve::{presolve, PresolveOptions, Stats};
use std::time::{Duration, Instant};

/// Options of [`solve`].
#[derive(Clone, Debug)]
pub struct Options {
    /// The aggressive presolve (`presolve.enabled = false` solves the model
    /// as written). Its `feasibility` and `int_tol` are overwritten by the
    /// fields below.
    pub presolve: PresolveOptions,
    /// microlp's own presolve on the reduced model.
    pub microlp_presolve: bool,
    /// Wall-clock budget for the whole call, re-solves included.
    pub time_limit: Option<Duration>,
    /// microlp's relative MIP gap.
    pub mip_gap: f64,
    /// The contract's absolute feasibility tolerance (microlp's
    /// `Tolerances::feasibility`), for presolve, microlp and the check.
    pub feasibility: f64,
    /// Integrality tolerance, for presolve, microlp and the check.
    pub int_tol: f64,
    /// Re-solves with more reductions undone after a failed check, before the
    /// original model is solved.
    pub max_retries: usize,
}

impl Default for Options {
    fn default() -> Self {
        Options {
            presolve: PresolveOptions::default(),
            microlp_presolve: true,
            time_limit: None,
            mip_gap: 0.0,
            feasibility: 1e-7,
            int_tol: 1e-6,
            max_retries: 2,
        }
    }
}

/// How the solve ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Status {
    Optimal,
    /// A checked point without a proof of optimality (a time or node limit).
    Feasible,
    Infeasible,
    Unbounded,
    /// A limit was reached before any point was found.
    NoSolution,
}

/// Which path produced the answer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Path {
    /// Presolve was off: microlp on the model as written.
    NoPresolve,
    /// The first reduced model's answer passed the check.
    Presolved,
    /// A re-solve with `k` rounds of protections passed the check.
    PresolvedRetry(usize),
    /// microlp on the model as written, after presolve could not answer.
    Original(String),
}

/// One microlp solve made on the way to the answer.
#[derive(Clone, Debug)]
pub struct Attempt {
    pub label: String,
    pub rows: usize,
    pub cols: usize,
    pub outcome: String,
    pub nodes: u64,
    pub time: Duration,
    /// The check of the postsolved point, where there was one.
    pub check_ok: Option<bool>,
    pub violations: usize,
}

/// The answer, in the original variables.
#[derive(Clone, Debug)]
pub struct Answer {
    pub status: Status,
    /// Objective on the original model, offset included.
    pub objective: Option<f64>,
    pub values: Option<Vec<f64>>,
    pub path: Path,
    /// Statistics of the presolve whose reduced model answered (or of the
    /// last one tried).
    pub presolve_stats: Option<Stats>,
    /// The check of `values` against the model as written.
    pub check: Option<CheckReport>,
    /// Branch & bound nodes over every attempt.
    pub nodes: u64,
    pub attempts: Vec<Attempt>,
    pub time: Duration,
}

/// What can stop [`solve`] without an answer.
#[derive(Clone, Debug, PartialEq)]
pub enum Error {
    /// The model cannot be given to microlp (see [`Model::to_problem`]).
    Model(String),
    /// microlp failed on the model as written.
    Microlp(microlp::Error),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Model(s) => write!(f, "model: {s}"),
            Error::Microlp(e) => write!(f, "microlp: {e}"),
        }
    }
}

impl std::error::Error for Error {}

enum Run {
    Point { values: Vec<f64>, optimal: bool },
    NoPoint,
    Failed(microlp::Error),
}

struct Runner<'a> {
    opts: &'a Options,
    deadline: Option<Instant>,
    attempts: Vec<Attempt>,
    nodes: u64,
}

impl Runner<'_> {
    fn microlp(&mut self, model: &Model, label: &str) -> Result<Run, String> {
        let started = Instant::now();
        let mut attempt = Attempt {
            label: label.to_string(),
            rows: model.rows.len(),
            cols: model.vars.len(),
            outcome: String::new(),
            nodes: 0,
            time: Duration::ZERO,
            check_ok: None,
            violations: 0,
        };
        let run = if model.vars.is_empty() {
            Run::Point {
                values: Vec::new(),
                optimal: true,
            }
        } else {
            let (problem, handles) = model.to_problem()?;
            let mut o = microlp::SolveOptions::default();
            o.presolve = self.opts.microlp_presolve;
            o.int_tol = self.opts.int_tol;
            o.mip_gap = self.opts.mip_gap;
            o.tolerances.feasibility = self.opts.feasibility;
            if let Some(d) = self.deadline {
                o.time_limit = Some(d.saturating_duration_since(Instant::now()));
            }
            match problem.solve_with(o) {
                Ok(microlp::SolveOutcome::Solution(sol)) => {
                    attempt.nodes = sol.stats().nodes_solved;
                    Run::Point {
                        values: handles.iter().map(|&h| sol.var_value(h)).collect(),
                        optimal: sol.status() == microlp::SolutionStatus::Optimal,
                    }
                }
                Ok(microlp::SolveOutcome::Interrupted(i)) => {
                    attempt.nodes = i.stats().nodes_solved;
                    Run::NoPoint
                }
                Err(e) => Run::Failed(e),
            }
        };
        attempt.outcome = match &run {
            Run::Point { optimal: true, .. } => "optimal".into(),
            Run::Point { optimal: false, .. } => "feasible".into(),
            Run::NoPoint => "no solution".into(),
            Run::Failed(e) => format!("{e}"),
        };
        attempt.time = started.elapsed();
        self.nodes += attempt.nodes;
        self.attempts.push(attempt);
        Ok(run)
    }
}

/// Solves `model`: aggressive presolve, microlp on the reduced model,
/// postsolve, and the check against `model` as written. A failed check
/// re-solves with the failing rows protected and the failing columns frozen,
/// up to `max_retries` times, then solves `model` with microlp alone. A
/// verdict without a point (`Infeasible`, `Unbounded`) from the reduced model,
/// and presolve's own infeasibility, are confirmed on `model`.
///
/// The answer is a point and an objective only: no duals, post-solve edits,
/// warm starts, resume or enumeration (see the crate docs).
pub fn solve(model: &Model, opts: &Options) -> Result<Answer, Error> {
    let started = Instant::now();
    let mut runner = Runner {
        opts,
        deadline: opts.time_limit.map(|t| started + t),
        attempts: Vec::new(),
        nodes: 0,
    };
    let check_opts = CheckOptions {
        feasibility: opts.feasibility,
        int_tol: opts.int_tol,
    };
    let mut popts = opts.presolve.clone();
    popts.feasibility = opts.feasibility;
    popts.int_tol = opts.int_tol;

    let mut last_stats = None;
    let reason = 'presolved: {
        if !popts.enabled {
            break 'presolved None;
        }
        for round in 0..=opts.max_retries {
            let pre = match presolve(model, &popts) {
                Ok(pre) => pre,
                Err(inf) => break 'presolved Some(format!("presolve infeasible ({})", inf.reason)),
            };
            last_stats = Some(pre.stats.clone());
            let label = if round == 0 {
                "reduced".to_string()
            } else {
                format!("reduced, retry {round}")
            };
            let run = match runner.microlp(&pre.reduced, &label) {
                Ok(run) => run,
                Err(e) => break 'presolved Some(format!("reduced model rejected: {e}")),
            };
            let (values, optimal) = match run {
                Run::Point { values, optimal } => (values, optimal),
                Run::NoPoint => {
                    return Ok(Answer {
                        status: Status::NoSolution,
                        objective: None,
                        values: None,
                        path: if round == 0 {
                            Path::Presolved
                        } else {
                            Path::PresolvedRetry(round)
                        },
                        presolve_stats: last_stats,
                        check: None,
                        nodes: runner.nodes,
                        attempts: runner.attempts,
                        time: started.elapsed(),
                    })
                }
                Run::Failed(e) => break 'presolved Some(format!("reduced model: {e}")),
            };
            let x = pre.postsolve.values(&values);
            let report = check(model, &x, &check_opts);
            let last = runner.attempts.last_mut().unwrap();
            last.check_ok = Some(report.ok());
            last.violations = report.violations.len();
            if report.ok() {
                return Ok(Answer {
                    status: if optimal {
                        Status::Optimal
                    } else {
                        Status::Feasible
                    },
                    objective: Some(model.objective(&x)),
                    values: Some(x),
                    path: if round == 0 {
                        Path::Presolved
                    } else {
                        Path::PresolvedRetry(round)
                    },
                    presolve_stats: last_stats,
                    check: Some(report),
                    nodes: runner.nodes,
                    attempts: runner.attempts,
                    time: started.elapsed(),
                });
            }
            let mut grew = false;
            for v in &report.violations {
                grew |= match *v {
                    Violation::Row { row, .. } | Violation::NotFinite { row: Some(row), .. } => {
                        popts.protected_rows.insert(row)
                    }
                    Violation::Bound { var, .. }
                    | Violation::Integrality { var, .. }
                    | Violation::NotFinite { var: Some(var), .. } => popts.frozen_cols.insert(var),
                    Violation::NotFinite { .. } => false,
                };
            }
            if !grew {
                break;
            }
        }
        Some(format!("check failed after {} retries", opts.max_retries))
    };

    // The model as written.
    let path = match reason {
        None => Path::NoPresolve,
        Some(r) => Path::Original(r),
    };
    let run = runner.microlp(model, "original").map_err(Error::Model)?;
    let (status, values) = match run {
        Run::Point { values, optimal } => (
            if optimal {
                Status::Optimal
            } else {
                Status::Feasible
            },
            Some(values),
        ),
        Run::NoPoint => (Status::NoSolution, None),
        Run::Failed(microlp::Error::Infeasible) => (Status::Infeasible, None),
        Run::Failed(microlp::Error::Unbounded) => (Status::Unbounded, None),
        Run::Failed(e) => return Err(Error::Microlp(e)),
    };
    let report = values.as_ref().map(|x| check(model, x, &check_opts));
    if let (Some(r), Some(last)) = (&report, runner.attempts.last_mut()) {
        last.check_ok = Some(r.ok());
        last.violations = r.violations.len();
    }
    Ok(Answer {
        status,
        objective: values.as_ref().map(|x| model.objective(x)),
        values,
        path,
        presolve_stats: last_stats,
        check: report,
        nodes: runner.nodes,
        attempts: runner.attempts,
        time: started.elapsed(),
    })
}
