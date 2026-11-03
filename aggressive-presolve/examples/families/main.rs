//! Exports the correctness suite's Rust-built MILP families
//! (`tests/suite/cases/{milp_families,milp_random,milp_unit}.rs`) to MPS, with
//! their known answers, into `target/families/`, for `examples/suite.rs`.
//!
//! The generators are compiled here unchanged through `#[path]`, against a
//! copy of the suite's `model.rs` that records the direction (`model.rs`
//! beside this file) and a shim of its `Case` type that keeps only the
//! model-building cases (`Case::solve`); custom cases are skipped.
//!
//! `cargo run --release --example families`

#![allow(dead_code)]

#[allow(clippy::all)]
#[path = "model.rs"]
mod model;
#[allow(clippy::all)]
#[path = "../../../tests/suite/oracles.rs"]
mod oracles;
#[allow(clippy::all)]
#[path = "../../../tests/suite/rng.rs"]
mod rng;

mod cases;

/// Stubs of the suite's runner items the generators touch only inside
/// custom cases, which this exporter skips.
mod verify {
    use crate::model::ModelSpec;
    use microlp::{Solution, SolveOutcome};

    pub fn require_solution(outcome: SolveOutcome, context: &str) -> Result<Solution, String> {
        outcome
            .into_solution()
            .map_err(|_| format!("{context}: no incumbent"))
    }

    pub fn validate_incumbent(_spec: &ModelSpec, _sol: &Solution) -> Result<(), String> {
        Ok(())
    }
}

thread_local! {
    pub static LAST_SOLVE: std::cell::RefCell<Option<(f64, f64)>> = const { std::cell::RefCell::new(None) };
}

use aggressive_presolve::{mps, Direction, Domain, Model, Row, Var};
use microlp::{ComparisonOp, OptimizationDirection};
use model::Expected;

fn main() {
    let mut cases = Vec::new();
    cases::milp_families::register(&mut cases);
    cases::milp_random::register(&mut cases);
    cases::milp_unit::register(&mut cases);
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("target/families");
    std::fs::create_dir_all(&dir).unwrap();
    let (mut written, mut skipped) = (0, 0);
    for case in cases {
        let Some(build) = case.build else {
            skipped += 1;
            continue;
        };
        let (spec, _problem, expected) = match build() {
            Ok(b) => b,
            Err(e) => {
                eprintln!("{}: {e}", case.name);
                skipped += 1;
                continue;
            }
        };
        let known = match expected {
            Expected::Objective { value, .. } => format!("{value:?}"),
            Expected::Infeasible => "infeasible".to_string(),
            Expected::Unbounded => "unbounded".to_string(),
        };
        let mut m = Model::new(match spec.direction {
            OptimizationDirection::Minimize => Direction::Minimize,
            OptimizationDirection::Maximize => Direction::Maximize,
        });
        m.name = case.name.clone();
        for (j, v) in spec.vars.iter().enumerate() {
            m.vars.push(Var {
                name: format!("x{j}"),
                lo: v.min,
                hi: v.max,
                cost: v.obj_coeff,
                domain: if v.domain == model::Domain::Integer {
                    Domain::Integer
                } else {
                    Domain::Real
                },
            });
        }
        for c in &spec.constraints {
            let (lo, hi) = match c.op {
                ComparisonOp::Le => (f64::NEG_INFINITY, c.rhs),
                ComparisonOp::Ge => (c.rhs, f64::INFINITY),
                ComparisonOp::Eq => (c.rhs, c.rhs),
            };
            let i = m.add_row(&c.terms, lo, hi);
            let _: &Row = &m.rows[i];
        }
        let stem = case.name.replace(['/', ' '], "_");
        std::fs::write(dir.join(format!("{stem}.mps")), mps::write(&m)).unwrap();
        std::fs::write(dir.join(format!("{stem}.known")), known).unwrap();
        written += 1;
    }
    println!(
        "wrote {written} instances to {}, skipped {skipped} custom cases",
        dir.display()
    );
}
