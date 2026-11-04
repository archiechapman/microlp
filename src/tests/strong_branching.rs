//! Reliability branching (`SolveOptions::strong_branch_reliability`) under
//! time limits: probes cut short by the deadline say nothing, and a search
//! sliced into many short calls ends on the optimum of one solved at once.

use crate::{ComparisonOp, OptimizationDirection, Problem, ResumeOptions, SolveOptions};
use rand::prelude::*;
use std::time::Duration;

/// A multi-constraint 0-1 knapsack: 30 items, 4 capacity rows at 40% of
/// their total weight.
fn knapsack(seed: u64) -> Problem {
    let mut rng = rand_pcg::Pcg64::seed_from_u64(seed);
    let mut p = Problem::new(OptimizationDirection::Maximize);
    let items: Vec<_> = (0..30)
        .map(|_| p.add_binary_var(rng.random_range(10..100) as f64))
        .collect();
    for _ in 0..4 {
        let weights: Vec<f64> = items
            .iter()
            .map(|_| rng.random_range(5..60) as f64)
            .collect();
        let capacity = 0.4 * weights.iter().sum::<f64>();
        p.add_constraint(
            items.iter().zip(&weights).map(|(&x, &w)| (x, w)),
            ComparisonOp::Le,
            capacity,
        );
    }
    p
}

#[test]
fn a_probing_search_sliced_by_time_limits_ends_on_the_optimum() {
    let mut interrupted = 0;
    for seed in 0..5 {
        let p = knapsack(seed);
        let plain = p.solve().unwrap().into_solution().unwrap();

        let mut options = SolveOptions::default();
        options.strong_branch_reliability = 8;
        let probed = p.solve_with(options.clone()).unwrap();
        assert!(probed.stats().strong_branch_lps > 0, "seed {seed}");
        let probed = probed.into_solution().unwrap();
        assert!(
            (probed.objective() - plain.objective()).abs() < 1e-6,
            "seed {seed}"
        );

        options.time_limit = Some(Duration::from_micros(200));
        let mut sliced = p.solve_with(options).unwrap();
        let mut slice = ResumeOptions::default();
        slice.time_limit = Some(Duration::from_micros(200));
        for _ in 0..1_000_000 {
            if sliced.is_optimal() {
                break;
            }
            interrupted += 1;
            sliced = sliced.resume_with(slice.clone()).unwrap();
        }
        let sliced = sliced.into_solution().unwrap();
        assert!(
            (sliced.objective() - plain.objective()).abs() < 1e-6,
            "seed {seed}: {} against {}",
            sliced.objective(),
            plain.objective()
        );
    }
    assert!(
        interrupted >= 10,
        "only {interrupted} slices were interrupted"
    );
}
