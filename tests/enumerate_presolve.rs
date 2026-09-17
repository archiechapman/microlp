//! `solve_enumerate` with `SolveOptions::presolve`: presolve keeps every feasible
//! point, candidates arrive in original variables, and rejection rows written over
//! eliminated variables are mapped onto the reduced problem. So with no-good cuts
//! (d = 1) the candidate sets with and without presolve must be identical.

use microlp::{
    CandidateAction, ComparisonOp, EnumerateReason, LinearExpr, OptimizationDirection, Problem,
    SolveOptions, Variable,
};
use std::collections::BTreeSet;

#[derive(Clone, Copy)]
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 32) as u32
    }

    fn range(&mut self, lo: i32, hi: i32) -> i32 {
        lo + (self.next() % (hi - lo + 1) as u32) as i32
    }
}

/// Binaries with structure presolve reduces: a fixed binary, equal pairs
/// (`x_i - x_j = 0`), a forcing row, a continuous variable defined by a doubleton
/// equation, and a knapsack row.
fn build(seed: u64) -> (Problem, Vec<Variable>, f64) {
    let mut rng = Lcg(seed ^ 0x5851_f42d_4c95_7f2d);
    let n = rng.range(5, 8) as usize;
    let dir = if seed % 2 == 0 {
        OptimizationDirection::Maximize
    } else {
        OptimizationDirection::Minimize
    };
    let mut problem = Problem::new(dir);
    let xs: Vec<Variable> = (0..n)
        .map(|_| problem.add_binary_var(rng.range(-3, 6) as f64))
        .collect();
    // A fixed binary.
    let f = rng.range(0, n as i32 - 1) as usize;
    problem.add_constraint([(xs[f], 1.0)], ComparisonOp::Eq, rng.range(0, 1) as f64);
    // Equal pairs.
    for _ in 0..rng.range(0, 2) {
        let i = rng.range(0, n as i32 - 1) as usize;
        let j = (i + 1 + rng.range(0, n as i32 - 2) as usize) % n;
        problem.add_constraint([(xs[i], 1.0), (xs[j], -1.0)], ComparisonOp::Eq, 0.0);
    }
    // A forcing row on two binaries.
    if rng.range(0, 2) == 0 {
        let i = rng.range(0, n as i32 - 1) as usize;
        let j = (i + 1) % n;
        problem.add_constraint([(xs[i], 1.0), (xs[j], 1.0)], ComparisonOp::Le, 0.0);
    }
    // y = 2·x_k + 0.5, costed, bounds implied.
    let k = rng.range(0, n as i32 - 1) as usize;
    let y = problem.add_var(rng.range(-2, 2) as f64 * 0.5, (0.0, 10.0));
    problem.add_constraint([(y, 1.0), (xs[k], -2.0)], ComparisonOp::Eq, 0.5);
    // Knapsack.
    let a: Vec<i32> = (0..n).map(|_| rng.range(0, 5)).collect();
    let b = a.iter().sum::<i32>() / 2 + rng.range(0, 2);
    problem.add_constraint(
        xs.iter().zip(&a).map(|(&x, &a)| (x, a as f64)),
        ComparisonOp::Le,
        b as f64,
    );
    let cutoff = match dir {
        OptimizationDirection::Maximize => -3.0,
        OptimizationDirection::Minimize => 6.0,
    };
    (problem, xs, cutoff)
}

fn enumerate(problem: &Problem, xs: &[Variable], cutoff: f64, presolve: bool) -> BTreeSet<Vec<u8>> {
    let mut options = SolveOptions::default();
    options.presolve = presolve;
    let mut found = BTreeSet::new();
    let outcome = problem
        .solve_enumerate(options, cutoff, |values, _objective| {
            let point: Vec<u8> = xs.iter().map(|x| values[x.idx()].round() as u8).collect();
            assert!(found.insert(point.clone()), "candidate repeated: {point:?}");
            // No-good cut: Σ_{p=0} x + Σ_{p=1} (1 - x) >= 1.
            let mut expr = LinearExpr::empty();
            let mut ones = 0.0;
            for (&x, &p) in xs.iter().zip(&point) {
                if p == 0 {
                    expr.add(x, 1.0);
                } else {
                    expr.add(x, -1.0);
                    ones += 1.0;
                }
            }
            CandidateAction::Reject(vec![(expr, ComparisonOp::Ge, 1.0 - ones)])
        })
        .expect("enumeration runs");
    assert_eq!(outcome.reason, EnumerateReason::Exhausted);
    found
}

#[test]
fn presolve_enumerates_the_same_points() {
    let mut nonempty = 0;
    for seed in 0..200 {
        let (problem, xs, cutoff) = build(seed);
        let plain = enumerate(&problem, &xs, cutoff, false);
        let pre = enumerate(&problem, &xs, cutoff, true);
        assert_eq!(plain, pre, "seed {seed}");
        nonempty += usize::from(!plain.is_empty());
    }
    assert!(nonempty > 50, "only {nonempty} instances had candidates");
}

#[test]
fn candidates_carry_original_objective_and_continuous_values() {
    let mut problem = Problem::new(OptimizationDirection::Maximize);
    let x = problem.add_binary_var(1.0);
    let z = problem.add_binary_var(1.0);
    let fixed = problem.add_binary_var(5.0);
    let y = problem.add_var(1.0, (0.0, 10.0));
    problem.add_constraint([(fixed, 1.0)], ComparisonOp::Ge, 1.0);
    problem.add_constraint([(y, 1.0), (x, -2.0)], ComparisonOp::Eq, 0.5);
    problem.add_constraint([(x, 1.0), (z, 1.0)], ComparisonOp::Le, 1.0);
    let mut options = SolveOptions::default();
    options.presolve = true;
    let mut seen = Vec::new();
    let outcome = problem
        .solve_enumerate(options, 0.0, |values, objective| {
            let expected = values[x.idx()] + values[z.idx()] + 5.0 * values[fixed.idx()] + values[y.idx()];
            assert!((objective - expected).abs() < 1e-9);
            assert!((values[y.idx()] - (2.0 * values[x.idx()] + 0.5)).abs() < 1e-9);
            assert_eq!(values[fixed.idx()], 1.0);
            seen.push((values[x.idx()], values[z.idx()]));
            let mut expr = LinearExpr::empty();
            let mut rhs = 1.0;
            for (v, val) in [(x, values[x.idx()]), (z, values[z.idx()]), (fixed, 1.0)] {
                if val == 0.0 {
                    expr.add(v, 1.0);
                } else {
                    expr.add(v, -1.0);
                    rhs -= 1.0;
                }
            }
            CandidateAction::Reject(vec![(expr, ComparisonOp::Ge, rhs)])
        })
        .unwrap();
    assert_eq!(outcome.reason, EnumerateReason::Exhausted);
    seen.sort_by(|a, b| a.partial_cmp(b).unwrap());
    assert_eq!(seen, vec![(0.0, 0.0), (0.0, 1.0), (1.0, 0.0)]);
}
