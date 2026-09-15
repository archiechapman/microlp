//! Reliability branching (`SolveOptions::strong_branch_reliability`): probing children to
//! calibrate pseudocosts must change only which variable is branched on, never the answer
//! or the search's budget contract.

use microlp::{
    ComparisonOp, Error, OptimizationDirection, Problem, SolveOptions, TerminationReason,
    Variable,
};

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

/// `c·x + cz·z` over binaries `x` and an integer `z in [0, zmax]`, subject to
/// `a_k·x + w_k·z <= b_k`.
#[derive(Clone, Debug)]
struct Case {
    n: usize,
    c: Vec<i32>,
    cz: i32,
    zmax: i32,
    rows: Vec<(Vec<i32>, i32, i32)>,
}

fn case(seed: u64) -> Case {
    let mut rng = Lcg(seed ^ 0x9e37_79b9_7f4a_7c15);
    let n = rng.range(4, 9) as usize;
    let c = (0..n).map(|_| rng.range(-3, 9)).collect();
    let rows = (0..rng.range(1, 4))
        .map(|_| {
            let a: Vec<i32> = (0..n).map(|_| rng.range(0, 7)).collect();
            let w = rng.range(0, 4);
            let b = a.iter().sum::<i32>() / 2 + rng.range(0, 4);
            (a, w, b)
        })
        .collect();
    Case { n, c, cz: rng.range(-2, 5), zmax: rng.range(1, 4), rows }
}

fn brute_force(case: &Case, dir: OptimizationDirection) -> i32 {
    let values = (0..1u32 << case.n)
        .flat_map(|m| (0..=case.zmax).map(move |z| (m, z)))
        .filter(|&(m, z)| {
            case.rows.iter().all(|(a, w, b)| {
                (0..case.n).filter(|&i| m >> i & 1 == 1).map(|i| a[i]).sum::<i32>() + w * z <= *b
            })
        })
        .map(|(m, z)| {
            (0..case.n).filter(|&i| m >> i & 1 == 1).map(|i| case.c[i]).sum::<i32>() + case.cz * z
        });
    match dir {
        OptimizationDirection::Maximize => values.max(),
        OptimizationDirection::Minimize => values.min(),
    }
    .expect("x = 0, z = 0 is feasible")
}

fn build(case: &Case, dir: OptimizationDirection) -> Problem {
    let mut p = Problem::new(dir);
    let xs: Vec<Variable> = case.c.iter().map(|&c| p.add_binary_var(c as f64)).collect();
    let z = p.add_integer_var(case.cz as f64, (0, case.zmax));
    for (a, w, b) in &case.rows {
        let mut row: Vec<(Variable, f64)> =
            xs.iter().zip(a).map(|(&x, &a)| (x, a as f64)).collect();
        row.push((z, *w as f64));
        p.add_constraint(&row[..], ComparisonOp::Le, *b as f64);
    }
    p
}

fn options(reliability: u32, gomory: u32) -> SolveOptions {
    let mut o = SolveOptions::default();
    o.strong_branch_reliability = reliability;
    o.gomory_rounds = gomory;
    o
}

const DIRS: [OptimizationDirection; 2] =
    [OptimizationDirection::Maximize, OptimizationDirection::Minimize];

#[test]
fn strong_branching_keeps_the_optimum() {
    let mut probed_any = false;
    for seed in 0..300 {
        let case = case(seed);
        for dir in DIRS {
            let best = brute_force(&case, dir) as f64;
            for (reliability, gomory) in [(8, 0), (4, 10)] {
                let sol = build(&case, dir)
                    .solve_with(options(reliability, gomory))
                    .unwrap_or_else(|e| panic!("{e:?} on {case:?} {dir:?}"))
                    .into_solution()
                    .expect("no limits");
                probed_any |= sol.stats().strong_branch_lps > 0;
                assert!(
                    (sol.objective() - best).abs() < 1e-6,
                    "optimum {} != brute force {best} on {case:?} {dir:?}",
                    sol.objective()
                );
            }
        }
    }
    assert!(probed_any, "no instance strong-branched; the test would prove nothing");
}

#[test]
fn strong_branching_matches_plain_branching_verdicts() {
    for seed in 300..380 {
        let case = case(seed);
        for dir in DIRS {
            let plain = build(&case, dir).solve().unwrap().into_solution().unwrap();
            let probed = build(&case, dir)
                .solve_with(options(8, 0))
                .unwrap()
                .into_solution()
                .unwrap();
            assert!(
                (plain.objective() - probed.objective()).abs() < 1e-6,
                "objective differs on {case:?} {dir:?}"
            );
        }
    }
}

#[test]
fn a_zero_node_budget_still_solves_no_nodes() {
    // Integer-infeasible (x must be 0.5) with an unbounded relaxation: strong branching
    // proves both children infeasible at once, but a zero node budget must still return a
    // resumable NodeLimit rather than a verdict.
    let mut p = Problem::new(OptimizationDirection::Minimize);
    let x = p.add_integer_var(7.0, (0, 1));
    let _y = p.add_var(-1.0, (0.0, f64::INFINITY));
    p.add_constraint(&[(x, 1.0)], ComparisonOp::Eq, 0.5);

    let mut o = options(8, 0);
    o.node_limit = Some(0);
    let outcome = p.solve_with(o).unwrap();
    assert!(outcome.solution().is_none());
    assert_eq!(outcome.termination_reason(), TerminationReason::NodeLimit);
    assert_eq!(outcome.stats().nodes_solved, 0);

    // With a budget, the same model is infeasible, with or without probing.
    assert_eq!(p.solve_with(options(8, 0)).unwrap_err(), Error::Infeasible);
    assert_eq!(p.solve().unwrap_err(), Error::Infeasible);
}

#[test]
fn probes_are_counted_and_bounded() {
    // 0/1 knapsack with a fractional LP optimum: probing happens, and stays within the
    // per-node candidate cap times the nodes solved (plus the root).
    let mut p = Problem::new(OptimizationDirection::Maximize);
    let xs: Vec<Variable> = [8.0, 11.0, 6.0, 4.0, 7.0, 3.0]
        .iter()
        .map(|&c| p.add_binary_var(c))
        .collect();
    let weights = [5.0, 7.0, 4.0, 3.0, 5.0, 2.0];
    p.add_constraint(
        &xs.iter().zip(weights).map(|(&x, w)| (x, w)).collect::<Vec<_>>()[..],
        ComparisonOp::Le,
        13.0,
    );
    let sol = p.solve_with(options(8, 0)).unwrap().into_solution().unwrap();
    let plain = p.solve().unwrap().into_solution().unwrap();
    assert!((sol.objective() - plain.objective()).abs() < 1e-9);
    let probes = sol.stats().strong_branch_lps;
    assert!(probes > 0, "expected strong branching on a fractional knapsack");
    // 8 candidates x 2 sides per node, and the root is one extra branching decision.
    assert!(
        probes <= 16 * (sol.stats().nodes_solved + 1),
        "{probes} probes for {} nodes",
        sol.stats().nodes_solved
    );
}
