//! Objective cutoff in the dual simplex: a node LP is abandoned as soon as its bound
//! passes the pruning cutoff. The bound is only valid while the basis is dual feasible, so
//! the risk is pruning a node that could still have improved — these check it doesn't.

use microlp::{ComparisonOp, OptimizationDirection, Problem, SolveOptions, Variable};

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

struct Case {
    n: usize,
    c: Vec<i32>,
    rows: Vec<(Vec<i32>, i32)>,
}

fn case(seed: u64) -> Case {
    let mut rng = Lcg(seed ^ 0x6a09_e667_f3bc_c908);
    let n = rng.range(5, 10) as usize;
    let c = (0..n).map(|_| rng.range(1, 9)).collect();
    let rows = (0..rng.range(1, 3))
        .map(|_| {
            let a: Vec<i32> = (0..n).map(|_| rng.range(1, 7)).collect();
            let b = a.iter().sum::<i32>() / 2;
            (a, b)
        })
        .collect();
    Case { n, c, rows }
}

fn brute_force(case: &Case) -> i32 {
    (0..1u32 << case.n)
        .filter(|m| {
            case.rows.iter().all(|(a, b)| {
                (0..case.n).filter(|&i| m >> i & 1 == 1).map(|i| a[i]).sum::<i32>() <= *b
            })
        })
        .map(|m| (0..case.n).filter(|&i| m >> i & 1 == 1).map(|i| case.c[i]).sum::<i32>())
        .max()
        .unwrap()
}

fn build(case: &Case) -> Problem {
    let mut p = Problem::new(OptimizationDirection::Maximize);
    let xs: Vec<Variable> = case.c.iter().map(|&c| p.add_binary_var(c as f64)).collect();
    for (a, b) in &case.rows {
        let row: Vec<(Variable, f64)> = xs.iter().zip(a).map(|(&x, &a)| (x, a as f64)).collect();
        p.add_constraint(&row[..], ComparisonOp::Le, *b as f64);
    }
    p
}

#[test]
fn cutoff_pruning_keeps_the_optimum() {
    let mut pruned_any = false;
    for seed in 0..400 {
        let case = case(seed);
        let best = brute_force(&case) as f64;
        let sol = build(&case).solve().unwrap().into_solution().unwrap();
        pruned_any |= sol.stats().cutoff_prunes > 0;
        assert!(
            (sol.objective() - best).abs() < 1e-6,
            "optimum {} != brute force {best} on seed {seed}",
            sol.objective()
        );
    }
    assert!(
        pruned_any,
        "no node was pruned at the cutoff; the test would prove nothing"
    );
}

#[test]
fn cutoff_pruning_survives_every_option_combination() {
    // The cutoff interacts with cuts (extra rows), strong branching (probe solves that
    // must not inherit it) and propagation (bounds changing before the LP).
    for seed in 400..460 {
        let case = case(seed);
        let best = brute_force(&case) as f64;
        for (gomory, sb, propagate) in [(10, 0, 0), (0, 8, 0), (0, 0, 5), (10, 8, 5)] {
            let mut o = SolveOptions::default();
            o.gomory_rounds = gomory;
            o.strong_branch_reliability = sb;
            o.propagate_rounds = propagate;
            let sol = build(&case)
                .solve_with(o)
                .unwrap()
                .into_solution()
                .unwrap();
            assert!(
                (sol.objective() - best).abs() < 1e-6,
                "optimum {} != brute force {best} on seed {seed} with ({gomory}, {sb}, {propagate})",
                sol.objective()
            );
        }
    }
}

#[test]
fn a_node_limited_solve_still_reports_its_nodes() {
    // A cutoff prune ends a node's LP early but the node still counts as solved, so the
    // node budget cannot be bypassed by pruning.
    let case = case(7);
    let mut o = SolveOptions::default();
    o.node_limit = Some(3);
    let outcome = build(&case).solve_with(o).unwrap();
    assert!(outcome.stats().nodes_solved <= 3);
}
