//! Node bound propagation (`SolveOptions::propagate_rounds`): deducing bounds from the
//! rows must never remove an integer solution, only save LP work.

use microlp::{
    ComparisonOp, Error, OptimizationDirection, Problem, SolveOptions, Variable,
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

/// `c·x + cz·z` over binaries `x` and an integer `z in [0, zmax]`, with rows
/// `a_k·x + w_k·z <= b_k` and, for some instances, a `>=` row that forces work.
#[derive(Clone, Debug)]
struct Case {
    n: usize,
    c: Vec<i32>,
    cz: i32,
    zmax: i32,
    rows: Vec<(Vec<i32>, i32, i32, bool)>, // coeffs, z coeff, rhs, is_ge
}

fn case(seed: u64) -> Case {
    let mut rng = Lcg(seed ^ 0xd1b5_4a32_d192_ed03);
    let n = rng.range(4, 9) as usize;
    let c = (0..n).map(|_| rng.range(-3, 9)).collect();
    let rows = (0..rng.range(1, 4))
        .map(|_| {
            let a: Vec<i32> = (0..n).map(|_| rng.range(0, 7)).collect();
            let w = rng.range(0, 4);
            let ge = rng.range(0, 3) == 0;
            let total: i32 = a.iter().sum();
            let b = if ge {
                (total / 4).max(1)
            } else {
                total / 2 + rng.range(0, 4)
            };
            (a, w, b, ge)
        })
        .collect();
    Case { n, c, cz: rng.range(-2, 5), zmax: rng.range(1, 4), rows }
}

fn feasible(case: &Case, mask: u32, z: i32) -> bool {
    case.rows.iter().all(|(a, w, b, ge)| {
        let lhs = (0..case.n).filter(|&i| mask >> i & 1 == 1).map(|i| a[i]).sum::<i32>() + w * z;
        if *ge { lhs >= *b } else { lhs <= *b }
    })
}

fn objective(case: &Case, mask: u32, z: i32) -> i32 {
    (0..case.n).filter(|&i| mask >> i & 1 == 1).map(|i| case.c[i]).sum::<i32>() + case.cz * z
}

fn brute_force(case: &Case, dir: OptimizationDirection) -> Option<i32> {
    let values = (0..1u32 << case.n)
        .flat_map(|m| (0..=case.zmax).map(move |z| (m, z)))
        .filter(|&(m, z)| feasible(case, m, z))
        .map(|(m, z)| objective(case, m, z));
    match dir {
        OptimizationDirection::Maximize => values.max(),
        OptimizationDirection::Minimize => values.min(),
    }
}

fn build(case: &Case, dir: OptimizationDirection) -> Problem {
    let mut p = Problem::new(dir);
    let xs: Vec<Variable> = case.c.iter().map(|&c| p.add_binary_var(c as f64)).collect();
    let z = p.add_integer_var(case.cz as f64, (0, case.zmax));
    for (a, w, b, ge) in &case.rows {
        let mut row: Vec<(Variable, f64)> =
            xs.iter().zip(a).map(|(&x, &a)| (x, a as f64)).collect();
        row.push((z, *w as f64));
        let op = if *ge { ComparisonOp::Ge } else { ComparisonOp::Le };
        p.add_constraint(&row[..], op, *b as f64);
    }
    p
}

fn options(propagate: u32, gomory: u32, strong_branch: u32) -> SolveOptions {
    let mut o = SolveOptions::default();
    o.propagate_rounds = propagate;
    o.gomory_rounds = gomory;
    o.strong_branch_reliability = strong_branch;
    o
}

const DIRS: [OptimizationDirection; 2] =
    [OptimizationDirection::Maximize, OptimizationDirection::Minimize];

#[test]
fn propagation_keeps_the_optimum() {
    let mut tightened_any = false;
    for seed in 0..300 {
        let case = case(seed);
        for dir in DIRS {
            let expected = brute_force(&case, dir);
            for (propagate, gomory, sb) in [(10, 0, 0), (2, 0, 0), (10, 10, 4)] {
                let result = build(&case, dir).solve_with(options(propagate, gomory, sb));
                match (expected, result) {
                    (Some(best), Ok(outcome)) => {
                        let sol = outcome.into_solution().expect("no limits");
                        tightened_any |= sol.stats().bounds_tightened > 0;
                        assert!(
                            (sol.objective() - best as f64).abs() < 1e-6,
                            "optimum {} != brute force {best} on {case:?} {dir:?}",
                            sol.objective()
                        );
                    }
                    (None, Err(Error::Infeasible)) => {}
                    (expected, result) => panic!(
                        "brute force {expected:?} vs solver {:?} on {case:?} {dir:?}",
                        result.map(|o| o.into_solution().map(|s| s.objective()))
                    ),
                }
            }
        }
    }
    assert!(tightened_any, "no instance propagated a bound; the test would prove nothing");
}

#[test]
fn propagation_matches_plain_verdicts() {
    for seed in 300..400 {
        let case = case(seed);
        for dir in DIRS {
            let plain = build(&case, dir).solve();
            let propagated = build(&case, dir).solve_with(options(10, 0, 0));
            match (plain, propagated) {
                (Ok(a), Ok(b)) => {
                    let (a, b) = (a.into_solution().unwrap(), b.into_solution().unwrap());
                    assert!(
                        (a.objective() - b.objective()).abs() < 1e-6,
                        "objective differs on {case:?} {dir:?}"
                    );
                }
                (Err(a), Err(b)) => assert_eq!(a, b, "error differs on {case:?} {dir:?}"),
                (a, b) => panic!("verdict differs on {case:?} {dir:?}: {a:?} vs {b:?}"),
            }
        }
    }
}

#[test]
fn propagation_tightens_a_declared_bound() {
    // 3z + 2x <= 7 with z declared over [0, 10]: the row alone gives z <= 2 (7/3, rounded
    // down because z is integer), before any LP.
    let mut p = Problem::new(OptimizationDirection::Maximize);
    let z = p.add_integer_var(5.0, (0, 10));
    let x = p.add_binary_var(4.0);
    p.add_constraint(&[(z, 3.0), (x, 2.0)], ComparisonOp::Le, 7.0);
    let sol = p
        .solve_with(options(10, 0, 0))
        .unwrap()
        .into_solution()
        .unwrap();
    assert!((sol.objective() - 10.0).abs() < 1e-9, "got {}", sol.objective());
    assert!(
        sol.stats().bounds_tightened > 0,
        "expected propagation to tighten z's upper bound"
    );
    // Same answer without propagation.
    let plain = p.solve().unwrap().into_solution().unwrap();
    assert!((plain.objective() - 10.0).abs() < 1e-9);
}

#[test]
fn propagation_reports_infeasibility_without_an_lp() {
    // Three binaries with coefficient 3 cannot reach 10: the rows alone prove it.
    let mut p = Problem::new(OptimizationDirection::Maximize);
    let xs: Vec<Variable> = (0..3).map(|_| p.add_binary_var(1.0)).collect();
    p.add_constraint(
        &xs.iter().map(|&x| (x, 3.0)).collect::<Vec<_>>()[..],
        ComparisonOp::Ge,
        10.0,
    );
    assert_eq!(
        p.solve_with(options(10, 0, 0)).unwrap_err(),
        Error::Infeasible
    );
    // Same verdict without propagation.
    assert_eq!(p.solve().unwrap_err(), Error::Infeasible);
}

#[test]
fn propagation_prunes_nodes_without_solving_them() {
    // Equality knapsack with a tight capacity: branching a variable up forces others down,
    // which propagation detects (and sometimes contradicts) before the node's LP.
    let mut p = Problem::new(OptimizationDirection::Maximize);
    let xs: Vec<Variable> = [7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]
        .iter()
        .map(|&c| p.add_binary_var(c))
        .collect();
    let weights = [9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0];
    p.add_constraint(
        &xs.iter().zip(weights).map(|(&x, w)| (x, w)).collect::<Vec<_>>()[..],
        ComparisonOp::Eq,
        21.0,
    );
    let plain = p.solve().unwrap().into_solution().unwrap();
    let propagated = p
        .solve_with(options(10, 0, 0))
        .unwrap()
        .into_solution()
        .unwrap();
    assert!((plain.objective() - propagated.objective()).abs() < 1e-9);
    let stats = propagated.stats();
    assert!(
        stats.bounds_tightened > 0 || stats.propagation_prunes > 0,
        "expected propagation to do something on a tight equality knapsack"
    );
    assert!(
        propagated.stats().nodes_solved <= plain.stats().nodes_solved,
        "propagation should not need more nodes here: {} vs {}",
        propagated.stats().nodes_solved,
        plain.stats().nodes_solved
    );
}
