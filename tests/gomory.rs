//! Root Gomory mixed-integer cuts (`SolveOptions::gomory_rounds`): cuts must never remove
//! an integer solution. Checked against brute force on small random programs (binaries
//! plus a general integer), through both `solve_with` and `solve_enumerate`, and on a
//! bound-magnitude case that once made a valid cut read as an infeasible relaxation.

use microlp::{
    CandidateAction, ComparisonOp, EnumerateReason, OptimizationDirection, Problem,
    SolveOptions, Variable,
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

/// Optimize `c·x + cz·z` over binaries `x` and an integer `z in [0, zmax]`,
/// subject to `a_k·x + w_k·z <= b_k`.
#[derive(Clone, Debug)]
struct Case {
    n: usize,
    c: Vec<i32>,
    cz: i32,
    zmax: i32,
    rows: Vec<(Vec<i32>, i32, i32)>,
}

fn case(seed: u64) -> Case {
    let mut rng = Lcg(seed ^ 0x2545_f491_4f6c_dd1d);
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

fn feasible(case: &Case, mask: u32, z: i32) -> bool {
    case.rows.iter().all(|(a, w, b)| {
        (0..case.n).filter(|&i| mask >> i & 1 == 1).map(|i| a[i]).sum::<i32>() + w * z <= *b
    })
}

fn objective(case: &Case, mask: u32, z: i32) -> i32 {
    (0..case.n).filter(|&i| mask >> i & 1 == 1).map(|i| case.c[i]).sum::<i32>() + case.cz * z
}

fn build(case: &Case, dir: OptimizationDirection) -> (Problem, Vec<Variable>, Variable) {
    let mut p = Problem::new(dir);
    let xs: Vec<Variable> = case.c.iter().map(|&c| p.add_binary_var(c as f64)).collect();
    let z = p.add_integer_var(case.cz as f64, (0, case.zmax));
    for (a, w, b) in &case.rows {
        let mut row: Vec<(Variable, f64)> =
            xs.iter().zip(a).map(|(&x, &a)| (x, a as f64)).collect();
        row.push((z, *w as f64));
        p.add_constraint(&row[..], ComparisonOp::Le, *b as f64);
    }
    (p, xs, z)
}

fn cuts(rounds: u32) -> SolveOptions {
    let mut o = SolveOptions::default();
    o.gomory_rounds = rounds;
    o
}

const DIRS: [OptimizationDirection; 2] =
    [OptimizationDirection::Maximize, OptimizationDirection::Minimize];

#[test]
fn cuts_keep_the_optimum() {
    let mut cut_any = false;
    for seed in 0..300 {
        let case = case(seed);
        for dir in DIRS {
            let points = (0..1u32 << case.n)
                .flat_map(|m| (0..=case.zmax).map(move |z| (m, z)))
                .filter(|&(m, z)| feasible(&case, m, z));
            let values = points.map(|(m, z)| objective(&case, m, z));
            let best = match dir {
                OptimizationDirection::Maximize => values.max(),
                OptimizationDirection::Minimize => values.min(),
            }
            .expect("x = 0, z = 0 is feasible");
            let (p, _, _) = build(&case, dir);
            let sol = p
                .solve_with(cuts(20))
                .unwrap_or_else(|e| panic!("{e:?} on {case:?} {dir:?}"))
                .into_solution()
                .expect("no limits");
            cut_any |= sol.stats().root_cuts > 0;
            assert!(
                (sol.objective() - best as f64).abs() < 1e-6,
                "optimum {} != brute force {best} with cuts on {case:?} {dir:?}",
                sol.objective()
            );
        }
    }
    assert!(cut_any, "no instance produced a cut; the test would prove nothing");
}

#[test]
fn enumeration_with_cuts_still_finds_every_point_within_the_cutoff() {
    // Binary-only instances (z fixed at 0), so a no-good cut excludes exactly one point.
    for seed in 0..150 {
        let case = Case { zmax: 0, cz: 0, ..case(seed) };
        let best = (0..1u32 << case.n)
            .filter(|&m| feasible(&case, m, 0))
            .map(|m| objective(&case, m, 0))
            .max()
            .unwrap();
        let cutoff = best - 2;
        let mut within: Vec<u32> = (0..1u32 << case.n)
            .filter(|&m| feasible(&case, m, 0) && objective(&case, m, 0) >= cutoff)
            .collect();

        let (p, xs, _) = build(&case, OptimizationDirection::Maximize);
        let mut found = vec![];
        let out = p
            .solve_enumerate(cuts(20), cutoff as f64, |vals, _| {
                let mask = (0..case.n).fold(0u32, |m, i| m | ((vals[xs[i].idx()] > 0.5) as u32) << i);
                found.push(mask);
                // sum_{b=0} x + sum_{b=1} (1 - x) >= 1
                let cut: Vec<(Variable, f64)> = xs
                    .iter()
                    .enumerate()
                    .map(|(i, &x)| (x, if mask >> i & 1 == 1 { -1.0 } else { 1.0 }))
                    .collect();
                let rhs = 1.0 - mask.count_ones() as f64;
                CandidateAction::Reject(vec![(cut.into_iter().collect(), ComparisonOp::Ge, rhs)])
            })
            .unwrap_or_else(|e| panic!("{e:?} on {case:?}"));
        assert_eq!(out.reason, EnumerateReason::Exhausted, "{case:?}");
        found.sort_unstable();
        within.sort_unstable();
        assert_eq!(found, within, "cuts changed the enumerated set on {case:?}");
    }
}

#[test]
fn cuts_tighten_the_root_bound() {
    // max 5x1 + 4x2 + 3x3 s.t. 2x1 + 3x2 + x3 <= 4, 4x1 + x2 + 2x3 <= 5 (binaries):
    // the LP optimum is fractional, and Gomory cuts move the root bound towards the
    // integer optimum (7, at x2 = x3 = 1).
    let (c, a1, a2) = ([5, 4, 3], [2, 3, 1], [4, 1, 2]);
    let best = (0..8u32)
        .filter(|m| {
            let dot = |a: [i32; 3]| (0..3).filter(|i| m >> i & 1 == 1).map(|i| a[i]).sum::<i32>();
            dot(a1) <= 4 && dot(a2) <= 5
        })
        .map(|m| (0..3).filter(|i| m >> i & 1 == 1).map(|i| c[i]).sum::<i32>())
        .max()
        .unwrap() as f64;
    let mut p = Problem::new(OptimizationDirection::Maximize);
    let x: Vec<Variable> = c.iter().map(|&ci| p.add_binary_var(ci as f64)).collect();
    let row = |a: [i32; 3]| x.iter().zip(a).map(|(&v, ai)| (v, ai as f64)).collect::<Vec<_>>();
    p.add_constraint(&row(a1)[..], ComparisonOp::Le, 4.0);
    p.add_constraint(&row(a2)[..], ComparisonOp::Le, 5.0);
    let plain = p.solve().unwrap().into_solution().unwrap();
    let cut = p.solve_with(cuts(10)).unwrap().into_solution().unwrap();
    assert!((plain.objective() - best).abs() < 1e-9);
    assert!((cut.objective() - best).abs() < 1e-9);
    let lp = cut.stats().root_lp_bound.unwrap();
    assert!(lp > best + 1e-6, "LP relaxation should be fractional here, got {lp}");
    assert!(cut.stats().root_cuts > 0);
    assert!(
        cut.stats().nodes_solved <= plain.stats().nodes_solved,
        "cuts should not need more nodes on this instance"
    );
}

#[test]
fn issue_42_large_integer_bound_with_cuts() {
    // A valid cut (y >= 1) passes through the only feasible point; with x's bound near
    // 2^k the re-solve's round-off once read that as an infeasible relaxation.
    for k in 20..=30u32 {
        for max in [(1i64 << k), (1i64 << k) + 1, (1i64 << k) + 2] {
            let mut p = Problem::new(OptimizationDirection::Maximize);
            let x = p.add_integer_var(1.0, (0, max as i32));
            let y = p.add_var(0.0, (0.0, 1.0));
            p.add_constraint([(x, 3.0), (y, 2.0)], ComparisonOp::Eq, 5.0);
            let sol = p
                .solve_with(cuts(20))
                .unwrap_or_else(|e| panic!("max={max}: {e:?}"))
                .into_solution()
                .unwrap();
            assert!((sol.var_value(x) - 1.0).abs() < 1e-6, "max={max}");
            assert!((sol.var_value(y) - 1.0).abs() < 1e-6, "max={max}");
        }
    }
}
