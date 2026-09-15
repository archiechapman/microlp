//! Single-tree enumeration (`Problem::solve_enumerate`) checked against brute force on
//! small random binary programs: the callback rejects every candidate `c` with the
//! cut `sum_{c_i=0} x_i + sum_{c_i=1} (1 - x_i) >= d` over the binaries (a no-good
//! cut for d = 1; for larger d it excludes everything within distance d - 1 of `c`).
//!
//! What must hold for every run:
//! - every candidate is feasible and within the cutoff;
//! - candidates are pairwise at distance >= d;
//! - the run ends `Exhausted`, and exhaustion is real: every feasible point within
//!   the cutoff is within distance < d of some candidate (so, at d = 1, the
//!   candidates are exactly the points within the cutoff).

use microlp::{
    CandidateAction, ComparisonOp, EnumerateReason, Error, LinearExpr, OptimizationDirection,
    Problem, SolveOptions, Variable,
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

/// Optimize `c·x + y` over binaries `x` and, when `link` is set, a continuous
/// `y in [0, cap]` with `y <= link·x`; subject to `a_k·x <= b_k`.
#[derive(Clone, Debug)]
struct Case {
    n: usize,
    c: Vec<i32>,
    rows: Vec<(Vec<i32>, i32)>,
    link: Option<(Vec<i32>, f64)>,
}

fn case(seed: u64, with_continuous: bool) -> Case {
    let mut rng = Lcg(seed ^ 0x9e37_79b9_7f4a_7c15);
    let n = rng.range(5, 10) as usize;
    let c = (0..n).map(|_| rng.range(-2, 8)).collect();
    let rows = (0..rng.range(1, 3))
        .map(|_| {
            let a: Vec<i32> = (0..n).map(|_| rng.range(0, 6)).collect();
            let b = a.iter().sum::<i32>() / 2 + rng.range(0, 3);
            (a, b)
        })
        .collect();
    let link = with_continuous.then(|| {
        let w = (0..n).map(|_| rng.range(0, 3)).collect();
        (w, rng.range(1, 6) as f64 + 0.5)
    });
    Case { n, c, rows, link }
}

fn bit(mask: u32, i: usize) -> bool {
    mask >> i & 1 == 1
}

fn feasible(case: &Case, mask: u32) -> bool {
    case.rows.iter().all(|(a, b)| {
        (0..case.n).filter(|&i| bit(mask, i)).map(|i| a[i]).sum::<i32>() <= *b
    })
}

/// Objective of `mask` with the continuous part at its optimum (max direction), or
/// at 0 (min direction; its coefficient is positive).
fn objective(case: &Case, mask: u32, dir: OptimizationDirection) -> f64 {
    let lin: i32 = (0..case.n).filter(|&i| bit(mask, i)).map(|i| case.c[i]).sum();
    let y = match (&case.link, dir) {
        (Some((w, cap)), OptimizationDirection::Maximize) => {
            let s: i32 = (0..case.n).filter(|&i| bit(mask, i)).map(|i| w[i]).sum();
            (s as f64).min(*cap)
        }
        _ => 0.0,
    };
    lin as f64 + y
}

fn build(case: &Case, dir: OptimizationDirection) -> (Problem, Vec<Variable>) {
    let mut p = Problem::new(dir);
    let xs: Vec<Variable> = case.c.iter().map(|&c| p.add_binary_var(c as f64)).collect();
    for (a, b) in &case.rows {
        let row: Vec<(Variable, f64)> = xs.iter().zip(a).map(|(&x, &a)| (x, a as f64)).collect();
        p.add_constraint(&row[..], ComparisonOp::Le, *b as f64);
    }
    if let Some((w, cap)) = &case.link {
        let y = p.add_var(1.0, (0.0, *cap));
        let mut row: Vec<(Variable, f64)> =
            xs.iter().zip(w).map(|(&x, &w)| (x, -(w as f64))).collect();
        row.push((y, 1.0));
        p.add_constraint(&row[..], ComparisonOp::Le, 0.0);
    }
    (p, xs)
}

fn distance_cut(xs: &[Variable], mask: u32, d: u32) -> (LinearExpr, ComparisonOp, f64) {
    let cut: LinearExpr = xs
        .iter()
        .enumerate()
        .map(|(i, &x)| (x, if bit(mask, i) { -1.0 } else { 1.0 }))
        .collect();
    (cut, ComparisonOp::Ge, d as f64 - mask.count_ones() as f64)
}

/// Enumerate with distance cuts; returns the candidates' masks and the reason.
fn enumerate(
    case: &Case,
    dir: OptimizationDirection,
    cutoff: f64,
    d: u32,
) -> (Vec<u32>, EnumerateReason) {
    let (p, xs) = build(case, dir);
    let mut found = vec![];
    let out = p
        .solve_enumerate(SolveOptions::default(), cutoff, |vals, obj| {
            let mask = xs
                .iter()
                .enumerate()
                .fold(0u32, |m, (i, x)| m | ((vals[x.idx()] > 0.5) as u32) << i);
            let expected = objective(case, mask, dir);
            assert!(
                (obj - expected).abs() < 1e-6,
                "candidate objective {obj} != {expected} for {mask:b} in {case:?}"
            );
            found.push(mask);
            CandidateAction::Reject(vec![distance_cut(&xs, mask, d)])
        })
        .unwrap_or_else(|e| panic!("solve_enumerate failed: {e:?} on {case:?}"));
    assert_eq!(out.candidates as usize, found.len());
    (found, out.reason)
}

fn check(case: &Case, dir: OptimizationDirection, delta: f64, d: u32) {
    let points: Vec<u32> = (0..1u32 << case.n).filter(|&m| feasible(case, m)).collect();
    let best = points
        .iter()
        .map(|&m| objective(case, m, dir))
        .fold(None, |acc: Option<f64>, v| {
            Some(match (acc, dir) {
                (None, _) => v,
                (Some(a), OptimizationDirection::Maximize) => a.max(v),
                (Some(a), OptimizationDirection::Minimize) => a.min(v),
            })
        })
        .expect("x = 0 is always feasible");
    let (cutoff, is_within): (f64, Box<dyn Fn(f64) -> bool>) = match dir {
        OptimizationDirection::Maximize => (best - delta, Box::new(move |v| v >= best - delta - 1e-9)),
        OptimizationDirection::Minimize => (best + delta, Box::new(move |v| v <= best + delta + 1e-9)),
    };
    let within: Vec<u32> = points
        .iter()
        .copied()
        .filter(|&m| is_within(objective(case, m, dir)))
        .collect();

    let (found, reason) = enumerate(case, dir, cutoff, d);
    let ctx = format!("{dir:?} delta={delta} d={d} {case:?}\nfound={found:?}\nwithin={within:?}");
    assert_eq!(reason, EnumerateReason::Exhausted, "{ctx}");
    for &m in &found {
        assert!(within.contains(&m), "candidate {m:b} beyond the cutoff: {ctx}");
    }
    for (i, &a) in found.iter().enumerate() {
        for &b in &found[i + 1..] {
            assert!((a ^ b).count_ones() >= d, "{a:b} and {b:b} closer than d: {ctx}");
        }
    }
    for &m in &within {
        assert!(
            found.iter().any(|&p| (p ^ m).count_ones() < d),
            "point {m:b} is at distance >= d from every candidate (not exhausted): {ctx}"
        );
    }
    if d == 1 {
        let (mut a, mut b) = (found.clone(), within.clone());
        a.sort_unstable();
        b.sort_unstable();
        assert_eq!(a, b, "d = 1 must enumerate every point within the cutoff: {ctx}");
    }
}

const DIRS: [OptimizationDirection; 2] =
    [OptimizationDirection::Maximize, OptimizationDirection::Minimize];

#[test]
fn d1_enumerates_every_point_within_the_cutoff() {
    for seed in 0..60 {
        for dir in DIRS {
            for delta in [0.0, 1.0, 3.0] {
                check(&case(seed, false), dir, delta, 1);
            }
        }
    }
}

#[test]
fn d_separated_sets_are_maximal() {
    for seed in 0..60 {
        for dir in DIRS {
            for (delta, d) in [(0.0, 2), (2.0, 2), (4.0, 3)] {
                check(&case(seed, false), dir, delta, d);
            }
        }
    }
}

#[test]
fn continuous_variables_take_their_optimal_values() {
    for seed in 0..60 {
        for (delta, d) in [(0.0, 1), (1.5, 1), (3.0, 2)] {
            check(&case(seed, true), OptimizationDirection::Maximize, delta, d);
        }
    }
}

#[test]
fn stop_ends_the_run() {
    // Five independent binaries with equal value: every subset of size 3 ties.
    let mut p = Problem::new(OptimizationDirection::Maximize);
    let xs: Vec<Variable> = (0..5).map(|_| p.add_binary_var(1.0)).collect();
    let row: Vec<(Variable, f64)> = xs.iter().map(|&x| (x, 1.0)).collect();
    p.add_constraint(&row[..], ComparisonOp::Le, 3.0);
    let mut seen = 0;
    let out = p
        .solve_enumerate(SolveOptions::default(), 3.0, |vals, obj| {
            assert_eq!(obj, 3.0);
            seen += 1;
            if seen == 4 {
                return CandidateAction::Stop;
            }
            let mask = (0..5).fold(0u32, |m, i| m | ((vals[i] > 0.5) as u32) << i);
            CandidateAction::Reject(vec![distance_cut(&xs, mask, 1)])
        })
        .unwrap();
    assert_eq!(out.reason, EnumerateReason::Stopped);
    assert_eq!(seen, 4);
    assert_eq!(out.lazy_rows, 3);
}

#[test]
fn ties_are_all_found_at_d1() {
    // C(5,3) = 10 optimal solutions, all tied.
    let mut p = Problem::new(OptimizationDirection::Maximize);
    let xs: Vec<Variable> = (0..5).map(|_| p.add_binary_var(1.0)).collect();
    let row: Vec<(Variable, f64)> = xs.iter().map(|&x| (x, 1.0)).collect();
    p.add_constraint(&row[..], ComparisonOp::Le, 3.0);
    let mut masks = vec![];
    let out = p
        .solve_enumerate(SolveOptions::default(), 3.0, |vals, _| {
            let mask = (0..5).fold(0u32, |m, i| m | ((vals[i] > 0.5) as u32) << i);
            masks.push(mask);
            CandidateAction::Reject(vec![distance_cut(&xs, mask, 1)])
        })
        .unwrap();
    assert_eq!(out.reason, EnumerateReason::Exhausted);
    masks.sort_unstable();
    masks.dedup();
    assert_eq!(masks.len(), 10);
    assert!(masks.iter().all(|m| m.count_ones() == 3));
}

#[test]
fn reject_rows_must_cut_off_the_candidate() {
    let mut p = Problem::new(OptimizationDirection::Maximize);
    let x = p.add_binary_var(1.0);
    let y = p.add_binary_var(1.0);
    p.add_constraint(&[(x, 1.0), (y, 1.0)], ComparisonOp::Le, 1.0);
    let res = p.solve_enumerate(SolveOptions::default(), 1.0, |_, _| {
        // x + y <= 5 is already satisfied: it cuts nothing off.
        CandidateAction::Reject(vec![(
            [(x, 1.0), (y, 1.0)].into_iter().collect(),
            ComparisonOp::Le,
            5.0,
        )])
    });
    assert!(matches!(res, Err(Error::InvalidOperation(_))), "{res:?}");
}

#[test]
fn nothing_within_the_cutoff_is_exhausted_immediately() {
    let mut p = Problem::new(OptimizationDirection::Maximize);
    let x = p.add_binary_var(2.0);
    let y = p.add_binary_var(3.0);
    p.add_constraint(&[(x, 1.0), (y, 1.0)], ComparisonOp::Le, 1.0);
    let out = p
        .solve_enumerate(SolveOptions::default(), 10.0, |_, _| {
            panic!("no candidate should be reported")
        })
        .unwrap();
    assert_eq!(out.reason, EnumerateReason::Exhausted);
    assert_eq!(out.candidates, 0);
}
