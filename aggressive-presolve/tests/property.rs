//! On random small MILPs built to trigger the reductions, the postsolved point
//! meets the original rows, and presolve on and off agree on the optimum.

use aggressive_presolve::{check, solve, CheckOptions, Direction, Model, Options, Path, Status};

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

/// Fixed vars, singleton rows, doubleton equations, forcing rows and
/// equalities with a free continuous var, over a mix of integer and
/// continuous vars, on top of random rows.
fn random_model(seed: u64) -> Model {
    let mut rng = Lcg(seed ^ 0x9e37_79b9_7f4a_7c15);
    let n = rng.range(2, 7) as usize;
    let mut m = Model::new(if rng.range(0, 1) == 0 {
        Direction::Minimize
    } else {
        Direction::Maximize
    });
    for _ in 0..n {
        let lo = rng.range(-2, 1) as f64;
        let hi = if rng.range(0, 5) == 0 {
            lo
        } else {
            rng.range(lo as i32, 4) as f64
        };
        let cost = rng.range(-5, 5) as f64;
        if rng.range(0, 2) == 0 {
            m.add_var(cost, lo, hi);
        } else {
            m.add_int_var(cost, lo, hi);
        }
    }
    // A free continuous var defined by an equality, used elsewhere.
    if rng.range(0, 1) == 0 {
        let w = m.add_var(rng.range(-3, 3) as f64, f64::NEG_INFINITY, f64::INFINITY);
        let mut terms = vec![(w, [-1.0, 2.0, -0.5][rng.range(0, 2) as usize])];
        for j in 0..n {
            if rng.range(0, 1) == 0 {
                terms.push((j, rng.range(-3, 3) as f64));
            }
        }
        m.add_eq(&terms, rng.range(-2, 2) as f64);
        let other = rng.range(0, n as i32 - 1) as usize;
        m.add_le(&[(w, 1.0), (other, 1.0)], rng.range(0, 6) as f64);
    }
    for _ in 0..rng.range(1, 6) {
        let mut a = vec![0.0; n];
        let kind = rng.range(0, 4);
        match kind {
            0 => a[rng.range(0, n as i32 - 1) as usize] = rng.range(1, 3) as f64 * [-1.0, 1.0][rng.range(0, 1) as usize],
            1 => {
                let i = rng.range(0, n as i32 - 1) as usize;
                let j = (i + 1) % n;
                a[i] = 1.0;
                a[j] = rng.range(-3, 3) as f64 * [1.0, 0.5][rng.range(0, 1) as usize];
            }
            _ => {
                for c in &mut a {
                    *c = rng.range(-4, 4) as f64;
                }
            }
        }
        let rhs = rng.range(-4, 6) as f64;
        let terms: Vec<(usize, f64)> = a.iter().enumerate().filter(|t| *t.1 != 0.0).map(|(j, &c)| (j, c)).collect();
        match rng.range(0, 3) {
            0 => m.add_le(&terms, rhs),
            1 => m.add_ge(&terms, rhs),
            2 => m.add_eq(&terms, rhs),
            _ => m.add_row(&terms, rhs - rng.range(0, 3) as f64, rhs),
        };
    }
    m
}

#[test]
fn presolve_on_and_off_agree_and_points_check() {
    let off = Options {
        presolve: aggressive_presolve::PresolveOptions {
            enabled: false,
            ..Default::default()
        },
        ..Options::default()
    };
    let on = Options::default();
    let (mut solved, mut reduced, mut retried, mut fell_back) = (0, 0, 0, 0);
    for seed in 0..3000u64 {
        let m = random_model(seed);
        let a = solve(&m, &off).unwrap_or_else(|e| panic!("seed {seed}: {e}"));
        let b = solve(&m, &on).unwrap_or_else(|e| panic!("seed {seed}: {e}"));
        let unbounded_or_inf = |s| matches!(s, Status::Unbounded | Status::Infeasible);
        assert!(
            a.status == b.status || unbounded_or_inf(a.status) && unbounded_or_inf(b.status),
            "seed {seed}: {:?} vs {:?}\n{m:?}",
            a.status,
            b.status
        );
        if b.status == Status::Optimal {
            solved += 1;
            let x = b.values.as_ref().unwrap();
            assert!(check(&m, x, &CheckOptions::default()).ok(), "seed {seed}");
            let (oa, ob) = (a.objective.unwrap(), b.objective.unwrap());
            assert!(
                (oa - ob).abs() <= 1e-6 * oa.abs().max(1.0),
                "seed {seed}: optimum {oa} without presolve, {ob} with ({:?})\n{m:?}",
                b.path
            );
        }
        match b.path {
            Path::Presolved => reduced += 1,
            Path::PresolvedRetry(_) => retried += 1,
            Path::Original(_) => fell_back += 1,
            Path::NoPresolve => unreachable!(),
        }
    }
    println!("{solved} optimal; paths: {reduced} presolved, {retried} retried, {fell_back} original");
    assert!(solved > 500);
}
