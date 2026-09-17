//! Presolve (`SolveOptions::presolve`): the reductions keep every feasible point,
//! so optima, verdicts and reported values must match a plain solve.

use microlp::{ComparisonOp, Error, OptimizationDirection, Problem, SolveOptions, Variable};

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

type Row = (Vec<(Variable, f64)>, ComparisonOp, f64);

fn options(presolve: bool) -> SolveOptions {
    let mut options = SolveOptions::default();
    options.presolve = presolve;
    options
}

/// An integer model built to trigger the reductions: fixed variables,
/// singleton rows, scaled duplicate rows, doubleton equations with unit
/// coefficients, and forcing rows, on top of random rows.
#[derive(Clone, Debug)]
struct Case {
    lo: Vec<i32>,
    hi: Vec<i32>,
    c: Vec<i32>,
    rows: Vec<(Vec<i32>, ComparisonOp, i32)>,
}

fn case(seed: u64) -> Case {
    let mut rng = Lcg(seed ^ 0x9e37_79b9_7f4a_7c15);
    let n = rng.range(2, 5) as usize;
    let mut lo = vec![0; n];
    let mut hi = vec![0; n];
    for v in 0..n {
        lo[v] = rng.range(-2, 1);
        hi[v] = if rng.range(0, 5) == 0 {
            lo[v]
        } else {
            rng.range(lo[v], 3)
        };
    }
    let c = (0..n).map(|_| rng.range(-5, 5)).collect();
    let mut rows = Vec::new();
    for _ in 0..rng.range(1, 6) {
        let mut a = vec![0; n];
        let op = [ComparisonOp::Le, ComparisonOp::Ge, ComparisonOp::Eq][rng.range(0, 2) as usize];
        match rng.range(0, 4) {
            0 => {
                a[rng.range(0, n as i32 - 1) as usize] =
                    rng.range(1, 3) * [-1, 1][rng.range(0, 1) as usize]
            }
            1 => {
                let i = rng.range(0, n as i32 - 1) as usize;
                let j = (i + 1) % n;
                a[i] = 1;
                a[j] = rng.range(-3, 3);
            }
            _ => {
                for coeff in &mut a {
                    *coeff = rng.range(-4, 4);
                }
            }
        }
        let rhs = rng.range(-4, 6);
        if rng.range(0, 3) == 0 {
            let k = rng.range(2, 3);
            rows.push((
                a.iter().map(|x| x * k).collect(),
                op,
                rhs * k + rng.range(-1, 1),
            ));
        }
        rows.push((a, op, rhs));
    }
    // A forcing row: the sum of the lower-bound activity.
    if rng.range(0, 3) == 0 {
        let min_act: i32 = lo.iter().sum();
        rows.push((vec![1; n], ComparisonOp::Le, min_act));
    }
    Case { lo, hi, c, rows }
}

fn feasible(case: &Case, x: &[i32]) -> bool {
    case.rows.iter().all(|(a, op, rhs)| {
        let lhs: i32 = a.iter().zip(x).map(|(a, x)| a * x).sum();
        match op {
            ComparisonOp::Le => lhs <= *rhs,
            ComparisonOp::Ge => lhs >= *rhs,
            ComparisonOp::Eq => lhs == *rhs,
        }
    })
}

fn brute_force(case: &Case, dir: OptimizationDirection) -> Option<i32> {
    let n = case.lo.len();
    let mut x = case.lo.clone();
    let mut best: Option<i32> = None;
    loop {
        if feasible(case, &x) {
            let obj: i32 = case.c.iter().zip(&x).map(|(c, x)| c * x).sum();
            best = Some(match (best, dir) {
                (None, _) => obj,
                (Some(b), OptimizationDirection::Minimize) => b.min(obj),
                (Some(b), OptimizationDirection::Maximize) => b.max(obj),
            });
        }
        let mut i = 0;
        while i < n && x[i] == case.hi[i] {
            x[i] = case.lo[i];
            i += 1;
        }
        if i == n {
            return best;
        }
        x[i] += 1;
    }
}

fn build(case: &Case, dir: OptimizationDirection) -> (Problem, Vec<Variable>) {
    let mut problem = Problem::new(dir);
    let vars: Vec<Variable> = (0..case.lo.len())
        .map(|v| problem.add_integer_var(case.c[v] as f64, (case.lo[v], case.hi[v])))
        .collect();
    for (a, op, rhs) in &case.rows {
        let terms: Vec<(Variable, f64)> = vars
            .iter()
            .zip(a)
            .filter(|(_, &a)| a != 0)
            .map(|(&v, &a)| (v, a as f64))
            .collect();
        problem.add_constraint(terms, *op, *rhs as f64);
    }
    (problem, vars)
}

#[test]
fn presolve_keeps_the_optimum_on_integer_models() {
    for seed in 0..600 {
        let case = case(seed);
        for dir in [
            OptimizationDirection::Minimize,
            OptimizationDirection::Maximize,
        ] {
            let expected = brute_force(&case, dir);
            let (problem, vars) = build(&case, dir);
            match (problem.solve_with(options(true)), expected) {
                (Ok(outcome), Some(best)) => {
                    let solution = outcome.into_solution().unwrap();
                    assert!(
                        (solution.objective() - best as f64).abs() < 1e-6,
                        "seed {seed} {dir:?}: presolve {} vs brute force {best}\n{case:?}",
                        solution.objective()
                    );
                    let x: Vec<i32> = vars.iter().map(|&v| solution.var_value(v) as i32).collect();
                    assert!(
                        feasible(&case, &x),
                        "seed {seed}: infeasible point {x:?}\n{case:?}"
                    );
                    let obj: i32 = case.c.iter().zip(&x).map(|(c, x)| c * x).sum();
                    assert_eq!(obj, best, "seed {seed}: values disagree with the objective");
                }
                (Err(Error::Infeasible), None) => {}
                (got, expected) => {
                    panic!("seed {seed} {dir:?}: got {got:?}, expected {expected:?}\n{case:?}")
                }
            }
        }
    }
}

/// Mixed models with continuous variables tied to integers through equality
/// rows, so the doubleton and implied-free substitutions fire.
#[test]
fn presolve_matches_plain_solve_on_mixed_models() {
    for seed in 0..400 {
        let mut rng = Lcg(seed ^ 0x2545_f491_4f6c_dd1d);
        let dir = if seed % 2 == 0 {
            OptimizationDirection::Minimize
        } else {
            OptimizationDirection::Maximize
        };
        let mut problem = Problem::new(dir);
        let ints: Vec<Variable> = (0..rng.range(2, 4))
            .map(|_| problem.add_integer_var(rng.range(-4, 4) as f64, (0, rng.range(1, 4))))
            .collect();
        let mut reals = Vec::new();
        let mut rows: Vec<Row> = Vec::new();
        for _ in 0..rng.range(1, 4) {
            let lo = rng.range(-6, 0) as f64;
            let hi = if rng.range(0, 2) == 0 {
                f64::INFINITY
            } else {
                rng.range(1, 8) as f64
            };
            let y = problem.add_var(rng.range(-3, 3) as f64 * 0.5, (lo, hi));
            reals.push((y, lo, hi));
            // y defined by the integers: a doubleton or a longer equality.
            let mut terms = vec![(y, [1.0, -2.0, 0.5][rng.range(0, 2) as usize])];
            for &x in ints.iter().take(rng.range(1, ints.len() as i32) as usize) {
                terms.push((x, rng.range(-3, 3) as f64 + 0.25));
            }
            rows.push((terms, ComparisonOp::Eq, rng.range(-2, 2) as f64));
        }
        for _ in 0..rng.range(1, 3) {
            let mut terms: Vec<(Variable, f64)> =
                ints.iter().map(|&x| (x, rng.range(-3, 3) as f64)).collect();
            let &(y, _, _) = &reals[rng.range(0, reals.len() as i32 - 1) as usize];
            terms.push((y, rng.range(-2, 2) as f64));
            terms.retain(|t| t.1 != 0.0);
            if terms.is_empty() {
                continue;
            }
            let op = if rng.range(0, 1) == 0 {
                ComparisonOp::Le
            } else {
                ComparisonOp::Ge
            };
            rows.push((terms, op, rng.range(-3, 5) as f64));
        }
        for (terms, op, rhs) in &rows {
            problem.add_constraint(terms.clone(), *op, *rhs);
        }

        let plain = problem.solve_with(options(false));
        let pre = problem.solve_with(options(true));
        match (&plain, &pre) {
            (Ok(a), Ok(b)) => {
                let (a, b) = (a.solution().unwrap(), b.solution().unwrap());
                assert!(
                    (a.objective() - b.objective()).abs() < 1e-6,
                    "seed {seed}: plain {} vs presolve {}",
                    a.objective(),
                    b.objective()
                );
                for (terms, op, rhs) in &rows {
                    let lhs: f64 = terms.iter().map(|&(v, c)| c * b.var_value(v)).sum();
                    let ok = match op {
                        ComparisonOp::Le => lhs <= rhs + 1e-6,
                        ComparisonOp::Ge => lhs >= rhs - 1e-6,
                        ComparisonOp::Eq => (lhs - rhs).abs() <= 1e-6,
                    };
                    assert!(ok, "seed {seed}: presolved point violates a row");
                }
                for &(y, lo, hi) in &reals {
                    let v = b.var_value(y);
                    assert!(
                        v >= lo - 1e-6 && v <= hi + 1e-6,
                        "seed {seed}: bound violated"
                    );
                }
            }
            (Err(Error::Infeasible), Err(Error::Infeasible))
            | (Err(Error::Unbounded), Err(Error::Unbounded)) => {}
            _ => panic!("seed {seed}: plain {plain:?} vs presolve {pre:?}"),
        }
    }
}

#[test]
fn doubleton_equation_moves_the_bounds_to_the_other_variable() {
    // x integer in [0, 10], y in [2, 5], x = 2y: y goes, x gets [4, 10].
    let mut problem = Problem::new(OptimizationDirection::Minimize);
    let x = problem.add_integer_var(1.0, (0, 10));
    let y = problem.add_var(0.0, (2.0, 5.0));
    problem.add_constraint([(x, 1.0), (y, -2.0)], ComparisonOp::Eq, 0.0);
    let solution = problem
        .solve_with(options(true))
        .unwrap()
        .into_solution()
        .unwrap();
    assert_eq!(solution.var_value(x), 4.0);
    assert!((solution.var_value(y) - 2.0).abs() < 1e-9);
    assert!((solution.objective() - 4.0).abs() < 1e-9);
}

#[test]
fn implied_free_column_is_substituted_with_its_cost() {
    // y = x1 + x2 + x3 over binaries, y in [-1, 10] is implied by the row.
    let mut problem = Problem::new(OptimizationDirection::Maximize);
    let xs: Vec<Variable> = (0..3)
        .map(|i| problem.add_binary_var(-(i as f64)))
        .collect();
    let y = problem.add_var(2.0, (-1.0, 10.0));
    let mut row: Vec<(Variable, f64)> = xs.iter().map(|&x| (x, 1.0)).collect();
    row.push((y, -1.0));
    problem.add_constraint(row, ComparisonOp::Eq, 0.0);
    problem.add_constraint([(xs[1], 1.0), (xs[2], 1.0)], ComparisonOp::Le, 1.0);
    let solution = problem
        .solve_with(options(true))
        .unwrap()
        .into_solution()
        .unwrap();
    // x0 = x1 = 1: 2·2 - 1 = 3.
    assert!((solution.objective() - 3.0).abs() < 1e-9);
    assert!((solution.var_value(y) - 2.0).abs() < 1e-9);
}

#[test]
fn singleton_rows_round_integer_bounds() {
    let mut problem = Problem::new(OptimizationDirection::Maximize);
    let x = problem.add_integer_var(1.0, (0, 100));
    problem.add_constraint([(x, 2.0)], ComparisonOp::Le, 7.0);
    let solution = problem
        .solve_with(options(true))
        .unwrap()
        .into_solution()
        .unwrap();
    assert_eq!(solution.var_value(x), 3.0);
}

#[test]
fn forcing_row_fixes_its_variables() {
    let mut problem = Problem::new(OptimizationDirection::Maximize);
    let xs: Vec<Variable> = (0..3).map(|_| problem.add_binary_var(1.0)).collect();
    let z = problem.add_integer_var(1.0, (0, 5));
    problem.add_constraint(xs.iter().map(|&x| (x, 1.0)), ComparisonOp::Le, 0.0);
    problem.add_constraint([(xs[0], 1.0), (z, 1.0)], ComparisonOp::Le, 4.0);
    let solution = problem
        .solve_with(options(true))
        .unwrap()
        .into_solution()
        .unwrap();
    assert!(xs.iter().all(|&x| solution.var_value(x) == 0.0));
    assert_eq!(solution.var_value(z), 4.0);
}

#[test]
fn parallel_rows_keep_the_tighter_side() {
    let mut problem = Problem::new(OptimizationDirection::Maximize);
    let x = problem.add_integer_var(1.0, (0, 10));
    let y = problem.add_integer_var(1.0, (0, 10));
    problem.add_constraint([(x, 1.0), (y, 1.0)], ComparisonOp::Le, 3.0);
    problem.add_constraint([(x, 2.0), (y, 2.0)], ComparisonOp::Le, 4.0);
    problem.add_constraint([(x, -1.0), (y, -1.0)], ComparisonOp::Le, -1.0);
    let solution = problem
        .solve_with(options(true))
        .unwrap()
        .into_solution()
        .unwrap();
    assert!((solution.objective() - 2.0).abs() < 1e-9);
}

#[test]
fn presolve_reports_infeasibility() {
    let mut problem = Problem::new(OptimizationDirection::Minimize);
    let x = problem.add_binary_var(1.0);
    let y = problem.add_binary_var(1.0);
    problem.add_constraint([(x, 1.0), (y, 1.0)], ComparisonOp::Ge, 3.0);
    assert!(matches!(
        problem.solve_with(options(true)),
        Err(Error::Infeasible)
    ));
}

#[test]
fn a_fully_reduced_problem_still_reports_values_and_objective() {
    let mut problem = Problem::new(OptimizationDirection::Maximize);
    let x = problem.add_integer_var(3.0, (2, 2));
    let y = problem.add_integer_var(-1.0, (0, 9));
    let z = problem.add_var(0.5, (0.0, 10.0));
    problem.add_constraint([(y, 1.0)], ComparisonOp::Ge, 4.0);
    problem.add_constraint([(y, 1.0)], ComparisonOp::Le, 4.0);
    problem.add_constraint([(x, 1.0), (z, -1.0)], ComparisonOp::Eq, -1.0);
    let solution = problem
        .solve_with(options(true))
        .unwrap()
        .into_solution()
        .unwrap();
    assert_eq!(solution.var_value(x), 2.0);
    assert_eq!(solution.var_value(y), 4.0);
    assert!((solution.var_value(z) - 3.0).abs() < 1e-9);
    assert!((solution.objective() - (6.0 - 4.0 + 1.5)).abs() < 1e-9);
    assert_eq!(solution.iter().count(), 3);
}

#[test]
fn warm_start_and_edits_use_original_variables() {
    let mut problem = Problem::new(OptimizationDirection::Maximize);
    let a = problem.add_integer_var(2.0, (0, 5));
    let b = problem.add_integer_var(3.0, (0, 5));
    let fixed = problem.add_integer_var(1.0, (1, 1));
    let slack = problem.add_var(0.0, (0.0, f64::INFINITY));
    problem.add_constraint([(a, 1.0), (b, 2.0), (slack, 1.0)], ComparisonOp::Eq, 8.0);
    problem.add_constraint([(a, 1.0), (fixed, 1.0)], ComparisonOp::Le, 5.0);

    let mut opts = options(true);
    opts.warm_start = Some(vec![(a, 4.0), (b, 2.0), (fixed, 1.0), (slack, 0.0)]);
    let solution = problem.solve_with(opts).unwrap().into_solution().unwrap();
    let plain = problem
        .solve_with(options(false))
        .unwrap()
        .into_solution()
        .unwrap();
    assert!((solution.objective() - plain.objective()).abs() < 1e-9);

    let edited = solution.fix_var(b, 1.0).unwrap().into_solution().unwrap();
    assert_eq!(edited.var_value(b), 1.0);
    assert_eq!(edited.var_value(fixed), 1.0);
    assert!((edited.objective() - (2.0 * 4.0 + 3.0 + 1.0)).abs() < 1e-9);

    let edited = edited
        .add_constraint([(a, 1.0)], ComparisonOp::Le, 2.0)
        .unwrap()
        .into_solution()
        .unwrap();
    assert_eq!(edited.var_value(a), 2.0);
    assert!((edited.var_value(slack) - 4.0).abs() < 1e-9);
}
