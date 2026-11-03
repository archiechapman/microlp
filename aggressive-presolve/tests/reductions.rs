//! One test per reduction, and postsolve round trips.

use aggressive_presolve::{
    check, presolve, CheckOptions, Direction, Model, PresolveOptions, Presolved,
};

fn reduce(m: &Model) -> Presolved {
    presolve(m, &PresolveOptions::default()).expect("feasible")
}

fn size(p: &Presolved) -> (usize, usize) {
    (p.reduced.rows.len(), p.reduced.vars.len())
}

#[test]
fn singleton_row_becomes_a_rounded_bound() {
    let mut m = Model::new(Direction::Maximize);
    let x = m.add_int_var(1.0, 0.0, 100.0);
    let y = m.add_int_var(1.0, 0.0, 100.0);
    m.add_le(&[(x, 2.0)], 7.0);
    m.add_le(&[(x, 1.0), (y, 3.0)], 50.0);
    let p = reduce(&m);
    assert_eq!(size(&p), (1, 2));
    assert_eq!(p.reduced.vars[0].hi, 3.0);
    // y <= 50/3 from the remaining row, rounded down.
    assert_eq!(p.reduced.vars[1].hi, 16.0);
    assert_eq!(p.stats.singleton_rows, 1);
}

#[test]
fn singleton_row_bounds_a_continuous_var() {
    let mut m = Model::new(Direction::Maximize);
    let x = m.add_var(1.0, 0.0, 100.0);
    let y = m.add_var(1.0, 0.0, 100.0);
    m.add_le(&[(x, 4.0)], 7.0);
    m.add_ge(&[(x, -1.0)], -10.0);
    m.add_le(&[(x, 1.0), (y, 3.0)], 50.0);
    let p = reduce(&m);
    assert_eq!(p.reduced.rows.len(), 1);
    assert_eq!((p.reduced.vars[0].lo, p.reduced.vars[0].hi), (0.0, 1.75));
}

#[test]
fn fixed_and_zero_cost_empty_columns_go() {
    let mut m = Model::new(Direction::Minimize);
    let x = m.add_var(2.0, 3.0, 3.0);
    let y = m.add_int_var(1.0, 0.0, 9.0);
    let _unused = m.add_var(0.0, -4.0, -1.0);
    let z = m.add_int_var(1.0, 0.0, 9.0);
    m.add_ge(&[(x, 1.0), (y, 1.0), (z, 1.0)], 5.0);
    let p = reduce(&m);
    assert_eq!(size(&p), (1, 2));
    assert_eq!(p.postsolve.offset, 6.0);
    assert_eq!(p.reduced.obj_offset, 6.0);
    assert_eq!(p.reduced.rows[0].lo, 2.0);
    assert_eq!(p.postsolve.values(&[1.0, 1.0]), vec![3.0, 1.0, -1.0, 1.0]);
}

#[test]
fn empty_integer_columns_are_kept() {
    let mut m = Model::new(Direction::Minimize);
    let _x = m.add_int_var(0.0, 0.0, 5.0);
    let y = m.add_var(0.0, 0.0, 5.0);
    m.add_ge(&[(y, 1.0), (_x, 0.0)], 1.0);
    let p = reduce(&m);
    assert_eq!(p.stats.empty_cols, 1);
    assert_eq!(p.reduced.vars.len(), 1);
    assert!(p.reduced.vars[0].is_integer());
}

#[test]
fn redundant_and_forcing_rows_go() {
    let mut m = Model::new(Direction::Minimize);
    let a = m.add_int_var(1.0, 0.0, 1.0);
    let b = m.add_int_var(1.0, 0.0, 1.0);
    let c = m.add_int_var(1.0, 0.0, 1.0);
    let d = m.add_int_var(-1.0, 0.0, 1.0);
    m.add_le(&[(a, 1.0), (b, 1.0)], 3.0); // redundant
    m.add_ge(&[(a, 1.0), (c, -1.0)], 1.0); // forcing: a = 1, c = 0
    m.add_le(&[(b, 1.0), (d, 1.0), (c, 1.0)], 1.0);
    let p = reduce(&m);
    assert_eq!(p.stats.redundant_rows, 1);
    assert_eq!(p.stats.forcing_rows, 1);
    assert_eq!(p.reduced.vars.len(), 2); // b, d
    assert_eq!(p.postsolve.values(&[0.0, 1.0]), vec![1.0, 0.0, 0.0, 1.0]);
}

#[test]
fn a_tight_row_is_not_dropped_as_redundant() {
    // x + y <= 2 with binaries is met at every corner, but with no room for
    // the bound tolerance the engine may use, so it stays.
    let mut m = Model::new(Direction::Maximize);
    let x = m.add_int_var(1.0, 0.0, 1.0);
    let y = m.add_int_var(1.0, 0.0, 1.0);
    let z = m.add_var(1.0, 0.0, 1.0);
    m.add_le(&[(x, 1.0), (y, 1.0)], 2.0);
    m.add_le(&[(x, 1.0), (z, 1.0)], 1.5);
    let p = reduce(&m);
    assert_eq!(p.stats.redundant_rows, 0);
}

#[test]
fn forcing_row_fixes_continuous_vars() {
    let mut m = Model::new(Direction::Minimize);
    let x = m.add_var(1.0, 0.0, 2.0);
    let y = m.add_var(1.0, -1.0, 4.0);
    let z = m.add_var(1.0, 0.0, 10.0);
    m.add_le(&[(x, 1.0), (y, -1.0)], -4.0); // min activity 0 - 4 = -4: forcing
    m.add_ge(&[(x, 1.0), (y, 1.0), (z, 1.0)], 6.0);
    let p = reduce(&m);
    assert_eq!(p.stats.forcing_rows, 1);
    let v = p.postsolve.values(&vec![0.0; p.reduced.vars.len()]);
    assert_eq!((v[0], v[1]), (0.0, 4.0));
}

#[test]
fn forcing_row_does_not_fix_an_integer_with_a_tiny_coefficient() {
    let mut m = Model::new(Direction::Minimize);
    let x = m.add_var(1.0, 0.0, 2.0);
    let k = m.add_int_var(1.0, 0.0, 1000.0);
    // Forcing within round-off; k's coefficient is below the room.
    m.add_le(&[(x, 1.0), (k, 1e-12)], 0.0);
    m.add_ge(&[(x, 1.0), (k, 1.0)], 0.0);
    let p = reduce(&m);
    assert_eq!(p.stats.forcing_rows, 0);
}

#[test]
fn doubleton_equation_eliminates_the_continuous_side() {
    let mut m = Model::new(Direction::Minimize);
    let x = m.add_int_var(1.0, 0.0, 10.0);
    let y = m.add_var(3.0, 2.0, 5.0);
    let z = m.add_int_var(1.0, 0.0, 10.0);
    m.add_eq(&[(x, 1.0), (y, -2.0)], 1.0);
    m.add_le(&[(y, 1.0), (z, 1.0)], 6.0);
    let p = reduce(&m);
    assert_eq!(size(&p), (1, 2));
    assert_eq!(p.stats.doubletons, 1);
    // y = (x - 1)/2 in [2, 5]  =>  x in [5, 11] ∩ [0, 10].
    assert_eq!((p.reduced.vars[0].lo, p.reduced.vars[0].hi), (5.0, 10.0));
    assert_eq!(p.reduced.vars[0].cost, 1.0 + 1.5);
    assert_eq!(p.postsolve.offset, -1.5);
    assert_eq!(p.postsolve.values(&[7.0, 2.0]), vec![7.0, 3.0, 2.0]);
}

#[test]
fn integer_doubleton_needs_an_integral_ratio() {
    let build = |a: f64, b: f64| {
        let mut m = Model::new(Direction::Minimize);
        let x = m.add_int_var(1.0, 0.0, 10.0);
        let y = m.add_int_var(1.0, 0.0, 10.0);
        let z = m.add_int_var(1.0, 0.0, 10.0);
        m.add_eq(&[(x, a), (y, b)], 1.0);
        m.add_le(&[(y, 1.0), (z, 1.0)], 6.0);
        m
    };
    // rhs/pivot is 1/2 or -1/3: eliminating either side could admit
    // fractional points.
    assert_eq!(reduce(&build(2.0, -3.0)).reduced.vars.len(), 3);
    assert_eq!(reduce(&build(3.0, -1.0)).reduced.vars.len(), 2);
}

#[test]
fn implied_free_substitution_respects_its_bounds() {
    // w = x1 + x2 + x3 appears in one more row; w's bounds are implied.
    let build = |w_hi: f64| {
        let mut m = Model::new(Direction::Minimize);
        let xs: Vec<_> = (0..3).map(|_| m.add_int_var(1.0, 0.0, 1.0)).collect();
        let w = m.add_var(1.0, 0.0, w_hi);
        let v = m.add_int_var(1.0, 0.0, 5.0);
        let mut row: Vec<_> = xs.iter().map(|&x| (x, 1.0)).collect();
        row.push((w, -1.0));
        m.add_eq(&row, 0.0);
        m.add_ge(&[(w, 1.0), (v, 1.0)], 2.0);
        m
    };
    let p = reduce(&build(3.0));
    assert_eq!(size(&p), (1, 4));
    assert_eq!(p.stats.substitutions, 1);
    let costs: Vec<f64> = p.reduced.vars.iter().map(|v| v.cost).collect();
    assert_eq!(costs, vec![2.0, 2.0, 2.0, 1.0]);
    assert_eq!(p.postsolve.values(&[1.0, 0.0, 1.0, 0.0])[3], 2.0);
    // Not implied: w in [0, 1] is tighter than the row's [0, 3].
    assert_eq!(reduce(&build(1.0)).reduced.vars.len(), 5);
}

#[test]
fn substitution_respects_the_fill_in_cap() {
    // A free w in one long equality and in many rows: substituting it would
    // add far more than the cap.
    let mut m = Model::new(Direction::Minimize);
    let xs: Vec<_> = (0..8).map(|_| m.add_var(1.0, 0.0, 1.0)).collect();
    let w = m.add_var(0.0, f64::NEG_INFINITY, f64::INFINITY);
    let mut row: Vec<_> = xs.iter().map(|&x| (x, 1.0)).collect();
    row.push((w, -1.0));
    m.add_eq(&row, 0.0);
    for k in 0..4 {
        let y = m.add_var(1.0, 0.0, 1.0);
        m.add_ge(&[(w, 1.0 + k as f64), (y, 1.0)], 1.0);
    }
    let p = reduce(&m);
    assert_eq!(p.stats.substitutions, 0);
    assert!(p.stats.refused_fill > 0);
    let loose = PresolveOptions {
        max_fill: 100,
        ..Default::default()
    };
    assert_eq!(presolve(&m, &loose).unwrap().stats.substitutions, 1);
}

#[test]
fn substitution_respects_the_ratio_growth_cap() {
    // x = 100 y + 100 z (ratio 100) into x + 0.01 w <= 5 (ratio 100) would
    // leave 100 y + 100 z + 0.01 w: ratio 1e4 > 4 · 100.
    let mut m = Model::new(Direction::Minimize);
    let x = m.add_var(1.0, f64::NEG_INFINITY, f64::INFINITY);
    let y = m.add_var(1.0, 0.0, 1.0);
    let z = m.add_var(1.0, 0.0, 1.0);
    let w = m.add_var(-1.0, 0.0, 100.0);
    m.add_eq(&[(x, 1.0), (y, -100.0), (z, -100.0)], 0.0);
    m.add_le(&[(x, 1.0), (w, 0.01)], 5.0);
    m.add_ge(&[(y, 1.0), (z, 1.0), (w, 1.0)], 0.5);
    let p = reduce(&m);
    assert_eq!(p.stats.substitutions, 0);
    assert!(p.stats.refused_ratio > 0);
    let loose = PresolveOptions {
        max_ratio_growth: f64::INFINITY,
        ..Default::default()
    };
    let p = presolve(&m, &loose).unwrap();
    assert_eq!(p.stats.substitutions, 1);
    assert!(p.stats.worst_ratio_after > 4.0 * p.stats.worst_ratio_before);
}

#[test]
fn substitution_pivot_must_not_be_tiny() {
    // x appears with 1e-3 beside 1: below 1% of the row's largest.
    let mut m = Model::new(Direction::Minimize);
    let x = m.add_var(1.0, f64::NEG_INFINITY, f64::INFINITY);
    let y = m.add_var(1.0, 0.0, 1.0);
    let z = m.add_var(1.0, 0.0, 1.0);
    m.add_eq(&[(x, 1e-3), (y, 1.0), (z, -1.0)], 0.0);
    m.add_le(&[(x, 1.0), (y, 1.0)], 5.0);
    let p = reduce(&m);
    assert_eq!(p.stats.substitutions, 0);
}

#[test]
fn protected_rows_and_frozen_cols_are_left_alone() {
    let mut m = Model::new(Direction::Minimize);
    let x = m.add_int_var(1.0, 0.0, 10.0);
    let y = m.add_var(3.0, 2.0, 5.0);
    let z = m.add_int_var(1.0, 0.0, 10.0);
    m.add_eq(&[(x, 1.0), (y, -2.0)], 1.0);
    m.add_le(&[(y, 1.0), (z, 1.0)], 6.0);
    m.add_le(&[(z, 2.0)], 9.0);
    let mut o = PresolveOptions::default();
    o.protected_rows.insert(2);
    let p = presolve(&m, &o).unwrap();
    assert_eq!(p.stats.singleton_rows, 0);
    assert_eq!(
        p.stats.doubletons, 1,
        "z is in a protected row, but y is not"
    );
    assert_eq!(p.reduced.rows.len(), 2);
    let mut o = PresolveOptions::default();
    o.frozen_cols.insert(y);
    o.frozen_cols.insert(x);
    let p = presolve(&m, &o).unwrap();
    assert_eq!(p.stats.doubletons, 0);
}

#[test]
fn contradictions_are_reported() {
    let mut m = Model::new(Direction::Minimize);
    let x = m.add_int_var(1.0, 0.0, 3.0);
    m.add_eq(&[(x, 2.0)], 3.0); // x = 1.5
    assert!(presolve(&m, &PresolveOptions::default()).is_err());
}

/// Chains of substitutions reconstruct points that meet the original rows.
#[test]
fn postsolve_round_trips_through_chained_substitutions() {
    let mut m = Model::new(Direction::Minimize);
    let a = m.add_var(1.0, 0.0, 10.0);
    let b = m.add_var(2.0, f64::NEG_INFINITY, f64::INFINITY);
    let c = m.add_var(-1.0, f64::NEG_INFINITY, f64::INFINITY);
    let d = m.add_var(1.0, 0.0, 10.0);
    let e = m.add_int_var(1.0, 0.0, 4.0);
    m.add_eq(&[(b, 1.0), (a, -0.5), (d, -1.5)], 2.0); // b free: substituted
    m.add_eq(&[(c, 2.0), (b, 1.0), (e, -1.0)], 1.0); // c free: substituted
    m.add_le(&[(a, 1.0), (b, 1.0), (c, 1.0), (d, 1.0)], 20.0);
    m.add_ge(&[(c, 1.0), (e, 1.0), (d, 1.0)], -3.0);
    let p = reduce(&m);
    assert!(p.stats.substitutions + p.stats.doubletons >= 2);
    // Any point of the reduced model maps to a point of the original model
    // with the same objective.
    for pt in [[0.0, 0.0, 0.0], [3.0, 7.0, 2.0], [10.0, 1.0, 4.0]] {
        let red: Vec<f64> = pt[..p.reduced.vars.len()].to_vec();
        let red_ok = check(&p.reduced, &red, &CheckOptions::default()).ok();
        let x = p.postsolve.values(&red);
        let orig_ok = check(&m, &x, &CheckOptions::default()).ok();
        assert_eq!(red_ok, orig_ok, "point {pt:?} -> {x:?}");
        let (o1, o2) = (p.reduced.objective(&red), m.objective(&x));
        assert!((o1 - o2).abs() <= 1e-12 * o1.abs().max(1.0), "{o1} vs {o2}");
    }
}
