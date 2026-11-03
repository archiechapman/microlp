//! The long-row round-off case: `max y` s.t. `y + Σ_{i<5000} c·x_i <= b`,
//! every `x_i` fixed at 1, all `c` equal (`1e6·U(1,2)`), `y` integer in
//! `[0, 100]`, `b = exact(Σc) + 7`. Folding 5,000 fixed values into the side
//! in plain `f64` loses the 7 to round-off; the answer must be 7.

use aggressive_presolve::{solve, sum::two_prod, Direction, Model, Options, Path, Status};

#[test]
fn long_row_of_fixed_terms_keeps_its_slack() {
    let n = 5000;
    // Deterministic U(1, 2) draws with a full mantissa; take the first whose
    // plain f64 fold of the side falls below 7 - 1e-4, so the case bites.
    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    let (c, b, plain) = loop {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let c = 1e6 * (1.0 + (state >> 11) as f64 / (1u64 << 53) as f64);
        // exact(Σc) + 7 = n·c + 7, the product carried exactly, rounded once.
        let (p, e) = two_prod(n as f64, c);
        let b = p + (e + 7.0);
        let mut plain = b;
        for _ in 0..n {
            plain -= c;
        }
        if plain < 7.0 - 1e-4 {
            break (c, b, plain);
        }
    };
    println!("c = {c}, plain fold leaves {plain}, exact 7");
    let mut m = Model::new(Direction::Maximize);
    let y = m.add_int_var(1.0, 0.0, 100.0);
    let xs: Vec<usize> = (0..n).map(|_| m.add_var(0.0, 1.0, 1.0)).collect();
    let mut terms = vec![(y, 1.0)];
    terms.extend(xs.iter().map(|&x| (x, c)));
    m.add_le(&terms, b);

    let ans = solve(&m, &Options::default()).unwrap();
    assert_eq!(ans.status, Status::Optimal);
    assert_eq!(ans.path, Path::Presolved);
    assert_eq!(ans.values.as_ref().unwrap()[y], 7.0);
    assert_eq!(ans.objective, Some(7.0));
}
