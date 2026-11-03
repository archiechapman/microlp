//! The paths of `solve`: presolve's own infeasibility and the reduced model's
//! verdicts are confirmed on the model as written.

use aggressive_presolve::{solve, Direction, Model, Options, Path, Status};

#[test]
fn presolve_infeasibility_is_confirmed_on_the_original() {
    let mut m = Model::new(Direction::Minimize);
    let x = m.add_int_var(1.0, 0.0, 3.0);
    let y = m.add_var(1.0, 0.0, 3.0);
    m.add_eq(&[(x, 2.0)], 3.0);
    m.add_le(&[(x, 1.0), (y, 1.0)], 4.0);
    let a = solve(&m, &Options::default()).unwrap();
    assert_eq!(a.status, Status::Infeasible);
    assert!(matches!(a.path, Path::Original(ref r) if r.starts_with("presolve infeasible")));
    assert_eq!(a.attempts.len(), 1);
    assert_eq!(a.attempts[0].label, "original");
}

#[test]
fn an_unbounded_reduced_model_is_confirmed_on_the_original() {
    let mut m = Model::new(Direction::Maximize);
    let x = m.add_var(1.0, 0.0, f64::INFINITY);
    let y = m.add_var(0.0, f64::NEG_INFINITY, f64::INFINITY);
    let z = m.add_int_var(1.0, 0.0, 5.0);
    m.add_eq(&[(y, 1.0), (x, -1.0), (z, 1.0)], 0.0);
    m.add_ge(&[(x, 1.0), (z, 1.0)], 1.0);
    let a = solve(&m, &Options::default()).unwrap();
    assert_eq!(a.status, Status::Unbounded);
    assert!(matches!(a.path, Path::Original(_)), "{:?}", a.path);
}

#[test]
fn a_model_presolve_empties_is_answered_without_microlp() {
    let mut m = Model::new(Direction::Minimize);
    let x = m.add_var(2.0, 1.0, 1.0);
    let y = m.add_int_var(3.0, 0.0, 9.0);
    m.add_ge(&[(x, 1.0), (y, 1.0)], 4.0);
    m.add_le(&[(y, 1.0)], 3.0);
    let a = solve(&m, &Options::default()).unwrap();
    assert_eq!(a.status, Status::Optimal);
    assert_eq!(a.objective, Some(11.0));
    assert_eq!(a.values, Some(vec![1.0, 3.0]));
    assert_eq!(a.path, Path::Presolved);
}

#[test]
fn presolve_off_solves_the_model_as_written() {
    let mut m = Model::new(Direction::Minimize);
    let x = m.add_var(1.0, 0.0, 10.0);
    m.add_ge(&[(x, 1.0)], 2.5);
    let mut o = Options::default();
    o.presolve.enabled = false;
    let a = solve(&m, &o).unwrap();
    assert_eq!(a.path, Path::NoPresolve);
    assert_eq!(a.objective, Some(2.5));
}
