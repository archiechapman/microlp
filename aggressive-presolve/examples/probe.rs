//! Prints the integer vars of an answer that sit off their integers, and
//! whether rounding them keeps the point feasible.
use aggressive_presolve::{check, read_file, solve, Options};
fn main() {
    let path = std::env::args().nth(1).unwrap();
    let m = read_file(&path).unwrap();
    let mut o = Options::default();
    o.microlp_presolve = std::env::args().nth(2).as_deref() != Some("off");
    let a = solve(&m, &o).unwrap();
    let x = a.values.unwrap();
    for (j, v) in m.vars.iter().enumerate() {
        if v.is_integer() && x[j] != x[j].round() {
            println!("{} = {:e} (cost {})", v.name, x[j], v.cost);
        }
    }
    let r: Vec<f64> = m.vars.iter().zip(&x).map(|(v, &x)| if v.is_integer() { x.round() } else { x }).collect();
    let rep = check(&m, &r, &Default::default());
    println!("objective {:?}, rounded {} check {:?}", a.objective, m.objective(&r), rep.violations);
}
