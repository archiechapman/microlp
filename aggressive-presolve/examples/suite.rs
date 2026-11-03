//! Runs microlp's correctness-suite files three ways and prints a markdown
//! table: microlp alone, aggressive presolve + microlp, and aggressive presolve
//! with microlp's own presolve off.
//!
//! `cargo run --release --example suite -- [--time-limit S] [--parallel N] [--filter S] [--out FILE]`
//!
//! Known values are the suite's (`tests/suite/cases/{netlib,miplib,milpbench}.rs`),
//! and also the Rust-built families exported by `examples/families.rs` when
//! their `.mps` files are present under `target/families/`.

use aggressive_presolve::{read_file, solve, Answer, Model, Options, Status};
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;

struct Known {
    name: &'static str,
    /// The optimum, or `None` for an expected `Infeasible` (NaN in an
    /// instance's value stands for an expected `Unbounded`).
    value: Option<f64>,
    abs: f64,
    rel: f64,
}

const NETLIB_TOL: (f64, f64) = (1e-6, 1e-6);
const EXACT: (f64, f64) = (1e-4, 1e-9);

fn known() -> Vec<Known> {
    let mk = |name, v: f64, (abs, rel): (f64, f64)| Known {
        name,
        value: Some(v),
        abs,
        rel,
    };
    vec![
        mk("afiro", -4.6475314286e2, NETLIB_TOL),
        mk("sc50a", -6.4575077059e1, NETLIB_TOL),
        mk("sc50b", -7.0000000000e1, NETLIB_TOL),
        mk("sc105", -5.2202061212e1, NETLIB_TOL),
        mk("adlittle", 2.2549496316e5, NETLIB_TOL),
        mk("blend", -3.0812149846e1, NETLIB_TOL),
        mk("kb2", -1.7499001299e3, NETLIB_TOL),
        mk("share2b", -4.1573224074e2, NETLIB_TOL),
        mk("stocfor1", -4.1131976219e4, NETLIB_TOL),
        mk("scagr7", -2.3313892548e6, NETLIB_TOL),
        mk("israel", -8.9664482186e5, NETLIB_TOL),
        mk("brandy", 1.5185098965e3, NETLIB_TOL),
        mk("beaconfd", 3.3592485807e4, NETLIB_TOL),
        mk("sctap1", 1.4122500000e3, NETLIB_TOL),
        mk("greenbea", -7.2555248130e7, NETLIB_TOL),
        mk("flugpl", 1201500.0, EXACT),
        mk("p0033", 3089.0, EXACT),
        mk("stein27", 18.0, EXACT),
        mk("rgn", 82.19999924, (0.001, 1e-6)),
        mk("lseu", 1120.0, EXACT),
        mk("p0201", 7615.0, EXACT),
        mk("misc03", 3360.0, EXACT),
        mk("stein45", 30.0, EXACT),
        mk("mod008", 307.0, EXACT),
        mk("enigma", 0.0, EXACT),
        mk("vpm1", 20.0, EXACT),
        mk("gt2", 21166.0, EXACT),
        mk("bell3a", 878430.32, (0.02, 1e-6)),
        mk("CFL_easy_instance_15", 9959.569649350798, (1e-3, 1e-6)),
        mk("CFL_easy_instance_12", 10000.194107917776, (1e-3, 1e-6)),
    ]
}

struct Instance {
    name: String,
    path: PathBuf,
    value: Option<Option<f64>>,
    abs: f64,
    rel: f64,
}

fn collect(root: &Path, filter: &Option<String>) -> Vec<Instance> {
    let mut files = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(d) = stack.pop() {
        let Ok(rd) = std::fs::read_dir(&d) else {
            continue;
        };
        for e in rd.flatten() {
            let p = e.path();
            if p.is_dir() {
                stack.push(p);
            } else {
                let n = p.file_name().unwrap().to_string_lossy().to_string();
                if n.ends_with(".mps")
                    || n.ends_with(".mps.gz")
                    || n.ends_with(".lp")
                    || n.ends_with(".lp.gz")
                {
                    files.push(p);
                }
            }
        }
    }
    files.sort();
    let known = known();
    let mut out = Vec::new();
    for p in files {
        let fname = p.file_name().unwrap().to_string_lossy().to_string();
        let stem = fname.split('.').next().unwrap().to_string();
        if p.components().any(|c| c.as_os_str() == "xhard") {
            continue; // beyond microlp's ceiling, 10-minute budgets
        }
        if let Some(f) = filter {
            if !stem.contains(f.as_str()) {
                continue;
            }
        }
        let k = known.iter().find(|k| k.name == stem);
        out.push(Instance {
            name: stem,
            path: p,
            value: k.map(|k| k.value),
            abs: k.map_or(1e-6, |k| k.abs),
            rel: k.map_or(1e-6, |k| k.rel),
        });
    }
    out
}

/// Families exported by `examples/families.rs`: `<name>.mps` next to
/// `<name>.known` holding the optimum (or `infeasible`).
fn collect_families(dir: &Path, filter: &Option<String>) -> Vec<Instance> {
    let mut out = Vec::new();
    let Ok(rd) = std::fs::read_dir(dir) else {
        return out;
    };
    let mut paths: Vec<PathBuf> = rd
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|e| e == "mps"))
        .collect();
    paths.sort();
    for p in paths {
        let stem = p.file_stem().unwrap().to_string_lossy().to_string();
        if let Some(f) = filter {
            if !stem.contains(f.as_str()) {
                continue;
            }
        }
        let known = std::fs::read_to_string(p.with_extension("known")).ok();
        let value = known.map(|s| match s.trim() {
            "infeasible" => None,
            "unbounded" => Some(f64::NAN),
            s => Some(s.parse::<f64>().expect("known value")),
        });
        out.push(Instance {
            name: format!("family/{stem}"),
            path: p,
            value,
            abs: 1e-6,
            rel: 1e-6,
        });
    }
    out
}

/// Rows, cols and integer cols of an instance, and its three runs.
type Row3 = (usize, usize, usize, [RunResult; 3]);

#[derive(Clone)]
struct RunResult {
    status: String,
    objective: Option<f64>,
    time: f64,
    nodes: u64,
    rows_after: usize,
    cols_after: usize,
    path: String,
    verdict: Verdict,
    check_ok: Option<bool>,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Verdict {
    Pass,
    /// A limit stopped the solve; whatever it returned is consistent.
    Timeout,
    Fail,
    /// No known value.
    Unknown,
}

/// The objective at `x` with every integer var rounded, where that point
/// still meets the model: an objective that misses the known value only by
/// integer vars off their integers within `int_tol` is judged on this.
fn rounded_objective(model: &Model, x: &[f64]) -> Option<f64> {
    let r: Vec<f64> = model
        .vars
        .iter()
        .zip(x)
        .map(|(v, &x)| if v.is_integer() { x.round() } else { x })
        .collect();
    aggressive_presolve::check(model, &r, &Default::default())
        .ok()
        .then(|| model.objective(&r))
}

fn judge(inst: &Instance, model: &Model, ans: &Result<Answer, String>, dir_max: bool) -> Verdict {
    let Ok(ans) = ans else { return Verdict::Fail };
    if ans.check.as_ref().is_some_and(|c| !c.ok()) {
        return Verdict::Fail;
    }
    let Some(expected) = inst.value else {
        return Verdict::Unknown;
    };
    // The suite's rule: |o - v| <= abs + rel·|v|.
    let near = |o: f64, v: f64| (o - v).abs() <= inst.abs + inst.rel * v.abs();
    let rounded = ans
        .values
        .as_ref()
        .and_then(|x| rounded_objective(model, x));
    let close = |o: f64, v: f64| near(o, v) || rounded.is_some_and(|r| near(r, v));
    match (expected, ans.status) {
        (Some(v), Status::Unbounded) if v.is_nan() => Verdict::Pass,
        (Some(v), Status::NoSolution) if v.is_nan() => Verdict::Timeout,
        (Some(v), _) if v.is_nan() => Verdict::Fail,
        (None, Status::Infeasible) => Verdict::Pass,
        (None, Status::NoSolution) => Verdict::Timeout,
        (None, _) => Verdict::Fail,
        (Some(v), Status::Optimal) => {
            if close(ans.objective.unwrap(), v) {
                Verdict::Pass
            } else {
                Verdict::Fail
            }
        }
        (Some(v), Status::Feasible) => {
            // An incumbent must not beat the optimum.
            let o = ans.objective.unwrap();
            let better = if dir_max { o > v } else { o < v };
            if better && !close(o, v) {
                Verdict::Fail
            } else {
                Verdict::Timeout
            }
        }
        (Some(_), Status::NoSolution) => Verdict::Timeout,
        (Some(_), _) => Verdict::Fail,
    }
}

fn run(model: &Model, inst: &Instance, mode: usize, limit: Duration) -> RunResult {
    let mut opts = Options {
        time_limit: Some(limit),
        ..Options::default()
    };
    match mode {
        0 => opts.presolve.enabled = false,
        1 => {}
        _ => opts.microlp_presolve = false,
    }
    let ans = solve(model, &opts).map_err(|e| e.to_string());
    let verdict = judge(
        inst,
        model,
        &ans,
        model.direction == aggressive_presolve::Direction::Maximize,
    );
    match ans {
        Ok(a) => RunResult {
            status: format!("{:?}", a.status),
            objective: a.objective,
            time: a.time.as_secs_f64(),
            nodes: a.nodes,
            rows_after: a
                .presolve_stats
                .as_ref()
                .map_or(model.rows.len(), |s| s.rows_after),
            cols_after: a
                .presolve_stats
                .as_ref()
                .map_or(model.vars.len(), |s| s.cols_after),
            path: match &a.path {
                aggressive_presolve::Path::NoPresolve => "-".into(),
                aggressive_presolve::Path::Presolved => "presolved".into(),
                aggressive_presolve::Path::PresolvedRetry(k) => format!("retry {k}"),
                aggressive_presolve::Path::Original(r) => format!("original: {r}"),
            },
            verdict,
            check_ok: a.check.as_ref().map(|c| c.ok()),
        },
        Err(e) => RunResult {
            status: format!("error: {e}"),
            objective: None,
            time: f64::NAN,
            nodes: 0,
            rows_after: 0,
            cols_after: 0,
            path: "-".into(),
            verdict,
            check_ok: None,
        },
    }
}

fn main() {
    let mut args = std::env::args().skip(1);
    let mut limit = 60.0;
    let mut parallel = 2usize;
    let mut filter = None;
    let mut out_path = None;
    while let Some(a) = args.next() {
        match a.as_str() {
            "--time-limit" => limit = args.next().unwrap().parse().unwrap(),
            "--parallel" => parallel = args.next().unwrap().parse().unwrap(),
            "--filter" => filter = args.next(),
            "--out" => out_path = args.next(),
            o => panic!("unknown argument {o}"),
        }
    }
    let crate_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut insts = collect(&crate_dir.join("../tests/suite/data"), &filter);
    insts.extend(collect_families(
        &crate_dir.join("target/families"),
        &filter,
    ));
    let insts = Arc::new(insts);
    let results: Arc<Mutex<Vec<Option<Row3>>>> = Arc::new(Mutex::new(vec![None; insts.len()]));
    let next = Arc::new(Mutex::new(0usize));
    let limit = Duration::from_secs_f64(limit);
    let mut handles = Vec::new();
    for _ in 0..parallel {
        let (insts, results, next) = (insts.clone(), results.clone(), next.clone());
        handles.push(std::thread::spawn(move || loop {
            let i = {
                let mut n = next.lock().unwrap();
                let i = *n;
                *n += 1;
                i
            };
            if i >= insts.len() {
                break;
            }
            let inst = &insts[i];
            let model = match read_file(&inst.path) {
                Ok(m) => m,
                Err(e) => {
                    eprintln!("{}: read error {e}", inst.name);
                    continue;
                }
            };
            let r = [
                run(&model, inst, 0, limit),
                run(&model, inst, 1, limit),
                run(&model, inst, 2, limit),
            ];
            eprintln!(
                "{:<28} {:>9.3}s {:?} | {:>9.3}s {:?} {} | {:>9.3}s {:?} {}",
                inst.name,
                r[0].time,
                r[0].verdict,
                r[1].time,
                r[1].verdict,
                r[1].path,
                r[2].time,
                r[2].verdict,
                r[2].path
            );
            results.lock().unwrap()[i] =
                Some((model.rows.len(), model.vars.len(), model.num_integer(), r));
        }));
    }
    for h in handles {
        h.join().unwrap();
    }
    let results = results.lock().unwrap();
    let mut md = String::new();
    let labels = [
        "microlp alone",
        "aggressive + microlp",
        "aggressive, microlp presolve off",
    ];
    let mut counts = [[0usize; 4]; 3];
    for (_, _, _, r) in results.iter().flatten() {
        for k in 0..3 {
            let idx = match r[k].verdict {
                Verdict::Pass => 0,
                Verdict::Timeout => 1,
                Verdict::Fail => 2,
                Verdict::Unknown => 3,
            };
            counts[k][idx] += 1;
        }
    }
    writeln!(
        md,
        "| path | pass | timeout | fail | no known value |\n|---|---|---|---|---|"
    )
    .unwrap();
    for k in 0..3 {
        writeln!(
            md,
            "| {} | {} | {} | {} | {} |",
            labels[k], counts[k][0], counts[k][1], counts[k][2], counts[k][3]
        )
        .unwrap();
    }
    writeln!(
        md,
        "\n| instance | rows×cols (int) | after presolve | known | A: microlp | B: aggr+microlp | C: aggr, no microlp presolve | B path | B/A time |\n|---|---|---|---|---|---|---|---|---|"
    )
    .unwrap();
    let fmt = |r: &RunResult| {
        let o = r.objective.map_or("-".to_string(), |o| format!("{o:.10e}"));
        let v = match r.verdict {
            Verdict::Pass => "ok",
            Verdict::Timeout => "limit",
            Verdict::Fail => "**FAIL**",
            Verdict::Unknown => "?",
        };
        format!("{v} {} {o}, {:.3}s, {} nodes", r.status, r.time, r.nodes)
    };
    for (inst, res) in insts.iter().zip(results.iter()) {
        let Some((rows, cols, ints, r)) = res else {
            writeln!(md, "| {} | read error | | | | | | | |", inst.name).unwrap();
            continue;
        };
        let known = match inst.value {
            Some(Some(v)) if v.is_nan() => "unbounded".into(),
            Some(Some(v)) => format!("{v}"),
            Some(None) => "infeasible".into(),
            None => "-".into(),
        };
        writeln!(
            md,
            "| {} | {}×{} ({}) | {}×{} | {} | {} | {} | {} | {} | {:.2} |",
            inst.name,
            rows,
            cols,
            ints,
            r[1].rows_after,
            r[1].cols_after,
            known,
            fmt(&r[0]),
            fmt(&r[1]),
            fmt(&r[2]),
            r[1].path,
            r[1].time / r[0].time
        )
        .unwrap();
        let _ = r[1].check_ok;
    }
    println!("{md}");
    if let Some(p) = out_path {
        std::fs::write(p, md).unwrap();
    }
}
