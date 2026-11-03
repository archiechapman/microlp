//! `aggressive-presolve <file.mps[.gz]|file.lp[.gz]> [--no-presolve] [--no-microlp-presolve] [--time-limit S]`

use aggressive_presolve::{read_file, solve, Options};
use std::time::{Duration, Instant};

fn main() {
    let mut args = std::env::args().skip(1);
    let mut file = None;
    let mut opts = Options::default();
    while let Some(a) = args.next() {
        match a.as_str() {
            "--no-presolve" => opts.presolve.enabled = false,
            "--no-microlp-presolve" => opts.microlp_presolve = false,
            "--time-limit" => {
                let s: f64 = args
                    .next()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or_else(|| usage("--time-limit needs seconds"));
                opts.time_limit = Some(Duration::from_secs_f64(s));
            }
            "-h" | "--help" => usage(""),
            f if !f.starts_with('-') && file.is_none() => file = Some(f.to_string()),
            o => usage(&format!("unknown argument {o}")),
        }
    }
    let file = file.unwrap_or_else(|| usage("no file given"));
    let t = Instant::now();
    let model = read_file(&file).unwrap_or_else(|e| {
        eprintln!("error: {e}");
        std::process::exit(2)
    });
    let read = t.elapsed();
    println!(
        "model: {} rows, {} cols ({} integer), {} nonzeros, read in {:.3}s",
        model.rows.len(),
        model.vars.len(),
        model.num_integer(),
        model.num_nonzeros(),
        read.as_secs_f64()
    );
    match solve(&model, &opts) {
        Ok(ans) => {
            println!("status: {:?}", ans.status);
            match ans.objective {
                Some(o) => println!("objective: {o:.12e}"),
                None => println!("objective: -"),
            }
            println!("time: {:.3}s", ans.time.as_secs_f64());
            println!("nodes: {}", ans.nodes);
            println!("path: {:?}", ans.path);
            if let Some(s) = &ans.presolve_stats {
                println!(
                    "presolve: {}x{} ({} nz) -> {}x{} ({} nz) in {:.3}s, {} passes",
                    s.rows_before,
                    s.cols_before,
                    s.nnz_before,
                    s.rows_after,
                    s.cols_after,
                    s.nnz_after,
                    s.time.as_secs_f64(),
                    s.passes
                );
                println!(
                    "  empty rows {}, singleton rows {}, redundant rows {}, forcing rows {}, \
                     int bounds tightened {}, fixed cols {}, empty cols {}, doubletons {}, \
                     substitutions {} (refused: fill {}, ratio {}), worst ratio {:.3e} -> {:.3e}",
                    s.empty_rows,
                    s.singleton_rows,
                    s.redundant_rows,
                    s.forcing_rows,
                    s.tightened_int_bounds,
                    s.fixed_cols,
                    s.empty_cols,
                    s.doubletons,
                    s.substitutions,
                    s.refused_fill,
                    s.refused_ratio,
                    s.worst_ratio_before,
                    s.worst_ratio_after
                );
            }
            for a in &ans.attempts {
                println!(
                    "attempt {}: {}x{}, {}, {} nodes, {:.3}s, check {:?} ({} violations)",
                    a.label,
                    a.rows,
                    a.cols,
                    a.outcome,
                    a.nodes,
                    a.time.as_secs_f64(),
                    a.check_ok,
                    a.violations
                );
            }
        }
        Err(e) => {
            println!("error: {e}");
            std::process::exit(1);
        }
    }
}

fn usage(msg: &str) -> ! {
    if !msg.is_empty() {
        eprintln!("error: {msg}");
    }
    eprintln!(
        "usage: aggressive-presolve <file.mps[.gz]|file.lp[.gz]> [--no-presolve] \
         [--no-microlp-presolve] [--time-limit SECONDS]"
    );
    std::process::exit(2)
}
