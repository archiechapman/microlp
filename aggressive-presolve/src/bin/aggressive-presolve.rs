//! `aggressive-presolve <file.mps|.lp[.gz]> [--no-presolve] [--no-microlp-presolve]
//! [--time-limit SECONDS]`: solves the file and prints the status, objective,
//! time and reduction stats.

use aggressive_presolve::{mps, solve, Options, Path};
use std::time::{Duration, Instant};

fn main() {
    let mut opts = Options::default();
    let mut file = None;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--no-presolve" => opts.presolve = false,
            "--no-microlp-presolve" => opts.microlp_presolve = false,
            "--time-limit" => {
                let s: f64 = args
                    .next()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or_else(|| usage("--time-limit needs seconds"));
                opts.time_limit = Some(Duration::from_secs_f64(s));
            }
            "-h" | "--help" => usage(""),
            _ if a.starts_with("--") => usage(&format!("unknown flag {}", a)),
            _ => file = Some(a),
        }
    }
    let file = file.unwrap_or_else(|| usage("no input file"));
    let t0 = Instant::now();
    let model = match mps::read_file(std::path::Path::new(&file)) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("error: {}", e);
            std::process::exit(2);
        }
    };
    let read_time = t0.elapsed();
    println!(
        "model: {} rows, {} cols ({} integer), {} nonzeros; read in {:.3}s",
        model.rows.len(),
        model.vars.len(),
        model.num_integer(),
        model.nnz(),
        read_time.as_secs_f64()
    );
    match solve(&model, &opts) {
        Ok(ans) => {
            println!("status: {:?}", ans.status);
            match ans.objective {
                Some(o) => println!("objective: {:.12e}", o),
                None => println!("objective: -"),
            }
            println!("time: {:.3}s", ans.time.as_secs_f64());
            println!("nodes: {}", ans.nodes());
            match &ans.path {
                Path::Reduced { retries } => println!("path: reduced ({} retries)", retries),
                Path::Original { reason } => println!("path: original ({:?})", reason),
            }
            for (k, a) in ans.attempts.iter().enumerate() {
                if let Some(st) = &a.presolve {
                    println!("attempt {} presolve: {}", k + 1, st);
                }
                println!(
                    "attempt {}: {:?}, {} nodes, {} LP iterations, {:.3}s{}",
                    k + 1,
                    a.status,
                    a.nodes,
                    a.lp_iterations,
                    a.time.as_secs_f64(),
                    if a.violations.is_empty() {
                        String::new()
                    } else {
                        format!(", check failed: {} violations, first {}", a.violations.len(), a.violations[0])
                    }
                );
            }
        }
        Err(e) => {
            eprintln!("error: {}", e);
            std::process::exit(1);
        }
    }
}

fn usage(msg: &str) -> ! {
    if !msg.is_empty() {
        eprintln!("error: {}", msg);
    }
    eprintln!(
        "usage: aggressive-presolve <file.mps|file.lp[.gz]> [--no-presolve] \
         [--no-microlp-presolve] [--time-limit SECONDS]"
    );
    std::process::exit(2)
}
