//! Aggressive presolve for MILPs, wrapping [microlp](https://github.com/Specy/microlp).
//!
//! [`solve`] reduces a [`Model`] with reductions microlp's built-in presolve
//! deliberately refuses (implied-free substitution, doubleton equations,
//! singleton rows as bounds for continuous vars, forcing rows on continuous
//! vars), solves the reduced model with microlp, maps the answer back with
//! [`Postsolve`], and checks it against the model as written at microlp's
//! tolerance contract. A failed check costs a re-solve with the reductions
//! behind the failing rows undone, and in the end a solve of the original
//! model: never a wrong answer. See `DESIGN.md`.
//!
//! # Not in version 1
//!
//! The answer is a point and an objective in the original variables, nothing
//! more. There are no duals (`Solution::dual` is not on microlp's master), no
//! post-solve edits (`add_constraint`, `fix_var`), no warm starts, no resume of
//! an interrupted solve and no enumeration: none of microlp's `Solution`
//! survives the postsolve.

pub mod check;
pub mod lp;
pub mod model;
pub mod mps;
pub mod presolve;
pub mod solve;
pub mod sum;

pub use check::{check, CheckOptions, CheckReport, Violation};
pub use model::{Direction, Domain, Model, Row, Var};
pub use presolve::{presolve, Postsolve, PresolveInfeasible, PresolveOptions, Presolved, Stats};
pub use solve::{solve, Answer, Attempt, Error, Options, Path, Status};

/// Reads a model from a file, by extension: `.mps`, `.lp`, optionally `.gz`.
pub fn read_file(path: impl AsRef<std::path::Path>) -> Result<Model, String> {
    let path = path.as_ref();
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let name = path
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_lowercase();
    let (text, name) = if let Some(stem) = name.strip_suffix(".gz") {
        use std::io::Read;
        let mut s = String::new();
        flate2::read::GzDecoder::new(&bytes[..])
            .read_to_string(&mut s)
            .map_err(|e| format!("{}: gunzip: {e}", path.display()))?;
        (s, stem.to_string())
    } else {
        (
            String::from_utf8(bytes).map_err(|e| format!("{}: {e}", path.display()))?,
            name,
        )
    };
    if name.ends_with(".lp") {
        lp::parse(&text)
    } else {
        mps::parse(&text, &mps::MpsOptions::default()).map(|r| r.model)
    }
    .map_err(|e| format!("{}: {e}", path.display()))
}
