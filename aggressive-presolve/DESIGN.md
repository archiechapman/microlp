# aggressive-presolve: design

A standalone crate that reduces a MILP with reductions microlp's own presolve
deliberately refuses, solves the reduced model with microlp, maps the answer
back, and **checks the answer against the model as written** before returning
it. A reduction that loses accuracy costs a re-solve, never a wrong answer.

It depends on microlp only through its public API (`Problem`, `SolveOptions`,
`SolveOutcome`, `Solution`). microlp's `src/` is not touched.

## 1. Why outside microlp

microlp's built-in presolve (`src/presolve/mod.rs`) keeps a continuous var's
bounds as written, never turns a row into bounds, never changes the variable
set and has no postsolve, so that resume, warm starts and post-solve edits work
off the original indices, and so that the engine's tolerance contract holds
for every row with no after-the-fact check. That rules out the reductions that
paid off on our models:

* substitution of an implied-free column out of an equality (the main win);
* doubleton equations;
* singleton rows turned into bounds of continuous vars;
* forcing rows on continuous vars.

These need a postsolve and can, through round-off and the engine's bound
tolerance, produce a point that misses a row of the original model by more
than the contract. This crate accepts that risk because it checks every answer
and falls back.

## 2. The model

`microlp::Problem` has no getters, so the crate owns its model:

```text
Model { name, direction: Min | Max, obj_offset, vars: Vec<Var>, rows: Vec<Row> }
Var   { name, lo, hi, obj, kind: Continuous | Integer }   // binary = Integer in [0, 1]
Row   { name, terms: Vec<(var, coeff)>, lo, hi }          // lo = hi: equality;
                                                          // one side infinite: <= or >=;
                                                          // both finite: ranged
```

Rows are stored as `lo <= a·x <= hi` so that MPS RANGES and both-sided rows
need no special case; `Model::add_row(terms, Sense::{Le,Ge,Eq}, rhs)` is the
usual entry point. Duplicate entries of one var in a row are summed, explicit
zeros dropped, and NaN/inconsistent data rejected by `Model::validate`.

`Model::to_problem()` builds a `microlp::Problem` through the public API: a
ranged row becomes a `>=` and a `<=` row; an integer var goes through
`add_integer_var`, whose bounds are `i32`. An infinite integer bound becomes
`i32::MIN`/`i32::MAX` (microlp's documented idiom); a finite integer bound
outside the `i32` range is emitted as `i32::MIN`/`MAX` plus a singleton row
carrying the real bound. The objective offset is added back by the crate.

## 3. Readers

* **MPS**, fixed and free (whitespace-tokenized; names with spaces in fixed
  columns are read by column position when the line has the fixed layout).
  Sections NAME, OBJSENSE (`MIN`/`MAX`/`MINIMIZE`/`MAXIMIZE`, same line or next
  line), ROWS (N/L/G/E; the first N row is the objective, later N rows are
  ignored), COLUMNS with `'MARKER' 'INTORG'`/`'INTEND'`, RHS (an entry on the
  objective row is the *negated* objective constant), RANGES (the standard
  table: `E` row with `R > 0` gives `[b, b+|R|]`, `R < 0` gives `[b-|R|, b]`;
  `L` gives `[b-|R|, b]`; `G` gives `[b, b+|R|]`), BOUNDS
  UP/LO/FX/FR/MI/PL/BV/LI/UI, ENDATA. Conventions: a negative `UP` with no
  `LO` on a continuous column moves the lower bound to `-inf` (classic MPS,
  with a warning); an integer column with no bounds at all defaults to
  `[0, 1]` (the MPSX convention MIPLIB 3 assumes; `MpsOptions` can switch it to
  `[0, +inf)`); `BV` makes the column integer in `[0, 1]`; `LI`/`UI` make it
  integer. Only the first RHS/RANGES/BOUNDS vector is used. Unsupported
  sections (SOS, QUADOBJ, QMATRIX, INDICATORS, ...) and `SC` bounds are
  errors, never guesses. `.gz` files are read through `flate2`.
* **CPLEX LP**: objective, `Subject To` with named rows, `<=`/`>=`/`=`/`=<`/`=>`,
  constants on either side, `Bounds` (including `free`, `-inf`, two-sided
  `l <= x <= u`), `General`/`Generals`/`Integers`, `Binary`/`Binaries`, `End`.
  Not supported: ranged-row syntax `R1: -x1 + x2 >= 1 <= 5` (not standard),
  semi-continuous, SOS, quadratic and indicator constraints (errors).

A writer for free MPS exists for tests and the suite exporter.

## 4. The reductions

Presolve works on a copy: rows `lo <= Σ a_j x_j <= hi` with the bounds held in
double-double (`hi + lo` parts, §7), columns with bounds, cost, kind, row list.
All reductions are primal: they use rows and bounds, never the objective,
except to carry costs through a substitution. They are iterated to a fixpoint
(at most 100 passes), substitution only once the cheap ones stop.

Notation: `f` = feasibility tolerance (microlp's, default `1e-7`);
`btol_j` = `max(f, 1e-14·|bound_j|)`; `rtol(m) = max(f, 1e-14·m)`;
`m = |rhs| + Σ|a_j x_j|` at the relevant corner.

| Reduction | What it does | Soundness | Integer rule |
|---|---|---|---|
| empty row | drop, or presolve-infeasible if `0 ∉ [lo, hi]` beyond `f` | exact | — |
| singleton row `lo <= a x <= hi` | becomes the bound `[lo/a, hi/a]` (swapped for `a < 0`) intersected with x's | exact up to one division | bounds rounded inward, keeping every integer whose row residual is within `rtol` |
| redundant row | dropped when the activity range over the bounds lies inside `[lo, hi]` with a margin of `Σ|a_j|·btol_j` + round-off | exact: every point within the bounds meets it; the margin covers microlp leaving a bound by its tolerance | — |
| forcing row | activity min ≥ `hi` (or max ≤ `lo`) within round-off: every var fixed at the bound attaining it, row dropped | the row holds only at that corner; if min exceeds `hi` by at most `rtol(m)` the corner is the only point within contract | integer vars are fixed at their (integral) bounds |
| integer bound tightening | an integer var's bounds tightened from the row and the others' bounds | keeps every integer value meeting the row within `rtol` | outward margin `max(int_tol, rtol/|a|)` before rounding; bounds beyond `1e9` are not derived |
| fixed column (`lo == hi`) | value folded into every row's bounds (double-double) and into the objective offset | exact | the value is integral (bounds were rounded) |
| empty column, continuous, zero cost | fixed at the value in its bounds closest to 0 | it touches nothing | **empty integer columns are kept**, so distinct integer points stay distinct |
| doubleton equation `p·y + a·x = b` | `y` eliminated, `y = (b − a x)/p`; y's bounds move to x | exact, a linear change of variables | `y` integer is eliminated only when `x` is integer and `b/p`, `a/p` are integers, so integral x gives integral y; x's new bounds are rounded |
| implied-free substitution | equality row, column `j` whose bounds are implied by the row and the others' bounds (with a margin of `Σ|a_i/p|·btol_i`); `x_j` substituted out of every other row and the objective, the row dropped | the implied bounds make j's own bounds redundant, so dropping them with the row is exact | same integer rule as the doubleton |

Parallel-row merging is **not** ported: it was measured slower on the model the
old presolve was built for and dropped (old history `dd276a2`). No dual
reductions (dominated columns, dual fixing) in v1.

### Substitution guards

A substitution of column `j` (pivot `p`) out of row `r` is taken only if:

1. **pivot**: `|p| >= 0.01 · max_i |a_ri|`;
2. **fill-in**: it adds at most `max_fill = 10` nonzeros net;
3. **ratio growth**: for every row `k` it rewrites, the new
   `max|coeff| / min|coeff|` is at most `max_ratio_growth` (default 4) times
   the ratio of row `k` in the original model (at least 1). The cap is against
   the *original* ratio, so it cannot compound over many substitutions;
4. **cancellation**: a rewritten coefficient below `1e-12` of the magnitudes
   it came from is dropped; if that happens the row's rhs absorbs nothing
   (the term was zero up to round-off), and the row is marked rewritten.

Doubleton equations go through guards 1 and 3 (they never add fill net).

## 5. Postsolve

Every elimination is recorded as an op, in order:

```text
Fix        { col, value }
Substitute { col, pivot, rhs: DD, terms: Vec<(col, coeff)> }   // x_col = (rhs − Σ a x)/pivot
```

Postsolve places the reduced point's values on the kept columns and replays
the ops **in reverse**: each op only reads columns that were alive when it was
recorded, which are either kept or eliminated by a later op, already
reconstructed. Substitution sums are compensated. The original objective is
recomputed from the original model at the postsolved point (compensated);
the reduced objective plus the offset is kept for cross-checking.

## 6. Check and fallback

The postsolved point is checked against **every row and bound of the
original model** at microlp's contract (ARCHITECTURE §7.1,
`src/solver/tolerances.rs::row_tolerance`):

* row: activity `s = Σ a_j x_j` (compensated), `m = |rhs| + Σ|a_j x_j|`
  (rhs = the violated side), violated if `s < lo − tol` or `s > hi + tol`
  with `tol = max(feas, 1e-14·m)`; a non-finite activity is a violation;
* bound: `x < lo − max(feas, 1e-14|lo|)` or `x > hi + max(feas, 1e-14|hi|)`;
* integer var: `|x − round(x)| > int_tol`.

On a failure the solve is retried, in this order:

1. **protected re-solve**: presolve again with the failing rows and the
   failing vars *protected*. A protected row is never removed (singleton,
   forcing, redundant, doubleton, substitution pivot row) and no column in it
   is substituted, so it reaches microlp as written, less only fixed columns
   folded into its bounds; a protected var is never eliminated. microlp then
   holds that row itself. Protection accumulates across retries
   (`Options::max_retries`, default 2).
2. **original**: solve the model as written with microlp (no aggressive
   presolve).

`Infeasible` from presolve itself is not final: the original model is solved.
`Infeasible`/`Unbounded` from microlp on the reduced model are confirmed on
the original model (`Options::confirm_infeasible`, default on), since the
crate has no certificate of its own. The `Answer` reports which path answered
(`Path::Reduced { retries }`, `Path::Original { reason }`) and every check
failure seen on the way.

The check guarantees feasibility, not optimality: a reduction that cut the
optimum through round-off would give a worse answer the check cannot see. The
reductions are exact up to round-off and every margin errs toward keeping
points (and keeping rows), which is the direction that cannot cut the optimum.

## 7. Summation

Every activity sum, implied-bound computation, rhs fold and postsolve
substitution is done in compensated arithmetic: products by `TwoProduct`
(`fma`), sums by Neumaier/`TwoSum`, and row bounds carried as double-double
through every fold. A review on 2026-11-03 found that microlp's own presolve
cuts integer points by allowing only `1e-14·m` for plain `f64` sums over
2,000+ terms (error up to `(n+2)·2^-53·m`); with compensation the error is a
few ulps of the result plus `~2^-104·m`, so a `1e-14·m` allowance is ample.
Pinned by `long_row_round_off` (5,000 fixed terms of `~1.5e6`, `b = Σc + 7`
exactly, `max y`, answer 7).

## 8. API (v1)

```rust
pub fn presolve(model: &Model, opts: &Options) -> Result<Presolved, PresolveInfeasible>;
pub struct Presolved { pub reduced: Model, pub postsolve: Postsolve, pub stats: Stats }
pub fn solve(model: &Model, opts: &Options) -> Result<Answer, Error>;
pub struct Answer { status, objective, values, path, check_failures, stats, timings, ... }
```

`Options`: `presolve` (aggressive on/off), `microlp_presolve`, `time_limit`,
`node_limit`, `mip_gap`, `int_tol`, `feasibility`, `max_fill`,
`max_ratio_growth`, `max_retries`, `confirm_infeasible`.

Binary: `aggressive-presolve <file.mps|.lp[.gz]> [--no-presolve]
[--no-microlp-presolve] [--time-limit S]`, printing status, objective, time and
reduction stats.

**Left out of v1** (stated in the API docs): duals (`Solution::dual` is not on
microlp master), post-solve edits (`add_constraint`/`fix_var` on the answer),
warm starts, resume of an interrupted solve, and solution enumeration. An
`Answer` is plain data: values in original indices, nothing live.

## 9. Testing

* unit tests per reduction and for postsolve round trips (ported from the old
  `src/presolve.rs` tests, rewritten on the crate's model);
* reader tests (fixed/free MPS, markers, every bound type, RANGES, OBJSENSE,
  objective constant; LP format);
* a property test: random small MILPs, presolve on vs off agree on the optimum
  and the postsolved point meets every original row;
* the pinned long-row round-off case (§7);
* a suite runner (`examples/suite.rs`) over microlp's correctness suite: its 28
  MPS files and its Rust-built MILP families (exported through the suite's own
  generators, compiled into the example by `#[path]`), each solved three ways.
  Results in `REPORT.md`.
