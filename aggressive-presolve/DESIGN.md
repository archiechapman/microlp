# aggressive-presolve: design

A standalone crate that reduces a MILP with reductions microlp's built-in
presolve deliberately refuses, solves the reduced model with microlp, maps the
answer back, and **checks the answer against the model as written** before
returning it. It depends on microlp only through its public API
(`Problem`, `SolveOptions`, `Solution`, `Error`).

## Why outside microlp

Master's presolve (`src/presolve/`, ARCHITECTURE §7.4) keeps a continuous var's
bounds as written, never turns a row into bounds, never changes the variable
set, and has no postsolve. That is what lets it keep resume, warm starts and
post-solve edits on the original indices. The reductions that paid off on our
models need all of those things:

- substitution of an implied-free column out of an equality row (the main win);
- doubleton equations `a·x + b·y = c` (eliminate `y`);
- singleton rows turned into bounds, continuous vars included;
- forcing rows that fix continuous vars at their bounds.

They change the variable set and take room from continuous vars, so they need a
postsolve and an independent check. This crate is that wrapper.

## Model

microlp's `Problem` has no getters, so the crate owns its model:

```text
Model { name, direction: Minimize|Maximize, obj_offset,
        vars: Vec<Var { name, lo, hi, cost, domain: Real|Integer }>,
        rows: Vec<Row { name, terms: Vec<(var, coeff)>, lo, hi }> }
```

A row is `lo <= a·x <= hi`; `<=` is `lo = -inf`, `>=` is `hi = +inf`, `=` is
`lo == hi`, and an MPS range is both finite. Binary is `Integer` on `[0, 1]`.
`Model::to_problem` builds a `microlp::Problem` through the public API: a
ranged row becomes two constraints, an integer var on `[0, 1]` becomes
`add_binary_var`, other integer vars use `add_integer_var` with their bounds
clamped to `i32` (an infinite side becomes `i32::MIN`/`i32::MAX`, as microlp's
docs recommend; a finite bound beyond `i32` is a documented limitation and is
reported by `to_problem` as an error rather than silently clamped).

## Readers

- **MPS**, fixed and free (`MpsFormat::{Free, Fixed, Auto}`; `Auto` splits on
  whitespace, which reads every fixed file whose names have no spaces, and
  `Fixed` uses the field columns 2-3, 5-12, 15-22, 25-36, 40-47, 50-61).
  Sections NAME, OBJSENSE (on its own line or inline, `MIN`/`MAX`/`MINIMIZE`/
  `MAXIMIZE`), OBJSENSE free-form, ROWS (N/L/G/E; the first N row is the
  objective, later N rows are dropped), COLUMNS with `'MARKER' 'INTORG'` /
  `'INTEND'`, RHS (an entry on the objective row is the *negated* objective
  constant), RANGES (E row: `R > 0` gives `[b, b+|R|]`, `R < 0` gives
  `[b-|R|, b]`; L row `[b-|R|, b]`; G row `[b, b+|R|]`), BOUNDS
  UP/LO/FX/FR/MI/PL/BV/LI/UI, ENDATA. Only the first RHS/RANGES/BOUNDS vector
  is honoured. Conventions: an integer column with no bound entry is binary
  (MIPLIB 3 / glpk; `MpsOptions::int_default_binary`), and a negative `UP` on a
  column with no `LO` sets the lower bound to `-inf` (with a warning, the
  classic MPSX rule). SOS, QUADOBJ, SC bounds and other unsupported sections
  are errors, never guessed at. `.gz` files are read through `flate2`.
- **CPLEX LP**, if time allows: objective, `Subject To`, `Bounds`,
  `General`/`Integer`, `Binary`, `End`.

## Reductions

All reductions are primal: they use bounds and rows, never the objective, so
no feasible point of the model is cut beyond the tolerances below, and the
objective only follows along (eliminated columns move their cost onto the
remaining columns and an offset). Iterated to a fixpoint (pass cap 100):

| reduction | soundness | integer rule |
|---|---|---|
| empty row | `0 ∈ [lo, hi]` or the model is infeasible | — |
| singleton row `lo <= a·x <= hi` | equivalent to a bound on `x` | integer `x`: the bound is relaxed outward by the row's contract over `|a|` and then rounded inward, so every integral `x` that meets the row within its tolerance stays |
| redundant row | activity bounds (with round-off allowance) lie inside `[lo, hi]` | — |
| forcing row | the extreme activity equals the side, so every var sits at the bound attaining it | an integer var is fixed only when its coefficient is larger than the room the round-off allowance leaves (it cannot move by a whole unit) |
| integer bound tightening | implied bounds from the row and the other vars' bounds | relaxed outward by the row's contract over `|a|` and the allowance, then rounded |
| fixed column | substituted into every row (in compensated arithmetic) and the objective constant | — |
| empty continuous column, zero cost | fixed at the point of its bounds nearest 0 | empty integer columns are kept, so distinct integer points stay distinct |
| doubleton equation `a·x + p·y = c` | `y = (c - a·x)/p`; `y`'s bounds become bounds on `x` | integer `y` only if `x` is integer and `c/p`, `a/p` are integral |
| implied-free substitution out of an equality row | `x_j = (b - Σ a_k x_k)/a_j`; `x_j`'s bounds are implied by the row and the other vars' bounds, so dropping them cuts nothing | integer `x_j` only if every other var of the row is integer and `b/a_j`, `a_k/a_j` are integral |

An integer column is eliminated only when its equality forces it integral, and
empty integer columns are kept. Parallel-row merging is deliberately left out:
it was measured slower on the case118 ROP model and dropped (history `dd276a2`).

### Guards on substitution

- **Pivot**: `|a_j| >= 1%` of the row's largest `|coeff|`, so reconstruction
  does not amplify the error of the other terms by more than 100.
- **Fill-in**: a substitution may add at most `max_fill` (default 10) net
  nonzeros.
- **Coefficient-ratio growth**: for every row the substitution rewrites, the
  new `max|a| / min|a|` must not exceed `max_ratio_growth` (default 4) times the
  larger of that row's ratio as written and the pivot row's ratio as written.
  The bound is against the *original* ratios, so repeated substitutions cannot
  compound it. (The old version grew the worst ratio about 10× with nothing to
  bound it.) The same cap applies to the objective-free rows only; the
  objective is not checked.
- **Cancellation**: a coefficient that cancels to `<= 1e-12` of the magnitudes
  it was computed from is dropped.

### Summation

Every activity sum is a Neumaier (compensated) sum, and every decision that
compares an activity with a side allows `(n + 2)·2^-53·m` for `n` terms of
magnitude `m` on top of it (the products are rounded once each). Every fold of
a fixed value into a row side (`b -= a·v`) and every substitution's rhs update
is carried in double-double (TwoSum, TwoProduct through `mul_add`), so 5,000
folds of `1e6`-sized values lose nothing. This is the gap a review of
2026-11-03 found in master's presolve, which allows only `1e-14·m` for plain
`f64` sums over 2,000+ terms and so can cut integer points. Postsolve's
reconstruction and the final check are compensated too.

## Postsolve

Each elimination appends an op: `Fix { col, value }` or
`Substitute { col, pivot, rhs, terms }`. An op only references columns that are
still alive when it is made, so replaying the ops **in reverse** reconstructs
every column from columns that are already known. The objective constant moved
out by eliminations is kept as `offset`. Kept columns map through `kept` /
`reduced_of`.

## Check and fallback

`solve` runs: presolve → `Model::to_problem` → microlp (its own presolve on by
default) → postsolve → **check** on the original model:

- every row: violation of `[lo, hi]` by the compensated activity within
  `max(feas, 1e-14·m)`, `m = |rhs| + Σ|a_j x_j|` (microlp's `row_tolerance`;
  `rhs` is the side checked);
- every bound within `max(feas, 1e-14·|bound|)`;
- every integer var within `int_tol` of an integer;
- the objective recomputed on the original costs.

On a failure the failing rows are *protected* (never dropped, never used as a
substitution source, and none of their columns eliminated by substitution or
doubleton) and failing columns are frozen (never eliminated), and presolve is
rerun from scratch and the model re-solved; after `max_retries` (default 2)
the original model is solved with microlp alone. A verdict without a point
(`Infeasible`, `Unbounded`) on the reduced model, and presolve's own
`Infeasible`, are confirmed by solving the original model: presolve carries no
certificate. So a reduction that loses accuracy costs a re-solve, never a wrong
answer. `Answer::path` reports which path answered (`Presolved`,
`PresolvedRetry(k)`, `Original(reason)`).

What the check cannot see: a reduction that wrongly cuts feasible points gives
a feasible but possibly worse answer. That is why every reduction above is exact
in exact arithmetic and every tolerance in it errs toward keeping points.

## API

```rust
pub fn presolve(model: &Model, opts: &Options) -> Result<Presolved, PresolveInfeasible>;
pub struct Presolved { pub reduced: Model, pub postsolve: Postsolve, pub stats: Stats }
pub fn solve(model: &Model, opts: &Options) -> Result<Answer, Error>;
pub struct Answer { status, objective, values, path, presolve_stats, nodes, attempts, ... }
```

Binary: `aggressive-presolve <file.mps[.gz]> [--no-presolve] [--no-microlp-presolve] [--time-limit S]`.

## Not in version 1

Duals (`Solution::dual` is not on master), post-solve edits
(`Solution::add_constraint`/`fix_var`), warm starts, resume of an interrupted
solve, and enumeration. The answer is a point and an objective in the original
variables; nothing of microlp's `Solution` survives the postsolve. Stated in the
crate's API docs too.
