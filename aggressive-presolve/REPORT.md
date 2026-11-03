# aggressive-presolve on microlp's correctness suite

Run on 2026-11-03, release build, 60 s per solve, two instances at a time on
a 4-core container, with
`cargo run --release --example families` (export) and
`cargo run --release --example suite -- --time-limit 60 --parallel 2`.

Three paths per instance:

- **A**: microlp alone (its own presolve on), `Options::presolve.enabled = false`;
- **B**: aggressive presolve, then microlp with its own presolve on (the default);
- **C**: aggressive presolve, then microlp with its own presolve off.

Every answer of every path is postsolved and checked against the model as
written (rows, bounds, integrality, at microlp's contract) before it is
judged. Instances: the suite's 28 MPS files (27 `.mps` + `greenbea.mps.gz`),
its two CPLEX-LP CFL instances (`medium/milpbench`, read by the crate's LP
reader), and 137 instances of the suite's Rust-built MILP families
(`milp_families.rs`, `milp_random.rs`, `milp_unit.rs`) exported to MPS by
`examples/families` with the suite's own oracle values (7 custom cases that
build no plain model are skipped). The xhard MILPBench files are left out
(10-minute budgets, beyond microlp's ceiling). 167 instances in all.

A known value is matched with the suite's rule `|o - v| <= abs + rel·|v|` and
its per-instance tolerances; an objective that misses only because integer
vars sit off their integers within `int_tol` is also judged on the point with
the integers rounded, where that point still checks. An expected
`Infeasible`/`Unbounded` must be answered so. A time limit is a pass only for
no answer, or an incumbent not better than the known optimum.

## Summary

| path | pass | timeout | fail | no known value |
|---|---|---|---|---|
| microlp alone | 166 | 1 | 0 | 0 |
| aggressive + microlp | 166 | 1 | 0 | 0 |
| aggressive, microlp presolve off | 165 | 1 | 1 | 0 |

### Paths taken (B)

| path | instances |
|---|---|
| presolved (first reduced model's answer checked) | 148 |
| presolved after a retry | 0 |
| original: presolve's own `Infeasible`, confirmed | 12 |
| original: reduced model `Infeasible`, confirmed | 4 |
| original: reduced model `Unbounded`, confirmed | 3 |

Every `original` path is an instance the suite expects to be infeasible or
unbounded: presolve's verdicts carry no certificate, so they are confirmed on
the model as written, and all 19 were. **No postsolved point failed the check
on any instance, so no re-solve was ever needed.** 53 of the 167 instances
were reduced.

## Disagreements and check failures

| instance | path | what | cause |
|---|---|---|---|
| `miplib/misc03` | C | objective `3359.99989576` vs known `3360` (tolerance `1.03e-4`, off by `1.04e-4`) | Not presolve: the aggressive presolve changes nothing on misc03 (96×160 in, 96×160 out, one pass), and microlp alone with its presolve off returns the same point (`aggressive-presolve misc03.mps --no-presolve --no-microlp-presolve`). Several binaries sit `6e-7` off their integers, within `int_tol = 1e-6`, and the one continuous var, which carries the cost, follows them; rounding the binaries breaks row 0 by `1e-4`. The point meets microlp's contract; the suite's `EXACT` tolerance is tighter than what `int_tol` lets that objective move. |
| `miplib/vpm1` | A, B, C | time limit, incumbent consistent | The suite lists vpm1 in `INT_MAY_INTERRUPT`. |

No check failure on any path, and no other disagreement with a known value.

## Largest speedups and slowdowns (B vs A, instances where A takes at least 50 ms)

| instance | A | B | C | note |
|---|---|---|---|---|
| `netlib/greenbea` | 12.10 s | 5.74 s (**2.1×**) | 5.59 s | 2392×5405 → 1237×3405: 383 doubletons, 340 implied-free substitutions, 251 forcing rows; worst row ratio 6.6e4 → 1.7e4 |
| `miplib/p0201` | 1.94 s, 3852 nodes | 1.12 s (**1.7×**), 2277 nodes | 1.43 s, 3171 nodes | 6 integer columns fixed by bound tightening |
| `miplib/misc03` | 0.52 s | 0.44 s | 0.44 s | model unchanged: timing noise |
| `miplib/mod008` | 2.75 s | 2.33 s | 2.66 s | model unchanged, same tree: timing noise |
| `miplib/bell3a` | 20.65 s, 111001 nodes | 18.84 s, 111003 nodes | **24.95 s**, 128617 nodes | 8 singleton rows to bounds; without microlp's presolve the tree grows 16% |
| `miplib/lseu` | 1.13 s, 19563 nodes | 1.13 s | **1.41 s**, 22550 nodes | model unchanged: C is microlp without its presolve |
| `milpbench/CFL_easy_instance_12` | 0.11 s | **0.14 s** | 0.21 s | 800×160400, nothing to reduce; the one presolve pass that finds nothing, with building its working copy, costs 0.03–0.1 s on 160k columns |
| `milpbench/CFL_easy_instance_15` | 0.12 s | **0.18 s** | 0.22 s | as above |
| `miplib/stein27`, `stein45`, `gt2` | | within ±5% | | nothing reduced; noise |

Totals over all 167 instances: A 129.1 s, B 119.9 s, C 125.1 s (vpm1's three
60 s limits included). The netlib LPs all shrink (sc50a 50×48 → 29×28,
scagr7 129×140 → 59×103, beaconfd 173×262 → 57×122) but solve in
milliseconds either way. On the MIPLIB MILPs the aggressive reductions find
little (these files have few equality rows with free or implied-free
columns); the real gain is on the LP greenbea. Turning microlp's own
presolve off behind ours (C) is never a clear win and loses on bell3a, lseu
and p0033, so B stays the default.

## The long-row round-off case

`tests/long_row.rs` pins `max y` s.t. `y + Σ_{i<5000} c·x_i <= b`, every
`x_i` fixed at 1, all `c = 1666022.6166951393` (a `1e6·U(1,2)` draw chosen so
a plain `f64` fold of the side leaves `6.9996` instead of 7), `y` integer in
`[0, 100]`, `b = exact(Σc) + 7`. The crate answers **7**. microlp alone answers
**6** with its own presolve on (the 2026-11-03 review's finding: its `1e-14·m`
allowance on a plain sum over 5,000 terms cuts the integer point 7), and
`6.99999994` with it off.

## Open problems

- The retry path (protect the failing rows, freeze the failing columns,
  re-presolve, re-solve) never triggered on the suite, and no test makes it
  trigger deterministically; it is covered only by the unit tests of
  protections themselves.
- Presolve cost on very wide models: each pass recomputes activities and
  looks coefficients up by linear scan; on the 160k-column CFL instances,
  where nothing reduces, one pass and the working copy cost 0.03–0.1 s,
  as much as microlp's whole solve. A pass that changes nothing in a row's
  columns could skip it.
- `Stats::refused_fill`/`refused_ratio` count refusals per attempt, and a row
  is attempted again every substitution pass, so they overcount distinct
  refusals.
- Integer vars with an infinite bound go to microlp with `i32::MIN`/`i32::MAX`
  (its documented convention); a finite integer bound outside `i32` is
  refused (`Model::to_problem`).
- An integer polish step (round, fix, re-solve the continuous part) would turn
  int_tol artifacts like misc03's into exact objectives; not in v1.
- The CPLEX LP reader covers what the suite's files use, not the whole
  format (no semi-continuous, SOS, indicator or quadratic sections).

## Per-instance results

Rows×cols are before presolve, and after the aggressive presolve (B; microlp's
own presolve then reduces further, which is not shown). Times are per solve
call, re-solves included, reading excluded.

| instance | rows×cols (int) | after presolve | known | A: microlp | B: aggr+microlp | C: aggr, no microlp presolve | B path | B/A time |
|---|---|---|---|---|---|---|---|---|
| bell3a | 123×133 (71) | 115×133 | 878430.32 | ok Optimal 8.7843031600e5, 20.651s, 111001 nodes | ok Optimal 8.7843031600e5, 18.839s, 111003 nodes | ok Optimal 8.7843031600e5, 24.952s, 128617 nodes | presolved | 0.91 |
| enigma | 21×100 (100) | 21×100 | 0 | ok Optimal 0.0000000000e0, 0.700s, 4425 nodes | ok Optimal 0.0000000000e0, 0.689s, 4425 nodes | ok Optimal 0.0000000000e0, 0.665s, 4425 nodes | presolved | 0.98 |
| gt2 | 29×188 (188) | 29×188 | 21166 | ok Optimal 2.1166000000e4, 0.214s, 3831 nodes | ok Optimal 2.1166000000e4, 0.224s, 3831 nodes | ok Optimal 2.1166000000e4, 0.177s, 3324 nodes | presolved | 1.05 |
| mod008 | 6×319 (319) | 6×319 | 307 | ok Optimal 3.0700000000e2, 2.751s, 19475 nodes | ok Optimal 3.0700000000e2, 2.330s, 19475 nodes | ok Optimal 3.0700000000e2, 2.656s, 19475 nodes | presolved | 0.85 |
| stein45 | 331×45 (45) | 331×45 | 30 | ok Optimal 3.0000000000e1, 27.640s, 58045 nodes | ok Optimal 3.0000000000e1, 27.954s, 58045 nodes | ok Optimal 3.0000000000e1, 26.145s, 58045 nodes | presolved | 1.01 |
| vpm1 | 234×378 (168) | 174×318 | 20 | limit Feasible 2.0000000000e1, 60.070s, 258106 nodes | limit Feasible 1.9999999366e1, 60.034s, 260736 nodes | limit Feasible 2.0000000000e1, 60.047s, 283256 nodes | presolved | 1.00 |
| beaconfd | 173×262 (0) | 57×122 | 33592.485807 | ok Optimal 3.3592485807e4, 0.004s, 0 nodes | ok Optimal 3.3592485807e4, 0.002s, 0 nodes | ok Optimal 3.3592485807e4, 0.002s, 0 nodes | presolved | 0.46 |
| brandy | 220×249 (0) | 94×176 | 1518.5098965 | ok Optimal 1.5185098965e3, 0.012s, 0 nodes | ok Optimal 1.5185098965e3, 0.007s, 0 nodes | ok Optimal 1.5185098965e3, 0.007s, 0 nodes | presolved | 0.56 |
| greenbea | 2392×5405 (0) | 1237×3405 | -72555248.13 | ok Optimal -7.2555248130e7, 12.095s, 0 nodes | ok Optimal -7.2555248131e7, 5.736s, 0 nodes | ok Optimal -7.2555248131e7, 5.585s, 0 nodes | presolved | 0.47 |
| israel | 174×142 (0) | 163×142 | -896644.82186 | ok Optimal -8.9664482186e5, 0.009s, 0 nodes | ok Optimal -8.9664482186e5, 0.010s, 0 nodes | ok Optimal -8.9664482186e5, 0.010s, 0 nodes | presolved | 1.10 |
| sctap1 | 300×480 (0) | 300×480 | 1412.25 | ok Optimal 1.4122500000e3, 0.009s, 0 nodes | ok Optimal 1.4122500000e3, 0.010s, 0 nodes | ok Optimal 1.4122500000e3, 0.012s, 0 nodes | presolved | 1.09 |
| CFL_easy_instance_12 | 800×160400 (148800) | 800×160400 | 10000.194107917776 | ok Optimal 1.0000194108e4, 0.107s, 0 nodes | ok Optimal 1.0000194108e4, 0.139s, 0 nodes | ok Optimal 1.0000194108e4, 0.213s, 0 nodes | presolved | 1.30 |
| CFL_easy_instance_15 | 800×160400 (150400) | 800×160400 | 9959.569649350798 | ok Optimal 9.9595696494e3, 0.124s, 0 nodes | ok Optimal 9.9595696494e3, 0.181s, 0 nodes | ok Optimal 9.9595696494e3, 0.215s, 0 nodes | presolved | 1.46 |
| flugpl | 18×18 (11) | 15×16 | 1201500 | ok Optimal 1.2015000000e6, 0.066s, 4309 nodes | ok Optimal 1.2015000000e6, 0.056s, 4309 nodes | ok Optimal 1.2015000000e6, 0.055s, 4309 nodes | presolved | 0.85 |
| lseu | 28×89 (89) | 28×89 | 1120 | ok Optimal 1.1200000000e3, 1.126s, 19563 nodes | ok Optimal 1.1200000000e3, 1.125s, 19563 nodes | ok Optimal 1.1200000000e3, 1.410s, 22550 nodes | presolved | 1.00 |
| misc03 | 96×160 (159) | 96×160 | 3360 | ok Optimal 3.3600000000e3, 0.522s, 629 nodes | ok Optimal 3.3600000000e3, 0.437s, 629 nodes | **FAIL** Optimal 3.3599998958e3, 0.439s, 596 nodes | presolved | 0.84 |
| p0033 | 16×33 (33) | 15×33 | 3089 | ok Optimal 3.0889998974e3, 0.037s, 1040 nodes | ok Optimal 3.0889998974e3, 0.034s, 1040 nodes | ok Optimal 3.0890000000e3, 0.065s, 1997 nodes | presolved | 0.91 |
| p0201 | 133×201 (201) | 133×195 | 7615 | ok Optimal 7.6150000000e3, 1.940s, 3852 nodes | ok Optimal 7.6150000000e3, 1.119s, 2277 nodes | ok Optimal 7.6150000000e3, 1.425s, 3171 nodes | presolved | 0.58 |
| rgn | 24×180 (100) | 24×180 | 82.19999924 | ok Optimal 8.2199999240e1, 0.479s, 4643 nodes | ok Optimal 8.2199999240e1, 0.466s, 4643 nodes | ok Optimal 8.2199999240e1, 0.482s, 4643 nodes | presolved | 0.97 |
| stein27 | 118×27 (27) | 118×27 | 18 | ok Optimal 1.8000000000e1, 0.475s, 4364 nodes | ok Optimal 1.8000000000e1, 0.501s, 4364 nodes | ok Optimal 1.8000000000e1, 0.536s, 4364 nodes | presolved | 1.05 |
| adlittle | 56×97 (0) | 53×96 | 225494.96316 | ok Optimal 2.2549496316e5, 0.003s, 0 nodes | ok Optimal 2.2549496316e5, 0.002s, 0 nodes | ok Optimal 2.2549496316e5, 0.002s, 0 nodes | presolved | 0.93 |
| afiro | 27×32 (0) | 20×28 | -464.75314286 | ok Optimal -4.6475314286e2, 0.000s, 0 nodes | ok Optimal -4.6475314286e2, 0.000s, 0 nodes | ok Optimal -4.6475314286e2, 0.000s, 0 nodes | presolved | 0.73 |
| blend | 74×83 (0) | 55×66 | -30.812149846 | ok Optimal -3.0812149846e1, 0.003s, 0 nodes | ok Optimal -3.0812149927e1, 0.002s, 0 nodes | ok Optimal -3.0812149927e1, 0.002s, 0 nodes | presolved | 0.66 |
| kb2 | 43×41 (0) | 38×36 | -1749.9001299 | ok Optimal -1.7499001299e3, 0.001s, 0 nodes | ok Optimal -1.7499001299e3, 0.001s, 0 nodes | ok Optimal -1.7499001299e3, 0.001s, 0 nodes | presolved | 0.92 |
| sc105 | 105×103 (0) | 60×59 | -52.202061212 | ok Optimal -5.2202061241e1, 0.002s, 0 nodes | ok Optimal -5.2202061212e1, 0.001s, 0 nodes | ok Optimal -5.2202061212e1, 0.001s, 0 nodes | presolved | 0.57 |
| sc50a | 50×48 (0) | 29×28 | -64.575077059 | ok Optimal -6.4575077081e1, 0.001s, 0 nodes | ok Optimal -6.4575077059e1, 0.000s, 0 nodes | ok Optimal -6.4575077059e1, 0.000s, 0 nodes | presolved | 0.73 |
| sc50b | 50×48 (0) | 28×28 | -70 | ok Optimal -7.0000000031e1, 0.001s, 0 nodes | ok Optimal -7.0000000000e1, 0.001s, 0 nodes | ok Optimal -7.0000000000e1, 0.000s, 0 nodes | presolved | 0.69 |
| scagr7 | 129×140 (0) | 59×103 | -2331389.2548 | ok Optimal -2.3313898243e6, 0.004s, 0 nodes | ok Optimal -2.3313898243e6, 0.002s, 0 nodes | ok Optimal -2.3313898243e6, 0.002s, 0 nodes | presolved | 0.59 |
| share2b | 96×79 (0) | 89×75 | -415.73224074 | ok Optimal -4.1573224074e2, 0.003s, 0 nodes | ok Optimal -4.1573224074e2, 0.004s, 0 nodes | ok Optimal -4.1573224074e2, 0.003s, 0 nodes | presolved | 1.06 |
| stocfor1 | 117×111 (0) | 59×68 | -41131.976219 | ok Optimal -4.1131976219e4, 0.002s, 0 nodes | ok Optimal -4.1131976219e4, 0.002s, 0 nodes | ok Optimal -4.1131976219e4, 0.002s, 0 nodes | presolved | 0.87 |
| family/family_degenerate-lp_3x3-0 | 6×9 (0) | 6×9 | 154 | ok Optimal 1.5400000000e2, 0.000s, 0 nodes | ok Optimal 1.5400000000e2, 0.000s, 0 nodes | ok Optimal 1.5400000000e2, 0.000s, 0 nodes | presolved | 0.61 |
| family/family_degenerate-lp_4x4-1 | 8×16 (0) | 8×16 | 161 | ok Optimal 1.6100000000e2, 0.000s, 0 nodes | ok Optimal 1.6100000000e2, 0.000s, 0 nodes | ok Optimal 1.6100000000e2, 0.000s, 0 nodes | presolved | 0.87 |
| family/family_degenerate-lp_5x4-2 | 9×20 (0) | 9×20 | 203 | ok Optimal 2.0300000000e2, 0.000s, 0 nodes | ok Optimal 2.0300000000e2, 0.000s, 0 nodes | ok Optimal 2.0300000000e2, 0.000s, 0 nodes | presolved | 0.93 |
| family/family_eq-chain_n06-0 | 6×6 (6) | 0×0 | 5 | ok Optimal 5.0000000000e0, 0.000s, 0 nodes | ok Optimal 5.0000000000e0, 0.000s, 0 nodes | ok Optimal 5.0000000000e0, 0.000s, 0 nodes | presolved | 0.14 |
| family/family_eq-chain_n08-1 | 8×8 (8) | 0×0 | -19 | ok Optimal -1.9000000000e1, 0.000s, 0 nodes | ok Optimal -1.9000000000e1, 0.000s, 0 nodes | ok Optimal -1.9000000000e1, 0.000s, 0 nodes | presolved | 0.22 |
| family/family_eq-chain_n10-2 | 10×10 (10) | 0×0 | 21 | ok Optimal 2.1000000000e1, 0.000s, 0 nodes | ok Optimal 2.1000000000e1, 0.000s, 0 nodes | ok Optimal 2.1000000000e1, 0.000s, 0 nodes | presolved | 0.25 |
| family/family_eq-chain_n12-3 | 12×12 (12) | 0×0 | 18 | ok Optimal 1.8000000000e1, 0.000s, 0 nodes | ok Optimal 1.8000000000e1, 0.000s, 0 nodes | ok Optimal 1.8000000000e1, 0.000s, 0 nodes | presolved | 0.27 |
| family/family_fixed-charge_n05-0 | 6×10 (5) | 6×10 | 531 | ok Optimal 5.3100000000e2, 0.000s, 4 nodes | ok Optimal 5.3100000000e2, 0.000s, 4 nodes | ok Optimal 5.3100000000e2, 0.000s, 4 nodes | presolved | 0.67 |
| family/family_fixed-charge_n06-1 | 7×12 (6) | 7×12 | 467 | ok Optimal 4.6700000000e2, 0.000s, 8 nodes | ok Optimal 4.6700000000e2, 0.000s, 8 nodes | ok Optimal 4.6700000000e2, 0.000s, 8 nodes | presolved | 1.23 |
| family/family_fixed-charge_n07-2 | 8×14 (7) | 8×14 | 341 | ok Optimal 3.4100000000e2, 0.000s, 4 nodes | ok Optimal 3.4100000000e2, 0.000s, 4 nodes | ok Optimal 3.4100000000e2, 0.000s, 4 nodes | presolved | 0.78 |
| family/family_fixed-charge_n08-3 | 9×16 (8) | 9×16 | 331 | ok Optimal 3.3100000000e2, 0.000s, 8 nodes | ok Optimal 3.3100000000e2, 0.000s, 8 nodes | ok Optimal 3.3100000000e2, 0.000s, 8 nodes | presolved | 1.30 |
| family/family_fixed-charge_n09-4 | 10×18 (9) | 10×18 | 432 | ok Optimal 4.3200000000e2, 0.000s, 6 nodes | ok Optimal 4.3200000000e2, 0.000s, 6 nodes | ok Optimal 4.3200000000e2, 0.000s, 6 nodes | presolved | 1.39 |
| family/family_fixed-charge_n10-5 | 11×20 (10) | 11×20 | 542 | ok Optimal 5.4200000000e2, 0.000s, 4 nodes | ok Optimal 5.4200000000e2, 0.000s, 4 nodes | ok Optimal 5.4200000000e2, 0.000s, 4 nodes | presolved | 0.97 |
| family/family_gap_a2t7-3 | 9×14 (14) | 2×7 | 59 | ok Optimal 5.9000000000e1, 0.000s, 6 nodes | ok Optimal 5.9000000000e1, 0.000s, 6 nodes | ok Optimal 5.9000000000e1, 0.000s, 6 nodes | presolved | 0.58 |
| family/family_gap_a3t4-0 | 7×12 (12) | 5×10 | 20 | ok Optimal 2.0000000000e1, 0.000s, 6 nodes | ok Optimal 2.0000000000e1, 0.000s, 5 nodes | ok Optimal 2.0000000000e1, 0.000s, 5 nodes | presolved | 1.30 |
| family/family_gap_a3t5-1 | 8×15 (15) | 8×15 | 23 | ok Optimal 2.3000000000e1, 0.000s, 0 nodes | ok Optimal 2.3000000000e1, 0.000s, 0 nodes | ok Optimal 2.3000000000e1, 0.000s, 0 nodes | presolved | 0.80 |
| family/family_gap_a4t4-2 | 8×16 (16) | 8×16 | 18 | ok Optimal 1.8000000000e1, 0.000s, 0 nodes | ok Optimal 1.8000000000e1, 0.000s, 0 nodes | ok Optimal 1.8000000000e1, 0.000s, 0 nodes | presolved | 0.73 |
| family/family_infeasible_capacity | 1×4 (4) | 1×4 | infeasible | ok Infeasible -, 0.000s, 0 nodes | ok Infeasible -, 0.000s, 0 nodes | ok Infeasible -, 0.000s, 0 nodes | original: presolve infeasible (row 0 cannot be met within its bounds) | 0.61 |
| family/family_infeasible_chain | 3×2 (2) | 3×2 | infeasible | ok Infeasible -, 0.000s, 0 nodes | ok Infeasible -, 0.000s, 0 nodes | ok Infeasible -, 0.000s, 0 nodes | original: presolve infeasible (integer col 1 has no integer in its bounds) | 0.71 |
| family/family_infeasible_pack-vs-partition | 4×3 (3) | 4×3 | infeasible | ok Infeasible -, 0.000s, 0 nodes | ok Infeasible -, 0.000s, 0 nodes | ok Infeasible -, 0.000s, 0 nodes | original: reduced model: problem is infeasible | 2.64 |
| family/family_infeasible_parity | 1×2 (2) | 1×2 | infeasible | ok Infeasible -, 0.000s, 0 nodes | ok Infeasible -, 0.000s, 0 nodes | ok Infeasible -, 0.000s, 0 nodes | original: presolve infeasible (row 0 is empty and excludes 0) | 1.39 |
| family/family_set-cover_e08s10-0 | 8×10 (10) | 5×8 | 23 | ok Optimal 2.3000000000e1, 0.000s, 0 nodes | ok Optimal 2.3000000000e1, 0.000s, 0 nodes | ok Optimal 2.3000000000e1, 0.000s, 0 nodes | presolved | 0.81 |
| family/family_set-cover_e10s12-1 | 10×12 (12) | 8×11 | 30 | ok Optimal 3.0000000000e1, 0.000s, 0 nodes | ok Optimal 3.0000000000e1, 0.000s, 0 nodes | ok Optimal 3.0000000000e1, 0.000s, 0 nodes | presolved | 0.95 |
| family/family_set-cover_e12s14-2 | 12×14 (14) | 12×14 | 5 | ok Optimal 5.0000000000e0, 0.000s, 0 nodes | ok Optimal 5.0000000000e0, 0.000s, 0 nodes | ok Optimal 5.0000000000e0, 0.000s, 0 nodes | presolved | 0.84 |
| family/family_set-cover_e12s16-3 | 12×16 (16) | 12×16 | 8 | ok Optimal 8.0000000000e0, 0.000s, 0 nodes | ok Optimal 8.0000000000e0, 0.000s, 0 nodes | ok Optimal 8.0000000000e0, 0.000s, 0 nodes | presolved | 0.99 |
| family/family_set-cover_e14s16-4 | 14×16 (16) | 14×16 | 6 | ok Optimal 6.0000000000e0, 0.000s, 0 nodes | ok Optimal 6.0000000000e0, 0.000s, 0 nodes | ok Optimal 6.0000000000e0, 0.000s, 0 nodes | presolved | 1.24 |
| family/family_set-cover_e14s18-5 | 14×18 (18) | 14×18 | 38 | ok Optimal 3.8000000000e1, 0.000s, 0 nodes | ok Optimal 3.8000000000e1, 0.000s, 0 nodes | ok Optimal 3.8000000000e1, 0.000s, 0 nodes | presolved | 0.74 |
| family/family_set-pack_i10r06-0 | 6×10 (10) | 6×10 | 71 | ok Optimal 7.1000000000e1, 0.000s, 0 nodes | ok Optimal 7.1000000000e1, 0.000s, 0 nodes | ok Optimal 7.1000000000e1, 0.000s, 0 nodes | presolved | 0.86 |
| family/family_set-pack_i12r08-1 | 8×12 (12) | 8×12 | 52 | ok Optimal 5.2000000000e1, 0.000s, 0 nodes | ok Optimal 5.2000000000e1, 0.000s, 0 nodes | ok Optimal 5.2000000000e1, 0.000s, 0 nodes | presolved | 0.88 |
| family/family_set-pack_i14r09-2 | 9×14 (14) | 9×14 | 77 | ok Optimal 7.7000000000e1, 0.000s, 0 nodes | ok Optimal 7.7000000000e1, 0.000s, 0 nodes | ok Optimal 7.7000000000e1, 0.000s, 0 nodes | presolved | 0.90 |
| family/family_set-pack_i16r10-3 | 10×16 (16) | 10×16 | 65 | ok Optimal 6.5000000000e1, 0.000s, 0 nodes | ok Optimal 6.5000000000e1, 0.000s, 0 nodes | ok Optimal 6.5000000000e1, 0.000s, 0 nodes | presolved | 0.92 |
| family/family_set-pack_i16r12-4 | 12×16 (16) | 12×16 | 87 | ok Optimal 8.7000000000e1, 0.000s, 0 nodes | ok Optimal 8.7000000000e1, 0.000s, 0 nodes | ok Optimal 8.7000000000e1, 0.000s, 0 nodes | presolved | 0.89 |
| family/family_set-pack_i18r12-5 | 12×18 (18) | 12×18 | 73 | ok Optimal 7.3000000000e1, 0.000s, 0 nodes | ok Optimal 7.3000000000e1, 0.000s, 0 nodes | ok Optimal 7.3000000000e1, 0.000s, 0 nodes | presolved | 1.29 |
| family/family_set-partition_e08s12-0 | 8×12 (12) | 3×6 | 17 | ok Optimal 1.7000000000e1, 0.000s, 0 nodes | ok Optimal 1.7000000000e1, 0.000s, 0 nodes | ok Optimal 1.7000000000e1, 0.000s, 0 nodes | presolved | 0.64 |
| family/family_set-partition_e10s14-1 | 10×14 (14) | 9×13 | 48 | ok Optimal 4.8000000000e1, 0.000s, 2 nodes | ok Optimal 4.8000000000e1, 0.000s, 2 nodes | ok Optimal 4.8000000000e1, 0.000s, 2 nodes | presolved | 1.08 |
| family/family_set-partition_e10s16-2 | 10×16 (16) | 5×10 | 20 | ok Optimal 2.0000000000e1, 0.000s, 0 nodes | ok Optimal 2.0000000000e1, 0.000s, 0 nodes | ok Optimal 2.0000000000e1, 0.000s, 0 nodes | presolved | 1.26 |
| family/family_set-partition_e12s16-3 | 12×16 (16) | 0×2 | 60 | ok Optimal 6.0000000000e1, 0.000s, 2 nodes | ok Optimal 6.0000000000e1, 0.000s, 0 nodes | ok Optimal 6.0000000000e1, 0.000s, 0 nodes | presolved | 0.40 |
| family/family_transport_3x3-0 | 6×9 (9) | 6×9 | 21 | ok Optimal 2.1000000000e1, 0.000s, 0 nodes | ok Optimal 2.1000000000e1, 0.000s, 0 nodes | ok Optimal 2.1000000000e1, 0.000s, 0 nodes | presolved | 0.79 |
| family/family_transport_3x3-1 | 6×9 (9) | 6×9 | 48 | ok Optimal 4.8000000000e1, 0.000s, 0 nodes | ok Optimal 4.8000000000e1, 0.000s, 0 nodes | ok Optimal 4.8000000000e1, 0.000s, 0 nodes | presolved | 0.84 |
| family/family_transport_3x3-2 | 6×9 (9) | 0×2 | 29 | ok Optimal 2.9000000000e1, 0.000s, 0 nodes | ok Optimal 2.9000000000e1, 0.000s, 0 nodes | ok Optimal 2.9000000000e1, 0.000s, 0 nodes | presolved | 0.49 |
| family/family_transport_3x3-3 | 6×9 (9) | 2×3 | 28 | ok Optimal 2.8000000000e1, 0.000s, 0 nodes | ok Optimal 2.8000000000e1, 0.000s, 0 nodes | ok Optimal 2.8000000000e1, 0.000s, 0 nodes | presolved | 0.72 |
| family/family_unbounded_mixed-min | 1×2 (1) | 1×2 | unbounded | ok Unbounded -, 0.000s, 0 nodes | ok Unbounded -, 0.000s, 0 nodes | ok Unbounded -, 0.000s, 0 nodes | original: reduced model: problem is unbounded | 0.97 |
| family/family_unbounded_real-max | 1×2 (0) | 1×2 | unbounded | ok Unbounded -, 0.000s, 0 nodes | ok Unbounded -, 0.000s, 0 nodes | ok Unbounded -, 0.000s, 0 nodes | original: reduced model: problem is unbounded | 1.45 |
| family/milp_all-fixed-ints | 1×3 (3) | 0×0 | 9 | ok Optimal 9.0000000000e0, 0.000s, 0 nodes | ok Optimal 9.0000000000e0, 0.000s, 0 nodes | ok Optimal 9.0000000000e0, 0.000s, 0 nodes | presolved | 0.19 |
| family/milp_alternate-optima | 1×2 (2) | 1×2 | 1 | ok Optimal 1.0000000000e0, 0.000s, 0 nodes | ok Optimal 1.0000000000e0, 0.000s, 0 nodes | ok Optimal 1.0000000000e0, 0.000s, 0 nodes | presolved | 0.66 |
| family/milp_bigm-implication | 2×2 (1) | 0×1 | 107 | ok Optimal 1.0700000000e2, 0.000s, 2 nodes | ok Optimal 1.0700000000e2, 0.000s, 0 nodes | ok Optimal 1.0700000000e2, 0.000s, 0 nodes | presolved | 0.28 |
| family/milp_ceil-trap-min | 1×1 (1) | 0×1 | 3 | ok Optimal 3.0000000000e0, 0.000s, 0 nodes | ok Optimal 3.0000000000e0, 0.000s, 0 nodes | ok Optimal 3.0000000000e0, 0.000s, 0 nodes | presolved | 0.81 |
| family/milp_coin-change-1_3_4-6 | 1×3 (3) | 1×3 | 2 | ok Optimal 2.0000000000e0, 0.000s, 4 nodes | ok Optimal 2.0000000000e0, 0.000s, 4 nodes | ok Optimal 2.0000000000e0, 0.000s, 4 nodes | presolved | 0.62 |
| family/milp_coin-change-1_5_10_25-30 | 1×4 (4) | 1×4 | 2 | ok Optimal 2.0000000000e0, 0.000s, 1 nodes | ok Optimal 2.0000000000e0, 0.000s, 1 nodes | ok Optimal 2.0000000000e0, 0.000s, 1 nodes | presolved | 0.89 |
| family/milp_coin-change-1_5_10_25-68 | 1×4 (4) | 1×4 | 7 | ok Optimal 7.0000000000e0, 0.000s, 24 nodes | ok Optimal 7.0000000000e0, 0.000s, 24 nodes | ok Optimal 7.0000000000e0, 0.000s, 24 nodes | presolved | 0.86 |
| family/milp_coin-change-1_5_10_25-99 | 1×4 (4) | 1×4 | 9 | ok Optimal 9.0000000000e0, 0.000s, 30 nodes | ok Optimal 9.0000000000e0, 0.000s, 30 nodes | ok Optimal 9.0000000000e0, 0.000s, 30 nodes | presolved | 0.96 |
| family/milp_diophantine-min | 1×2 (2) | 1×2 | -3 | ok Optimal -3.0000000000e0, 0.000s, 0 nodes | ok Optimal -3.0000000000e0, 0.000s, 0 nodes | ok Optimal -3.0000000000e0, 0.000s, 0 nodes | presolved | 0.78 |
| family/milp_fixed-infeasible | 1×2 (2) | 1×2 | infeasible | ok Infeasible -, 0.000s, 0 nodes | ok Infeasible -, 0.000s, 0 nodes | ok Infeasible -, 0.000s, 0 nodes | original: presolve infeasible (row 0 cannot be met within its bounds) | 0.75 |
| family/milp_floor-trap-max | 1×1 (1) | 0×1 | 4 | ok Optimal 4.0000000000e0, 0.000s, 0 nodes | ok Optimal 4.0000000000e0, 0.000s, 0 nodes | ok Optimal 4.0000000000e0, 0.000s, 0 nodes | presolved | 0.66 |
| family/milp_fractional-cut-needed | 2×2 (2) | 1×2 | 2 | ok Optimal 2.0000000000e0, 0.000s, 0 nodes | ok Optimal 2.0000000000e0, 0.000s, 0 nodes | ok Optimal 2.0000000000e0, 0.000s, 4 nodes | presolved | 0.74 |
| family/milp_huge-int-bound | 0×1 (1) | 0×1 | 2147483647 | ok Optimal 2.1474836470e9, 0.000s, 0 nodes | ok Optimal 2.1474836470e9, 0.000s, 0 nodes | ok Optimal 2.1474836470e9, 0.000s, 0 nodes | presolved | 0.71 |
| family/milp_int-eq-forcing | 1×2 (2) | 1×2 | 3 | ok Optimal 3.0000000000e0, 0.000s, 0 nodes | ok Optimal 3.0000000000e0, 0.000s, 0 nodes | ok Optimal 3.0000000000e0, 0.000s, 0 nodes | presolved | 0.93 |
| family/milp_interval-infeasible | 2×1 (1) | 2×1 | infeasible | ok Infeasible -, 0.000s, 0 nodes | ok Infeasible -, 0.000s, 0 nodes | ok Infeasible -, 0.000s, 0 nodes | original: presolve infeasible (integer col 0 has no integer in its bounds) | 0.32 |
| family/milp_knapsack-scaled | 1×3 (3) | 1×3 | 400000 | ok Optimal 4.0000018000e5, 0.000s, 0 nodes | ok Optimal 4.0000018000e5, 0.000s, 0 nodes | ok Optimal 4.0000000000e5, 0.000s, 4 nodes | presolved | 0.68 |
| family/milp_knapsack-tight-gap | 1×3 (3) | 1×3 | 4 | ok Optimal 4.0000018000e0, 0.000s, 0 nodes | ok Optimal 4.0000018000e0, 0.000s, 0 nodes | ok Optimal 4.0000000000e0, 0.000s, 4 nodes | presolved | 0.96 |
| family/milp_magic-square-center-max | 26×81 (81) | 26×81 | 5 | ok Optimal 5.0000000000e0, 0.002s, 8 nodes | ok Optimal 5.0000000000e0, 0.002s, 8 nodes | ok Optimal 5.0000000000e0, 0.002s, 8 nodes | presolved | 1.03 |
| family/milp_magic-square-center-min | 26×81 (81) | 26×81 | 5 | ok Optimal 5.0000000000e0, 0.002s, 11 nodes | ok Optimal 5.0000000000e0, 0.003s, 11 nodes | ok Optimal 5.0000000000e0, 0.004s, 11 nodes | presolved | 1.20 |
| family/milp_mixed-textbook | 2×3 (1) | 2×3 | 405 | ok Optimal 4.0500000000e2, 0.000s, 2 nodes | ok Optimal 4.0500000000e2, 0.000s, 2 nodes | ok Optimal 4.0500000000e2, 0.000s, 2 nodes | presolved | 0.62 |
| family/milp_negative-floor-trap | 1×1 (1) | 0×1 | -3 | ok Optimal -3.0000000000e0, 0.000s, 0 nodes | ok Optimal -3.0000000000e0, 0.000s, 0 nodes | ok Optimal -3.0000000000e0, 0.000s, 0 nodes | presolved | 1.20 |
| family/milp_negative-range-min | 1×1 (1) | 0×1 | -4 | ok Optimal -4.0000000000e0, 0.000s, 0 nodes | ok Optimal -4.0000000000e0, 0.000s, 0 nodes | ok Optimal -4.0000000000e0, 0.000s, 0 nodes | presolved | 0.74 |
| family/milp_parity-infeasible | 1×2 (2) | 1×2 | infeasible | ok Infeasible -, 0.000s, 0 nodes | ok Infeasible -, 0.000s, 0 nodes | ok Infeasible -, 0.000s, 0 nodes | original: presolve infeasible (integer col 0 has no integer in its bounds) | 0.71 |
| family/milp_production-planning | 8×13 (4) | 6×10 | 3440 | ok Optimal 3.4400000000e3, 0.000s, 4 nodes | ok Optimal 3.4400000000e3, 0.000s, 4 nodes | ok Optimal 3.4400000000e3, 0.000s, 4 nodes | presolved | 0.84 |
| family/milp_queens-5 | 24×25 (25) | 24×25 | 5 | ok Optimal 5.0000000000e0, 0.000s, 3 nodes | ok Optimal 5.0000000000e0, 0.000s, 3 nodes | ok Optimal 5.0000000000e0, 0.000s, 3 nodes | presolved | 0.99 |
| family/milp_queens-6 | 30×36 (36) | 30×36 | 6 | ok Optimal 6.0000000000e0, 0.002s, 19 nodes | ok Optimal 6.0000000000e0, 0.002s, 19 nodes | ok Optimal 6.0000000000e0, 0.002s, 19 nodes | presolved | 0.98 |
| family/milp_readme-example | 2×2 (1) | 2×2 | 7 | ok Optimal 7.0000000000e0, 0.000s, 0 nodes | ok Optimal 7.0000000000e0, 0.000s, 0 nodes | ok Optimal 7.0000000000e0, 0.000s, 0 nodes | presolved | 0.54 |
| family/milp_unbounded-mixed | 1×2 (1) | 0×2 | unbounded | ok Unbounded -, 0.000s, 0 nodes | ok Unbounded -, 0.000s, 0 nodes | ok Unbounded -, 0.000s, 0 nodes | original: reduced model: problem is unbounded | 0.96 |
| family/oracle_assignment_n5-s00 | 10×25 (25) | 10×25 | 68 | ok Optimal 6.8000000000e1, 0.000s, 0 nodes | ok Optimal 6.8000000000e1, 0.000s, 0 nodes | ok Optimal 6.8000000000e1, 0.000s, 0 nodes | presolved | 0.89 |
| family/oracle_assignment_n5-s01 | 10×25 (25) | 10×25 | 56 | ok Optimal 5.6000000000e1, 0.000s, 0 nodes | ok Optimal 5.6000000000e1, 0.000s, 0 nodes | ok Optimal 5.6000000000e1, 0.000s, 0 nodes | presolved | 1.04 |
| family/oracle_assignment_n6-s02 | 12×36 (36) | 12×36 | 34 | ok Optimal 3.4000000000e1, 0.000s, 0 nodes | ok Optimal 3.4000000000e1, 0.000s, 0 nodes | ok Optimal 3.4000000000e1, 0.000s, 0 nodes | presolved | 1.23 |
| family/oracle_assignment_n6-s03 | 12×36 (36) | 12×36 | 80 | ok Optimal 8.0000000000e1, 0.000s, 0 nodes | ok Optimal 8.0000000000e1, 0.000s, 0 nodes | ok Optimal 8.0000000000e1, 0.000s, 0 nodes | presolved | 0.89 |
| family/oracle_assignment_n7-s04 | 14×49 (49) | 14×49 | 49 | ok Optimal 4.9000000000e1, 0.000s, 0 nodes | ok Optimal 4.9000000000e1, 0.000s, 0 nodes | ok Optimal 4.9000000000e1, 0.000s, 0 nodes | presolved | 1.09 |
| family/oracle_assignment_n7-s05 | 14×49 (49) | 14×49 | 53 | ok Optimal 5.3000000000e1, 0.000s, 0 nodes | ok Optimal 5.3000000000e1, 0.000s, 0 nodes | ok Optimal 5.3000000000e1, 0.000s, 0 nodes | presolved | 1.29 |
| family/oracle_ilp-box_s00 | 3×4 (4) | 3×4 | 4 | ok Optimal 4.0000000000e0, 0.000s, 3 nodes | ok Optimal 4.0000000000e0, 0.000s, 3 nodes | ok Optimal 4.0000000000e0, 0.000s, 3 nodes | presolved | 0.50 |
| family/oracle_ilp-box_s01 | 5×6 (6) | 5×6 | infeasible | ok Infeasible -, 0.000s, 0 nodes | ok Infeasible -, 0.000s, 0 nodes | ok Infeasible -, 0.000s, 0 nodes | original: presolve infeasible (row 3 cannot be met within its bounds) | 0.72 |
| family/oracle_ilp-box_s02 | 5×6 (6) | 5×6 | infeasible | ok Infeasible -, 0.000s, 0 nodes | ok Infeasible -, 0.000s, 0 nodes | ok Infeasible -, 0.000s, 0 nodes | original: presolve infeasible (row 2 cannot be met within its bounds) | 0.98 |
| family/oracle_ilp-box_s03 | 4×6 (6) | 4×6 | infeasible | ok Infeasible -, 0.000s, 0 nodes | ok Infeasible -, 0.000s, 0 nodes | ok Infeasible -, 0.000s, 0 nodes | original: presolve infeasible (row 3 cannot be met within its bounds) | 0.76 |
| family/oracle_ilp-box_s04 | 3×6 (6) | 3×6 | 47 | ok Optimal 4.7000000000e1, 0.000s, 8 nodes | ok Optimal 4.7000000000e1, 0.000s, 8 nodes | ok Optimal 4.7000000000e1, 0.000s, 8 nodes | presolved | 0.77 |
| family/oracle_ilp-box_s05 | 4×6 (6) | 4×6 | infeasible | ok Infeasible -, 0.000s, 0 nodes | ok Infeasible -, 0.000s, 0 nodes | ok Infeasible -, 0.000s, 0 nodes | original: presolve infeasible (row 3 cannot be met within its bounds) | 0.80 |
| family/oracle_ilp-box_s06 | 3×6 (6) | 3×6 | -13 | ok Optimal -1.3000000000e1, 0.000s, 8 nodes | ok Optimal -1.3000000000e1, 0.000s, 8 nodes | ok Optimal -1.3000000000e1, 0.000s, 8 nodes | presolved | 0.74 |
| family/oracle_ilp-box_s07 | 4×5 (5) | 0×1 | -21 | ok Optimal -2.1000000000e1, 0.000s, 0 nodes | ok Optimal -2.1000000000e1, 0.000s, 0 nodes | ok Optimal -2.1000000000e1, 0.000s, 0 nodes | presolved | 0.69 |
| family/oracle_ilp-box_s08 | 5×7 (7) | 3×5 | -32 | ok Optimal -3.2000000000e1, 0.000s, 0 nodes | ok Optimal -3.2000000000e1, 0.000s, 0 nodes | ok Optimal -3.2000000000e1, 0.000s, 0 nodes | presolved | 0.93 |
| family/oracle_ilp-box_s09 | 3×6 (6) | 2×6 | -73 | ok Optimal -7.3000000000e1, 0.000s, 0 nodes | ok Optimal -7.3000000000e1, 0.000s, 0 nodes | ok Optimal -7.3000000000e1, 0.000s, 0 nodes | presolved | 0.54 |
| family/oracle_ilp-box_s10 | 3×7 (7) | 3×5 | 42 | ok Optimal 4.2000000000e1, 0.000s, 2 nodes | ok Optimal 4.2000000000e1, 0.000s, 2 nodes | ok Optimal 4.2000000000e1, 0.000s, 2 nodes | presolved | 0.90 |
| family/oracle_ilp-box_s11 | 4×6 (6) | 2×6 | -15 | ok Optimal -1.5000000000e1, 0.000s, 4 nodes | ok Optimal -1.5000000000e1, 0.000s, 4 nodes | ok Optimal -1.5000000000e1, 0.000s, 4 nodes | presolved | 0.83 |
| family/oracle_ilp-box_s12 | 4×7 (7) | 4×7 | -23 | ok Optimal -2.3000000000e1, 0.000s, 16 nodes | ok Optimal -2.3000000000e1, 0.000s, 16 nodes | ok Optimal -2.3000000000e1, 0.000s, 16 nodes | presolved | 0.92 |
| family/oracle_ilp-box_s13 | 4×7 (7) | 4×7 | -58 | ok Optimal -5.8000000000e1, 0.000s, 10 nodes | ok Optimal -5.8000000000e1, 0.000s, 10 nodes | ok Optimal -5.8000000000e1, 0.000s, 10 nodes | presolved | 0.74 |
| family/oracle_ilp-box_s14 | 5×5 (5) | 5×5 | infeasible | ok Infeasible -, 0.000s, 0 nodes | ok Infeasible -, 0.000s, 0 nodes | ok Infeasible -, 0.000s, 0 nodes | original: presolve infeasible (integer col 0 has no integer in its bounds) | 0.82 |
| family/oracle_ilp-box_s15 | 4×4 (4) | 1×4 | -54 | ok Optimal -5.4000000000e1, 0.000s, 0 nodes | ok Optimal -5.4000000000e1, 0.000s, 0 nodes | ok Optimal -5.4000000000e1, 0.000s, 0 nodes | presolved | 0.79 |
| family/oracle_ilp-box_s16 | 4×5 (5) | 3×5 | 0 | ok Optimal 0.0000000000e0, 0.000s, 11 nodes | ok Optimal 0.0000000000e0, 0.000s, 11 nodes | ok Optimal 0.0000000000e0, 0.000s, 11 nodes | presolved | 0.83 |
| family/oracle_ilp-box_s17 | 3×7 (7) | 3×7 | -53 | ok Optimal -5.3000000000e1, 0.000s, 10 nodes | ok Optimal -5.3000000000e1, 0.000s, 10 nodes | ok Optimal -5.3000000000e1, 0.000s, 10 nodes | presolved | 1.11 |
| family/oracle_ilp-box_s18 | 5×4 (4) | 5×4 | infeasible | ok Infeasible -, 0.000s, 0 nodes | ok Infeasible -, 0.000s, 0 nodes | ok Infeasible -, 0.000s, 0 nodes | original: presolve infeasible (row 3 cannot be met within its bounds) | 0.76 |
| family/oracle_ilp-box_s19 | 5×7 (7) | 5×7 | -29 | ok Optimal -2.9000000000e1, 0.000s, 0 nodes | ok Optimal -2.9000000000e1, 0.000s, 0 nodes | ok Optimal -2.9000000000e1, 0.000s, 0 nodes | presolved | 0.50 |
| family/oracle_knap-bounded_n06-s00 | 1×6 (6) | 1×6 | 91 | ok Optimal 9.1000000000e1, 0.000s, 16 nodes | ok Optimal 9.1000000000e1, 0.000s, 16 nodes | ok Optimal 9.1000000000e1, 0.000s, 16 nodes | presolved | 0.79 |
| family/oracle_knap-bounded_n07-s01 | 1×7 (7) | 1×7 | 137 | ok Optimal 1.3700000000e2, 0.000s, 5 nodes | ok Optimal 1.3700000000e2, 0.000s, 5 nodes | ok Optimal 1.3700000000e2, 0.000s, 5 nodes | presolved | 0.74 |
| family/oracle_knap-bounded_n08-s02 | 1×8 (8) | 1×8 | 173 | ok Optimal 1.7300000000e2, 0.000s, 6 nodes | ok Optimal 1.7300000000e2, 0.000s, 6 nodes | ok Optimal 1.7300000000e2, 0.000s, 6 nodes | presolved | 1.43 |
| family/oracle_knap-bounded_n08-s03 | 1×8 (8) | 1×8 | 134 | ok Optimal 1.3400000000e2, 0.000s, 5 nodes | ok Optimal 1.3400000000e2, 0.000s, 5 nodes | ok Optimal 1.3400000000e2, 0.000s, 5 nodes | presolved | 0.85 |
| family/oracle_knap-bounded_n09-s04 | 1×9 (9) | 1×9 | 229 | ok Optimal 2.2900000000e2, 0.000s, 3 nodes | ok Optimal 2.2900000000e2, 0.000s, 3 nodes | ok Optimal 2.2900000000e2, 0.000s, 3 nodes | presolved | 0.89 |
| family/oracle_knap-bounded_n10-s05 | 1×10 (10) | 1×10 | 294 | ok Optimal 2.9400000000e2, 0.000s, 23 nodes | ok Optimal 2.9400000000e2, 0.000s, 23 nodes | ok Optimal 2.9400000000e2, 0.000s, 23 nodes | presolved | 0.90 |
| family/oracle_knap-bounded_n10-s06 | 1×10 (10) | 1×10 | 169 | ok Optimal 1.6900000000e2, 0.000s, 0 nodes | ok Optimal 1.6900000000e2, 0.000s, 0 nodes | ok Optimal 1.6900000000e2, 0.000s, 0 nodes | presolved | 0.82 |
| family/oracle_knap-bounded_n12-s07 | 1×12 (12) | 1×12 | 235 | ok Optimal 2.3500000000e2, 0.000s, 21 nodes | ok Optimal 2.3500000000e2, 0.000s, 21 nodes | ok Optimal 2.3500000000e2, 0.000s, 21 nodes | presolved | 0.35 |
| family/oracle_knap01-ties_n12-s12 | 1×12 (12) | 1×12 | 60 | ok Optimal 6.0000000000e1, 0.000s, 11 nodes | ok Optimal 6.0000000000e1, 0.000s, 11 nodes | ok Optimal 6.0000000000e1, 0.000s, 11 nodes | presolved | 0.91 |
| family/oracle_knap01-ties_n16-s13 | 1×16 (16) | 1×16 | 100 | ok Optimal 1.0000000000e2, 0.000s, 7 nodes | ok Optimal 1.0000000000e2, 0.000s, 7 nodes | ok Optimal 1.0000000000e2, 0.000s, 7 nodes | presolved | 0.92 |
| family/oracle_knap01-ties_n20-s14 | 1×20 (20) | 1×20 | 140 | ok Optimal 1.4000000000e2, 0.000s, 14 nodes | ok Optimal 1.4000000000e2, 0.000s, 14 nodes | ok Optimal 1.4000000000e2, 0.000s, 14 nodes | presolved | 0.92 |
| family/oracle_knap01-ties_n24-s15 | 1×24 (24) | 1×24 | 170 | ok Optimal 1.7000000000e2, 0.000s, 7 nodes | ok Optimal 1.7000000000e2, 0.000s, 7 nodes | ok Optimal 1.7000000000e2, 0.000s, 7 nodes | presolved | 0.98 |
| family/oracle_knap01_n10-s00 | 1×10 (10) | 1×10 | 166 | ok Optimal 1.6600000000e2, 0.000s, 15 nodes | ok Optimal 1.6600000000e2, 0.000s, 15 nodes | ok Optimal 1.6600000000e2, 0.000s, 15 nodes | presolved | 1.11 |
| family/oracle_knap01_n12-s01 | 1×12 (12) | 1×12 | 171 | ok Optimal 1.7100000000e2, 0.000s, 53 nodes | ok Optimal 1.7100000000e2, 0.000s, 53 nodes | ok Optimal 1.7100000000e2, 0.000s, 53 nodes | presolved | 0.87 |
| family/oracle_knap01_n14-s02 | 1×14 (14) | 1×14 | 196 | ok Optimal 1.9600000000e2, 0.000s, 53 nodes | ok Optimal 1.9600000000e2, 0.000s, 53 nodes | ok Optimal 1.9600000000e2, 0.000s, 53 nodes | presolved | 0.91 |
| family/oracle_knap01_n16-s03 | 1×16 (16) | 1×16 | 179 | ok Optimal 1.7900000000e2, 0.000s, 43 nodes | ok Optimal 1.7900000000e2, 0.000s, 43 nodes | ok Optimal 1.7900000000e2, 0.000s, 43 nodes | presolved | 0.80 |
| family/oracle_knap01_n16-s04 | 1×16 (16) | 1×16 | 257 | ok Optimal 2.5700000000e2, 0.000s, 13 nodes | ok Optimal 2.5700000000e2, 0.000s, 13 nodes | ok Optimal 2.5700000000e2, 0.000s, 13 nodes | presolved | 0.92 |
| family/oracle_knap01_n18-s05 | 1×18 (18) | 1×18 | 300 | ok Optimal 3.0000000000e2, 0.000s, 13 nodes | ok Optimal 3.0000000000e2, 0.000s, 13 nodes | ok Optimal 3.0000000000e2, 0.000s, 13 nodes | presolved | 0.87 |
| family/oracle_knap01_n18-s06 | 1×18 (18) | 1×18 | 313 | ok Optimal 3.1300000000e2, 0.000s, 11 nodes | ok Optimal 3.1300000000e2, 0.000s, 11 nodes | ok Optimal 3.1300000000e2, 0.000s, 11 nodes | presolved | 0.91 |
| family/oracle_knap01_n20-s07 | 1×20 (20) | 1×20 | 290 | ok Optimal 2.9000000000e2, 0.000s, 19 nodes | ok Optimal 2.9000000000e2, 0.000s, 19 nodes | ok Optimal 2.9000000000e2, 0.000s, 19 nodes | presolved | 1.08 |
| family/oracle_knap01_n20-s08 | 1×20 (20) | 1×20 | 316 | ok Optimal 3.1600000000e2, 0.000s, 51 nodes | ok Optimal 3.1600000000e2, 0.000s, 51 nodes | ok Optimal 3.1600000000e2, 0.000s, 51 nodes | presolved | 0.88 |
| family/oracle_knap01_n22-s09 | 1×22 (22) | 1×22 | 364 | ok Optimal 3.6400000000e2, 0.001s, 96 nodes | ok Optimal 3.6400000000e2, 0.001s, 96 nodes | ok Optimal 3.6400000000e2, 0.001s, 96 nodes | presolved | 1.03 |
| family/oracle_knap01_n22-s10 | 1×22 (22) | 1×22 | 373 | ok Optimal 3.7300000000e2, 0.000s, 26 nodes | ok Optimal 3.7300000000e2, 0.000s, 26 nodes | ok Optimal 3.7300000000e2, 0.000s, 26 nodes | presolved | 0.66 |
| family/oracle_knap01_n24-s11 | 1×24 (24) | 1×24 | 373 | ok Optimal 3.7300000000e2, 0.000s, 53 nodes | ok Optimal 3.7300000000e2, 0.000s, 53 nodes | ok Optimal 3.7300000000e2, 0.000s, 53 nodes | presolved | 0.79 |
| family/oracle_mixed_s00 | 1×6 (5) | 0×6 | 0 | ok Optimal 0.0000000000e0, 0.000s, 0 nodes | ok Optimal 0.0000000000e0, 0.000s, 0 nodes | ok Optimal 0.0000000000e0, 0.000s, 0 nodes | presolved | 0.62 |
| family/oracle_mixed_s01 | 3×7 (4) | 3×7 | -12 | ok Optimal -1.2000000000e1, 0.000s, 2 nodes | ok Optimal -1.2000000000e1, 0.000s, 2 nodes | ok Optimal -1.2000000000e1, 0.000s, 2 nodes | presolved | 0.76 |
| family/oracle_mixed_s02 | 2×6 (4) | 2×6 | -4 | ok Optimal -4.0000000000e0, 0.000s, 0 nodes | ok Optimal -4.0000000000e0, 0.000s, 0 nodes | ok Optimal -4.0000000000e0, 0.000s, 2 nodes | presolved | 0.86 |
| family/oracle_mixed_s03 | 2×5 (3) | 2×5 | -30 | ok Optimal -3.0000000000e1, 0.000s, 0 nodes | ok Optimal -3.0000000000e1, 0.000s, 0 nodes | ok Optimal -3.0000000000e1, 0.000s, 0 nodes | presolved | 0.82 |
| family/oracle_mixed_s04 | 2×5 (3) | 2×5 | -19 | ok Optimal -1.9000000000e1, 0.000s, 0 nodes | ok Optimal -1.9000000000e1, 0.000s, 0 nodes | ok Optimal -1.9000000000e1, 0.000s, 0 nodes | presolved | 0.89 |
| family/oracle_mixed_s05 | 3×7 (4) | 3×7 | 1 | ok Optimal 1.0000000000e0, 0.000s, 4 nodes | ok Optimal 1.0000000000e0, 0.000s, 4 nodes | ok Optimal 1.0000000000e0, 0.000s, 4 nodes | presolved | 0.73 |
| family/oracle_mixed_s06 | 3×7 (4) | 3×7 | -7 | ok Optimal -7.0000000000e0, 0.000s, 6 nodes | ok Optimal -7.0000000000e0, 0.000s, 6 nodes | ok Optimal -7.0000000000e0, 0.000s, 6 nodes | presolved | 0.87 |
| family/oracle_mixed_s07 | 1×6 (5) | 1×6 | -17 | ok Optimal -1.7000000000e1, 0.000s, 1 nodes | ok Optimal -1.7000000000e1, 0.000s, 1 nodes | ok Optimal -1.7000000000e1, 0.000s, 1 nodes | presolved | 0.70 |
| family/oracle_subset-sum-infeasible_n10-s00 | 1×10 (10) | 1×4 | infeasible | ok Infeasible -, 0.000s, 0 nodes | ok Infeasible -, 0.000s, 0 nodes | ok Infeasible -, 0.000s, 0 nodes | original: reduced model: problem is infeasible | 1.60 |
| family/oracle_subset-sum-infeasible_n14-s01 | 1×14 (14) | 1×13 | infeasible | ok Infeasible -, 0.003s, 0 nodes | ok Infeasible -, 0.004s, 0 nodes | ok Infeasible -, 0.005s, 0 nodes | original: reduced model: problem is infeasible | 1.71 |
| family/oracle_subset-sum-infeasible_n18-s02 | 1×18 (18) | 1×14 | infeasible | ok Infeasible -, 0.002s, 0 nodes | ok Infeasible -, 0.003s, 0 nodes | ok Infeasible -, 0.003s, 0 nodes | original: reduced model: problem is infeasible | 1.84 |
| family/oracle_subset-sum_n14-s00 | 1×14 (14) | 1×14 | 146 | ok Optimal 1.4600000000e2, 0.001s, 103 nodes | ok Optimal 1.4600000000e2, 0.001s, 103 nodes | ok Optimal 1.4600000000e2, 0.001s, 103 nodes | presolved | 1.08 |
| family/oracle_subset-sum_n16-s01 | 1×16 (16) | 1×16 | 304 | ok Optimal 3.0400000000e2, 0.001s, 283 nodes | ok Optimal 3.0400000000e2, 0.001s, 283 nodes | ok Optimal 3.0400000000e2, 0.001s, 283 nodes | presolved | 0.95 |
| family/oracle_subset-sum_n18-s02 | 1×18 (18) | 1×18 | 236 | ok Optimal 2.3600000000e2, 0.000s, 34 nodes | ok Optimal 2.3600000000e2, 0.000s, 34 nodes | ok Optimal 2.3600000000e2, 0.000s, 34 nodes | presolved | 0.90 |
| family/oracle_subset-sum_n20-s03 | 1×20 (20) | 1×20 | 298 | ok Optimal 2.9800000000e2, 0.001s, 82 nodes | ok Optimal 2.9800000000e2, 0.001s, 82 nodes | ok Optimal 2.9800000000e2, 0.001s, 82 nodes | presolved | 0.87 |
| family/oracle_subset-sum_n22-s04 | 1×22 (22) | 1×22 | 367 | ok Optimal 3.6700000000e2, 0.000s, 32 nodes | ok Optimal 3.6700000000e2, 0.000s, 32 nodes | ok Optimal 3.6700000000e2, 0.000s, 32 nodes | presolved | 1.06 |

