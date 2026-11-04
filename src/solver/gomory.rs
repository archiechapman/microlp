//! Gomory mixed-integer cuts from the optimal tableau ([`Solver::gomory_cuts`])
//! and the rows that hold them ([`Solver::add_cut`]).
//!
//! A cut is implied by the model's rows, bounds and integrality, and is only
//! there to tighten the relaxation: the validation of a solution
//! ([`Solver::first_violated_row`]) skips cut rows, so the round-off of a cut
//! can never reject a solution of the model as written.

use super::{Solver, StopReason};
use crate::{ComparisonOp, CsVec, Error, VarDomain};

/// A row is used only where its basic integer var is at least this far from
/// an integer: closer rows give weak cuts and large coefficients.
const AWAY: f64 = 0.01;
/// Tableau entries below this (engine units) are taken as zero.
const ZERO: f64 = 1e-11;
/// A cut coefficient below this fraction of the cut's largest is removed,
/// the cut's right-hand side lowered by the most the term can contribute.
const REL_ZERO: f64 = 1e-9;
/// The largest ratio allowed between a cut's largest and smallest
/// coefficient: beyond it, its round-off can matter.
const MAX_DYNAMISM: f64 = 1e6;
/// The smallest violation of the current point, per unit of the cut's
/// Euclidean norm, for a cut to be worth a row.
const MIN_EFFICACY: f64 = 1e-5;
/// Each cut's right-hand side is lowered by this fraction of its magnitude
/// (`|rhs| + Σ|c_k|·max(|lo_k|, |hi_k|)`, at least 1), against the round-off
/// of the tableau it was read from.
const SAFETY: f64 = 1e-9;

impl Solver {
    /// At most `max_cuts` Gomory mixed-integer cuts at the current optimum,
    /// each `(c, b)` meaning `c·x ≥ b` over the structural vars in user units.
    /// They are valid for every point within the current bounds whose integer
    /// vars are integral: at the root of a search, for the whole search.
    ///
    /// For a basic integer var with a value `v` at least [`AWAY`] from an
    /// integer, its tableau row is written in the non-basic vars' distances
    /// from the bounds they sit at, `t_j ≥ 0`: `x_B + Σ c_j t_j = v`. With
    /// `f0 = frac(v)`, every such point has `Σ g_j t_j ≥ 1`, where `g_j` is
    /// `f_j / f0` or `(1 − f_j) / (1 − f0)`, whichever is smaller, for an
    /// integer `t_j` (`f_j = frac(c_j)`: an integer var at an integral bound),
    /// and `c_j / f0` or `−c_j / (1 − f0)` for a continuous one (Gomory 1960;
    /// Cornuéjols, *Valid inequalities for mixed integer linear programs*,
    /// 2008). The current vertex (every `t_j = 0`) violates it. A slack's
    /// distance is replaced by its row, giving the cut over the structural
    /// vars. A row with a non-basic var strictly between its bounds has no
    /// distance for it and is skipped; so is a cut whose coefficients span
    /// more than [`MAX_DYNAMISM`] or whose efficacy is below
    /// [`MIN_EFFICACY`]. Most fractional rows are tried first.
    pub(crate) fn gomory_cuts(&mut self, max_cuts: usize) -> Vec<(CsVec, f64)> {
        let n = self.num_vars;
        let is_int = |solver: &Solver, v: usize| {
            v < n
                && matches!(
                    solver.orig_var_domains[v],
                    VarDomain::Integer | VarDomain::Boolean
                )
        };
        let mut rows: Vec<(f64, usize)> = Vec::new();
        for (r, &var) in self.basic_vars.iter().enumerate() {
            if !is_int(self, var) {
                continue;
            }
            let x = self.get_value(var);
            let f0 = x - x.floor();
            if (AWAY..=1.0 - AWAY).contains(&f0) {
                rows.push(((f0 - 0.5).abs(), r));
            }
        }
        rows.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));

        // User-unit bounds and values of the structural vars.
        let bounds: Vec<(f64, f64)> = (0..n)
            .map(|v| {
                let (lo, hi) = self.model_bounds(v);
                (lo * self.col_scales[v], hi * self.col_scales[v])
            })
            .collect();
        let point: Vec<f64> = (0..n).map(|v| self.get_value(v)).collect();

        let mut coef = vec![0.0; n];
        let mut touched: Vec<usize> = Vec::new();
        let mut cuts = Vec::new();
        for &(_, r) in &rows {
            if cuts.len() >= max_cuts {
                break;
            }
            let basic = self.basic_vars[r];
            let s_b = self.col_scales[basic];
            let v = self.get_value(basic);
            let f0 = v - v.floor();
            self.calc_row_coeffs(r);
            let entries: Vec<(usize, f64)> = self.row_coeffs.iter().map(|(c, &a)| (c, a)).collect();
            for &k in &touched {
                coef[k] = 0.0;
            }
            touched.clear();
            let mut add = |k: usize, c: f64, coef: &mut [f64]| {
                if coef[k] == 0.0 {
                    touched.push(k);
                }
                coef[k] += c;
            };

            let mut rhs = 1.0;
            let mut usable = true;
            for (c, alpha) in entries {
                if alpha.abs() < ZERO {
                    continue;
                }
                let var = self.nb_vars[c];
                let val = self.nb_var_vals[c];
                let (lo, hi) = self.model_bounds(var);
                if lo == hi {
                    continue; // fixed: its distance is identically zero
                }
                let at_lower = val == lo;
                if !at_lower && val != hi {
                    usable = false;
                    break;
                }
                let a = if at_lower { alpha } else { -alpha };
                // The coefficient of the distance in user units (a slack's
                // distance stays in engine units: it is continuous).
                let c_j = if var < n {
                    s_b * a / self.col_scales[var]
                } else {
                    s_b * a
                };
                let integral_distance = is_int(self, var) && {
                    let bound = if at_lower {
                        bounds[var].0
                    } else {
                        bounds[var].1
                    };
                    bound == bound.round()
                };
                let g = if integral_distance {
                    let f = c_j - c_j.floor();
                    if f <= f0 {
                        f / f0
                    } else {
                        (1.0 - f) / (1.0 - f0)
                    }
                } else if c_j >= 0.0 {
                    c_j / f0
                } else {
                    -c_j / (1.0 - f0)
                };
                if g == 0.0 {
                    continue;
                }
                if var < n {
                    // t = x − lo, or hi − x.
                    if at_lower {
                        add(var, g, &mut coef);
                        rhs += g * bounds[var].0;
                    } else {
                        add(var, -g, &mut coef);
                        rhs -= g * bounds[var].1;
                    }
                } else {
                    // The slack of engine row `row` is `b' − Σ (a'_k / s_k) x_k`
                    // (`a'`, `b'` the row in engine units), and its distance
                    // `σ (s − bound)`, `σ = 1` at its lower bound, `−1` at its
                    // upper one.
                    let row = var - n;
                    let (sigma, bound) = if at_lower { (1.0, lo) } else { (-1.0, hi) };
                    rhs -= g * sigma * (self.orig_rhs[row] - bound);
                    let terms: Vec<(usize, f64)> = self
                        .orig_constraints
                        .outer_view(row)
                        .unwrap()
                        .iter()
                        .filter(|&(k, _)| k < n)
                        .map(|(k, &a)| (k, a / self.col_scales[k]))
                        .collect();
                    for (k, a) in terms {
                        add(k, -g * sigma * a, &mut coef);
                    }
                }
            }
            if !usable {
                continue;
            }
            if let Some(cut) = finish_cut(&coef, &touched, rhs, &bounds, &point) {
                cuts.push(cut);
            }
        }
        cuts
    }

    /// Add the cut `coeffs·x ≥ rhs` (user units) as a row that the validation
    /// of a solution skips, and re-solve.
    pub(crate) fn add_cut(&mut self, coeffs: CsVec, rhs: f64) -> Result<StopReason, Error> {
        let rows = self.num_constraints();
        let stop = self.add_row(coeffs, ComparisonOp::Ge, rhs, None);
        if self.num_constraints() > rows {
            self.cut_rows.resize(self.num_constraints(), false);
            self.cut_rows[rows] = true;
        }
        stop
    }
}

/// The cut `Σ coef_k x_k ≥ rhs` over the `touched` vars, filtered and made
/// safe (module constants), or `None`.
fn finish_cut(
    coef: &[f64],
    touched: &[usize],
    mut rhs: f64,
    bounds: &[(f64, f64)],
    point: &[f64],
) -> Option<(CsVec, f64)> {
    let mut vars = touched.to_vec();
    vars.sort_unstable();
    vars.dedup();
    let mut terms: Vec<(usize, f64)> = vars
        .iter()
        .map(|&k| (k, coef[k]))
        .filter(|&(_, c)| c != 0.0 && c.is_finite())
        .collect();
    if !rhs.is_finite() {
        return None;
    }
    let largest = terms.iter().fold(0.0_f64, |m, &(_, c)| m.max(c.abs()));
    if largest == 0.0 {
        return None;
    }
    // Remove the negligible terms, lowering the rhs by the most each can add.
    let mut kept = Vec::with_capacity(terms.len());
    for (k, c) in terms.drain(..) {
        if c.abs() >= REL_ZERO * largest {
            kept.push((k, c));
            continue;
        }
        let (lo, hi) = bounds[k];
        let most = (c * lo).max(c * hi);
        if !most.is_finite() {
            return None;
        }
        rhs -= most;
    }
    let smallest = kept.iter().fold(f64::INFINITY, |m, &(_, c)| m.min(c.abs()));
    if largest > MAX_DYNAMISM * smallest {
        return None;
    }
    let magnitude = kept.iter().fold(rhs.abs(), |m, &(k, c)| {
        let (lo, hi) = bounds[k];
        let reach = [lo, hi, point[k]]
            .iter()
            .filter(|b| b.is_finite())
            .fold(0.0_f64, |r, b| r.max(b.abs()));
        m + c.abs() * reach
    });
    rhs -= SAFETY * magnitude.max(1.0);
    let norm = kept.iter().map(|&(_, c)| c * c).sum::<f64>().sqrt();
    let activity: f64 = kept.iter().map(|&(k, c)| c * point[k]).sum();
    if (rhs - activity) / norm < MIN_EFFICACY {
        return None;
    }
    kept.sort_by_key(|&(k, _)| k);
    let (indices, data): (Vec<usize>, Vec<f64>) = kept.into_iter().unzip();
    Some((CsVec::new(point.len(), indices, data), rhs))
}
