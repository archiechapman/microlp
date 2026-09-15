use crate::solver::EPS;
use crate::sparse::{Error, Perm, ScatteredVec, SparseMat, TriangleMat};
use std::cmp::Ordering;

#[derive(Clone)]
pub struct LUFactors {
    lower: TriangleMat,
    upper: TriangleMat,
    row_perm: Option<Perm>,
    col_perm: Option<Perm>,
}

#[derive(Clone, Debug)]
pub struct ScratchSpace {
    rhs: ScatteredVec,
    dense_rhs: Vec<f64>,
    mark_nonzero: MarkNonzero,
}

impl ScratchSpace {
    pub fn with_capacity(n: usize) -> ScratchSpace {
        ScratchSpace {
            rhs: ScatteredVec::empty(n),
            dense_rhs: vec![0.0; n],
            mark_nonzero: MarkNonzero::with_capacity(n),
        }
    }

    pub(crate) fn clear_sparse(&mut self, size: usize) {
        self.rhs.clear_and_resize(size);
        self.mark_nonzero.clear_and_resize(size);
    }
}

impl std::fmt::Debug for LUFactors {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "L:\n{:?}", self.lower)?;
        writeln!(f, "U:\n{:?}", self.upper)?;
        writeln!(
            f,
            "row_perm.new2orig: {:?}",
            self.row_perm.as_ref().map(|p| &p.new2orig)
        )?;
        writeln!(
            f,
            "col_perm.new2orig: {:?}",
            self.col_perm.as_ref().map(|p| &p.new2orig)
        )?;
        Ok(())
    }
}

impl LUFactors {
    pub fn nnz(&self) -> usize {
        self.lower.nondiag.nnz() + self.upper.nondiag.nnz() + self.lower.cols()
    }

    pub fn solve_dense(&self, rhs: &mut [f64], scratch: &mut ScratchSpace) {
        scratch.dense_rhs.resize(rhs.len(), 0.0);

        if let Some(row_perm) = &self.row_perm {
            for (i, rhs_el) in rhs.iter().enumerate() {
                scratch.dense_rhs[row_perm.orig2new[i]] = *rhs_el;
            }
        } else {
            scratch.dense_rhs.copy_from_slice(rhs);
        }

        tri_solve_dense(&self.lower, Triangle::Lower, &mut scratch.dense_rhs);
        tri_solve_dense(&self.upper, Triangle::Upper, &mut scratch.dense_rhs);

        if let Some(col_perm) = &self.col_perm {
            for i in 0..rhs.len() {
                rhs[col_perm.new2orig[i]] = scratch.dense_rhs[i];
            }
        } else {
            rhs.copy_from_slice(&scratch.dense_rhs);
        }
    }

    pub fn solve(&self, rhs: &mut ScatteredVec, scratch: &mut ScratchSpace) {
        if let Some(row_perm) = &self.row_perm {
            scratch.rhs.clear();
            for &i in &rhs.nonzero {
                let new_i = row_perm.orig2new[i];
                scratch.rhs.nonzero.push(new_i);
                scratch.rhs.is_nonzero[new_i] = true;
                scratch.rhs.values[new_i] = rhs.values[i];
            }
        } else {
            std::mem::swap(&mut scratch.rhs, rhs);
        }

        tri_solve_sparse(&self.lower, scratch);
        tri_solve_sparse(&self.upper, scratch);

        if let Some(col_perm) = &self.col_perm {
            rhs.clear();
            for &i in &scratch.rhs.nonzero {
                let new_i = col_perm.new2orig[i];
                rhs.nonzero.push(new_i);
                rhs.is_nonzero[new_i] = true;
                rhs.values[new_i] = scratch.rhs.values[i];
            }
        } else {
            std::mem::swap(rhs, &mut scratch.rhs);
        }
    }

    pub fn transpose(&self) -> LUFactors {
        LUFactors {
            lower: self.upper.transpose(),
            upper: self.lower.transpose(),
            row_perm: self.col_perm.clone(),
            col_perm: self.row_perm.clone(),
        }
    }
}

pub fn lu_factorize<'a>(
    size: usize,
    get_col: impl Fn(usize) -> (&'a [usize], &'a [f64]),
    stability_coeff: f64,
    scratch: &mut ScratchSpace,
) -> Result<LUFactors, Error> {
    lu_factorize_impl(size, get_col, stability_coeff, scratch, None)
}

/// A column of the input matrix that [`lu_factorize_repairing`] replaced by a unit
/// column because it was numerically dependent on the columns factored before it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Replacement {
    /// Index of the dependent input column.
    pub col: usize,
    /// Row `r` of the unit column `e_r` that took its place.
    pub row: usize,
}

/// Like [`lu_factorize`], but a numerically dependent column does not fail the
/// factorization: it is replaced by the unit column `e_r` of a row `r` that has
/// no pivot yet and for which `row_available(r)` holds, and the replacement is
/// reported. The factors are those of the repaired matrix. This is the standard
/// simplex basis repair: `e_r` is the column of row `r`'s slack variable, so the
/// caller swaps that slack into the basis in place of the dependent column.
///
/// `row_available` must reject rows whose unit column is already part of the
/// matrix (a basic slack); among the unpivoted rows there is always one it
/// accepts, because a unit column pivots on its own row.
pub fn lu_factorize_repairing<'a>(
    size: usize,
    get_col: impl Fn(usize) -> (&'a [usize], &'a [f64]),
    stability_coeff: f64,
    scratch: &mut ScratchSpace,
    row_available: &dyn Fn(usize) -> bool,
) -> Result<(LUFactors, Vec<Replacement>), Error> {
    let mut replaced = Vec::new();
    let lu = lu_factorize_impl(
        size,
        get_col,
        stability_coeff,
        scratch,
        Some((row_available, &mut replaced)),
    )?;
    Ok((lu, replaced))
}

/// The repair hook of [`lu_factorize_impl`]: the `row_available` predicate of
/// [`lu_factorize_repairing`], paired with the log the substitutions it makes
/// are reported through. `None` factorizes strictly, failing on a dependent
/// column.
type RepairHook<'a> = (&'a dyn Fn(usize) -> bool, &'a mut Vec<Replacement>);

fn lu_factorize_impl<'a>(
    size: usize,
    get_col: impl Fn(usize) -> (&'a [usize], &'a [f64]),
    stability_coeff: f64,
    scratch: &mut ScratchSpace,
    mut repair: Option<RepairHook<'_>>,
) -> Result<LUFactors, Error> {
    // Implementation of the Gilbert-Peierls algorithm:
    //
    // Gilbert, John R., and Tim Peierls. "Sparse partial pivoting in time
    // proportional to arithmetic operations." SIAM Journal on Scientific and
    // Statistical Computing 9.5 (1988): 862-874.
    //
    // https://ecommons.cornell.edu/bitstream/handle/1813/6623/86-783.pdf

    let mat_nnz = (0..size).map(|c| get_col(c).0.len()).sum::<usize>();
    trace!(
        "lu_factorize: starting, matrix size: {}, nnz: {} (excess: {})",
        size,
        mat_nnz,
        mat_nnz - size,
    );

    let col_perm = super::ordering::order_simple(size, |c| get_col(c).0)?;

    let mut orig_row2elt_count = vec![0; size];
    for col_rows in (0..size).map(|c| get_col(c).0) {
        for &orig_r in col_rows {
            orig_row2elt_count[orig_r] += 1;
        }
    }

    scratch.clear_sparse(size);

    let mut lower = SparseMat::new(size);
    let mut upper = SparseMat::new(size);
    let mut upper_diag = Vec::with_capacity(size);

    let mut new2orig_row = (0..size).collect::<Vec<_>>();
    let mut orig2new_row = new2orig_row.clone();

    for i_col in 0..size {
        let mat_col = get_col(col_perm.new2orig[i_col]);

        // Solve the equation L'_j * x = a_j (x will be in scratch.rhs).
        // L'_j is a sq. matrix with the first j columns of L
        // and columns of identity matrix after j.
        // Part of x above the diagonal (in the new row indices) is the column of U
        // and part of x below the diagonal divided by the pivot value is the column of L

        scratch.rhs.set(mat_col.0.iter().copied().zip(mat_col.1));

        scratch.mark_nonzero.run(
            &mut scratch.rhs,
            |new_i| lower.col_rows(new_i),
            |new_i| new_i < i_col,
            |orig_r| orig2new_row[orig_r],
        );

        // At this point all future nonzero positions of scratch.rhs are marked
        // and the order in which variables depend on each other is determined.
        // rev() because DFS returns vertices in reverse topological order.
        for &orig_i in scratch.mark_nonzero.visited.iter().rev() {
            // values[orig_i] is already fully calculated, diag coeff = 1.0.
            let new_i = orig2new_row[orig_i];
            if new_i < i_col {
                let x_val = scratch.rhs.values[orig_i];
                for (orig_r, coeff) in lower.col_iter(new_i) {
                    scratch.rhs.values[orig_r] -= x_val * coeff;
                }
            }
        }

        // Next we choose a pivot among values of x below the diagonal.
        // Pivoting by choosing the max element is good for stability,
        // but bad for sparseness, so we do threshold pivoting instead.

        let pivot_orig_r = {
            let mut max_abs: f64 = 0.0;
            for &orig_r in &scratch.rhs.nonzero {
                if orig2new_row[orig_r] < i_col {
                    continue;
                }

                let abs = f64::abs(scratch.rhs.values[orig_r]);
                if abs > max_abs {
                    max_abs = abs;
                }
            }

            if max_abs < EPS {
                let Some((row_available, replaced)) = repair.as_mut() else {
                    return Err(Error::SingularMatrix);
                };
                let pos = col_perm.new2orig[i_col];
                // Substitute the unit column of an unpivoted, available row. Its
                // forward solve is the unit vector itself (no pivoted row touches
                // it), so it pivots on that row with value 1.
                let Some(row) = (0..size)
                    .filter(|&r| orig2new_row[r] >= i_col && row_available(r))
                    .min_by_key(|&r| orig_row2elt_count[r])
                else {
                    return Err(Error::SingularMatrix);
                };
                replaced.push(Replacement { col: pos, row });
                scratch.rhs.clear();
                *scratch.rhs.get_mut(row) = 1.0;
                max_abs = 1.0;
            }

            assert!(max_abs.is_normal());

            // Choose among eligible pivot rows one with the least elements.
            // Gilbert-Peierls suggest to choose row with least elements *to the right*,
            // but it yielded poor results. Our heuristic is not a huge improvement either,
            // but at least we are less dependent on initial row ordering.
            let mut best_orig_r = None;
            let mut best_elt_count = None;
            for &orig_r in &scratch.rhs.nonzero {
                if orig2new_row[orig_r] < i_col {
                    continue;
                }

                if f64::abs(scratch.rhs.values[orig_r]) >= stability_coeff * max_abs {
                    let elt_count = orig_row2elt_count[orig_r];
                    if best_elt_count.is_none() || best_elt_count.unwrap() > elt_count {
                        best_orig_r = Some(orig_r);
                        best_elt_count = Some(elt_count);
                    }
                }
            }
            best_orig_r.unwrap()
        };

        let pivot_val = scratch.rhs.values[pivot_orig_r];

        {
            // Keep track of row permutations.
            let row = i_col;
            let orig_row = new2orig_row[row];
            let pivot_row = orig2new_row[pivot_orig_r];
            new2orig_row.swap(row, pivot_row);
            orig2new_row.swap(orig_row, pivot_orig_r);
        }

        // Gather the values of x into lower and upper matrices.

        for &orig_r in &scratch.rhs.nonzero {
            let val = scratch.rhs.values[orig_r];

            if val == 0.0 {
                continue;
            }

            let new_r = orig2new_row[orig_r];
            match new_r.cmp(&i_col) {
                Ordering::Less => upper.push(new_r, val),
                Ordering::Equal => upper_diag.push(pivot_val),
                Ordering::Greater => lower.push(orig_r, val / pivot_val),
            }
        }

        upper.seal_column();
        lower.seal_column();
    }

    // permute rows of lower to "new" indices.
    for i_col in 0..lower.cols() {
        for r in lower.col_rows_mut(i_col) {
            *r = orig2new_row[*r];
        }
    }

    let lower_nnz = lower.nnz();
    let upper_nnz = upper.nnz();
    trace!(
        "lu_factorize: done, lower nnz: {} (excess: {}), upper nnz: {} (excess: {}), additional fill-in: {}",
        lower_nnz + size,
        lower_nnz,
        upper_nnz + size,
        upper_nnz,
        (lower_nnz + upper_nnz + size) as isize - mat_nnz as isize,
    );

    let res = LUFactors {
        lower: TriangleMat {
            nondiag: lower,
            diag: None,
        },
        upper: TriangleMat {
            nondiag: upper,
            diag: Some(upper_diag),
        },
        row_perm: Some(Perm {
            orig2new: orig2new_row,
            new2orig: new2orig_row,
        }),
        col_perm: Some(col_perm),
    };

    Ok(res)
}

#[derive(Clone, Debug)]
struct MarkNonzero {
    dfs_stack: Vec<DfsStep>,
    is_visited: Vec<bool>,
    visited: Vec<usize>, // in reverse topological order
}

#[derive(Clone, Debug)]
struct DfsStep {
    orig_i: usize,
    cur_child: usize,
}

impl MarkNonzero {
    fn with_capacity(n: usize) -> MarkNonzero {
        MarkNonzero {
            dfs_stack: Vec::with_capacity(n),
            is_visited: vec![false; n],
            visited: vec![],
        }
    }

    fn clear(&mut self) {
        assert!(self.dfs_stack.is_empty());
        for &i in &self.visited {
            self.is_visited[i] = false;
        }
        self.visited.clear();
    }

    fn clear_and_resize(&mut self, n: usize) {
        self.clear();
        self.dfs_stack.reserve(n);
        self.is_visited.resize(n, false);
    }

    // compute the non-zero elements of the result by dfs traversal
    fn run<'a>(
        &mut self,
        rhs: &mut ScatteredVec,
        get_children: impl Fn(usize) -> &'a [usize] + 'a,
        filter: impl Fn(usize) -> bool,
        orig2new_row: impl Fn(usize) -> usize,
    ) {
        self.clear();

        for &orig_r in &rhs.nonzero {
            let new_r = orig2new_row(orig_r);
            if !filter(new_r) {
                continue;
            }
            if self.is_visited[orig_r] {
                continue;
            }

            self.dfs_stack.push(DfsStep {
                orig_i: orig_r,
                cur_child: 0,
            });
            while !self.dfs_stack.is_empty() {
                let cur_step = self.dfs_stack.last_mut().unwrap();
                let new_i = orig2new_row(cur_step.orig_i);
                let children = if filter(new_i) {
                    get_children(new_i)
                } else {
                    &[]
                };
                if !self.is_visited[cur_step.orig_i] {
                    self.is_visited[cur_step.orig_i] = true;
                } else {
                    cur_step.cur_child += 1;
                }

                while cur_step.cur_child < children.len() {
                    let child_orig_r = children[cur_step.cur_child];
                    if !self.is_visited[child_orig_r] {
                        break;
                    }
                    cur_step.cur_child += 1;
                }

                if cur_step.cur_child < children.len() {
                    let i_child = cur_step.cur_child;
                    self.dfs_stack.push(DfsStep {
                        orig_i: children[i_child],
                        cur_child: 0,
                    });
                } else {
                    self.visited.push(cur_step.orig_i);
                    self.dfs_stack.pop();
                }
            }
        }

        for &i in &self.visited {
            if !rhs.is_nonzero[i] {
                rhs.is_nonzero[i] = true;
                rhs.nonzero.push(i)
            }
        }
    }
}

enum Triangle {
    Lower,
    Upper,
}

fn tri_solve_dense(tri_mat: &TriangleMat, triangle: Triangle, rhs: &mut [f64]) {
    assert_eq!(tri_mat.rows(), rhs.len());
    match triangle {
        Triangle::Lower => {
            for col in 0..tri_mat.cols() {
                tri_solve_process_col(tri_mat, col, rhs);
            }
        }

        Triangle::Upper => {
            for col in (0..tri_mat.cols()).rev() {
                tri_solve_process_col(tri_mat, col, rhs);
            }
        }
    };
}

/// rhs is passed via scratch.visited, scratch.values.
fn tri_solve_sparse(tri_mat: &TriangleMat, scratch: &mut ScratchSpace) {
    assert_eq!(tri_mat.rows(), scratch.rhs.len());

    // compute the non-zero elements of the result by dfs traversal
    scratch.mark_nonzero.run(
        &mut scratch.rhs,
        |col| tri_mat.nondiag.col_rows(col),
        |_| true,
        |orig_i| orig_i,
    );

    // solve for the non-zero values into dense workspace.
    // rev() because DFS returns vertices in reverse topological order.
    for &col in scratch.mark_nonzero.visited.iter().rev() {
        tri_solve_process_col(tri_mat, col, &mut scratch.rhs.values);
    }
}

fn tri_solve_process_col(tri_mat: &TriangleMat, col: usize, rhs: &mut [f64]) {
    // all other variables in this row (multiplied by their coeffs)
    // are already subtracted from rhs[col].
    let x_val = if let Some(diag) = tri_mat.diag.as_ref() {
        rhs[col] / diag[col]
    } else {
        rhs[col]
    };

    rhs[col] = x_val;
    for (r, &coeff) in tri_mat.nondiag.col_iter(col) {
        rhs[r] -= x_val * coeff;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::helpers::{assert_matrix_eq, to_dense, to_sparse};
    use sprs::{CsMat, CsVec, TriMat};

    fn init() {
        let _ = env_logger::builder().is_test(true).try_init();
    }

    fn mat_from_triplets(rows: usize, cols: usize, triplets: &[(usize, usize, f64)]) -> CsMat<f64> {
        let mut mat = TriMat::with_capacity((rows, cols), triplets.len());
        for (r, c, val) in triplets {
            mat.add_triplet(*r, *c, *val);
        }
        mat.to_csc()
    }

    #[test]
    fn lu_simple() {
        init();
        let mat = mat_from_triplets(
            3,
            4,
            &[
                (0, 1, 2.0),
                (0, 0, 2.0),
                (0, 2, 123.0),
                (1, 2, 456.0),
                (1, 3, 1.0),
                (2, 1, 4.0),
                (2, 0, 3.0),
                (2, 2, 789.0),
                (2, 3, 1.0),
            ],
        );

        let mut scratch = ScratchSpace::with_capacity(mat.rows());
        let lu = lu_factorize(
            mat.rows(),
            |c| mat.outer_view([1, 0, 3][c]).unwrap().into_raw_storage(),
            0.9,
            &mut scratch,
        )
        .unwrap();
        let lu_transp = lu.transpose();

        let l_nondiag_ref = [
            vec![0.0, 0.0, 0.0],
            vec![0.5, 0.0, 0.0],
            vec![0.0, 0.0, 0.0],
        ];
        assert_matrix_eq(&lu.lower.nondiag.to_csmat(), &l_nondiag_ref);
        assert_eq!(lu.lower.diag, None);

        let u_nondiag_ref = [
            vec![0.0, 3.0, 1.0],
            vec![0.0, 0.0, -0.5],
            vec![0.0, 0.0, 0.0],
        ];
        let u_diag_ref = [4.0, 0.5, 1.0];
        assert_matrix_eq(&lu.upper.nondiag.to_csmat(), &u_nondiag_ref);
        assert_eq!(lu.upper.diag.as_ref().unwrap(), &u_diag_ref);

        assert_eq!(lu.row_perm.as_ref().unwrap().new2orig, &[2, 0, 1]);
        assert_eq!(lu.col_perm.as_ref().unwrap().new2orig, &[0, 1, 2]);

        {
            let mut rhs_dense = [6.0, 3.0, 13.0];
            lu.solve_dense(&mut rhs_dense, &mut scratch);
            assert_eq!(&rhs_dense, &[1.0, 2.0, 3.0]);
        }

        {
            let mut rhs_dense_t = [14.0, 11.0, 5.0];
            lu_transp.solve_dense(&mut rhs_dense_t, &mut scratch);
            assert_eq!(&rhs_dense_t, &[1.0, 2.0, 3.0]);
        }

        {
            let mut rhs = ScatteredVec::empty(3);
            rhs.set(to_sparse(&[0.0, -1.0, 0.0]).iter());
            lu.solve(&mut rhs, &mut scratch);
            assert_eq!(to_dense(&rhs.to_csvec()), vec![1.0, -1.0, -1.0]);
        }

        {
            let mut rhs = ScatteredVec::empty(3);
            rhs.set(to_sparse(&[0.0, -1.0, 1.0]).iter());
            lu_transp.solve(&mut rhs, &mut scratch);
            assert_eq!(to_dense(&rhs.to_csvec()), vec![-2.0, 0.0, 1.0]);
        }
    }

    #[test]
    fn lu_singular() {
        init();
        let size = 3;

        {
            let symbolically_singular = mat_from_triplets(
                size,
                size,
                &[(0, 0, 1.0), (1, 0, 1.0), (1, 1, 2.0), (1, 2, 3.0)],
            );

            let mut scratch = ScratchSpace::with_capacity(size);
            let err = lu_factorize(
                size,
                |c| {
                    symbolically_singular
                        .outer_view(c)
                        .unwrap()
                        .into_raw_storage()
                },
                0.9,
                &mut scratch,
            );
            assert_eq!(err.unwrap_err(), Error::SingularMatrix);
        }

        {
            let numerically_singular = mat_from_triplets(
                size,
                size,
                &[
                    (0, 0, 1.0),
                    (1, 0, 1.0),
                    (1, 1, 2.0),
                    (1, 2, 3.0),
                    (2, 0, 2.0),
                    (2, 1, 2.0),
                    (2, 2, 3.0),
                ],
            );

            let mut scratch = ScratchSpace::with_capacity(size);
            let err = lu_factorize(
                size,
                |c| {
                    numerically_singular
                        .outer_view(c)
                        .unwrap()
                        .into_raw_storage()
                },
                0.9,
                &mut scratch,
            );
            assert_eq!(err.unwrap_err(), Error::SingularMatrix);
        }
    }

    /// The same two singular matrices as `lu_singular`, but factored with a
    /// repair hook: a dependent column is replaced by the unit column of a row
    /// that has no pivot yet, the substitution is reported, and the factors
    /// returned are those of the repaired matrix (so they invert it).
    #[test]
    fn lu_repairing() {
        init();
        let size = 3;

        let symbolically_singular: &[(usize, usize, f64)] =
            &[(0, 0, 1.0), (1, 0, 1.0), (1, 1, 2.0), (1, 2, 3.0)];
        let numerically_singular: &[(usize, usize, f64)] = &[
            (0, 0, 1.0),
            (1, 0, 1.0),
            (1, 1, 2.0),
            (1, 2, 3.0),
            (2, 0, 2.0),
            (2, 1, 2.0),
            (2, 2, 3.0),
        ];

        for triplets in [symbolically_singular, numerically_singular] {
            let mat = mat_from_triplets(size, size, triplets);

            let mut scratch = ScratchSpace::with_capacity(size);
            let (lu, replaced) = lu_factorize_repairing(
                size,
                |c| mat.outer_view(c).unwrap().into_raw_storage(),
                0.9,
                &mut scratch,
                &|_| true,
            )
            .expect("a repairing factorization must not fail on a singular matrix");

            // Both matrices have rank 2: exactly one column is dependent.
            assert_eq!(replaced.len(), 1);
            let Replacement { col, row } = replaced[0];

            // Rebuild the matrix the repair actually factored: `col` swapped
            // for the unit column of `row`.
            let repaired = {
                let mut kept = triplets
                    .iter()
                    .copied()
                    .filter(|(_, c, _)| *c != col)
                    .collect::<Vec<_>>();
                kept.push((row, col, 1.0));
                mat_from_triplets(size, size, &kept)
            };

            // Solving with the factors and multiplying back by the repaired
            // matrix must return the right-hand side.
            let sparse_rhs = to_sparse(&[1.0, -2.0, 3.0]);
            let mut rhs = ScatteredVec::empty(size);
            rhs.set(sparse_rhs.iter());
            lu.solve(&mut rhs, &mut scratch);
            let diff = &sparse_rhs - &(&repaired * &rhs.to_csvec());
            assert!(
                diff.norm(1.0) < 1e-5,
                "the factors do not invert the repaired matrix"
            );
        }
    }

    #[test]
    fn lu_rand() {
        init();
        let size = 10;

        let mut rng = rand_pcg::Pcg64::seed_from_u64(12345);
        use rand::prelude::*;

        let mut mat = TriMat::new((size, size));
        for r in 0..size {
            for c in 0..size {
                if rng.random_range(0..2) == 0 {
                    mat.add_triplet(r, c, rng.random_range(0.0..1.0));
                }
            }
        }
        let mat: CsMat<f64> = mat.to_csc();

        let mut scratch = ScratchSpace::with_capacity(mat.rows());

        // TODO: random permutation?
        let cols: Vec<_> = (0..size).collect();

        let lu = lu_factorize(
            size,
            |c| mat.outer_view(cols[c]).unwrap().into_raw_storage(),
            0.1,
            &mut scratch,
        )
        .unwrap();
        let lu_transp = lu.transpose();

        let multiplied = &lu.lower.to_csmat() * &lu.upper.to_csmat();
        assert!(multiplied.is_csc());
        for (i, &c) in cols.iter().enumerate() {
            let permuted = {
                let mut is = vec![];
                let mut vs = vec![];
                for (i, &val) in mat.outer_view(c).unwrap().iter() {
                    is.push(lu.row_perm.as_ref().unwrap().orig2new[i]);
                    vs.push(val);
                }
                CsVec::new_from_unsorted(size, is, vs).unwrap()
            };
            let diff = &multiplied
                .outer_view(lu.col_perm.as_ref().unwrap().orig2new[i])
                .unwrap()
                - &permuted;
            assert!(diff.norm(1.0) < 1e-5);
        }

        let sparse_rhs = {
            let mut res = CsVec::empty(size);
            for i in 0..size {
                if rng.random_range(0..3) == 0 {
                    res.append(i, rng.random_range(0.0..1.0));
                }
            }
            res
        };

        {
            let mut rhs = ScatteredVec::empty(size);
            rhs.set(sparse_rhs.iter());
            lu.solve(&mut rhs, &mut scratch);
            let diff = &sparse_rhs - &(&mat * &rhs.to_csvec());
            assert!(diff.norm(1.0) < 1e-5);
        }

        {
            let mut rhs_t = ScatteredVec::empty(size);
            rhs_t.set(sparse_rhs.iter());
            lu_transp.solve(&mut rhs_t, &mut scratch);
            let diff = &sparse_rhs - &(&mat.transpose_view() * &rhs_t.to_csvec());
            assert!(diff.norm(1.0) < 1e-5);
        }
    }
}
