//! Plain-data branch & bound tree nodes.

use crate::solver::Basis;
use std::collections::BTreeMap;
use std::sync::Arc;

/// One open node of the search tree. Contains no solver machinery — applying
/// `bound_changes` on top of the root bounds plus loading `basis` fully
/// reconstructs the node's starting state.
#[derive(Clone, Debug)]
pub(crate) struct Node {
    /// Cumulative bound changes from the root, in creation order; for a var that
    /// appears multiple times the LAST entry is its current bounds.
    pub bound_changes: Vec<(usize, f64, f64)>,
    /// Optimal basis of the parent node (dual-feasible warm start after tightening).
    pub basis: Basis,
    /// Parent's LP objective in internal (minimize) space — a valid lower bound
    /// for this node, used for pruning before any LP work.
    pub lp_bound: f64,
    pub depth: u32,
    /// Sequence number of the branching that created this node; used to detect
    /// "the solver is already at my parent's optimum" (warm dive).
    pub parent_id: u64,
    /// Which variable this node's creating branch changed. Root-like nodes that
    /// do not represent a variable branch carry `None` and cannot update a
    /// variable pseudocost.
    pub branch_var: Option<usize>,
    /// Direction and parent fractionality for a variable branch.
    pub branch_up: bool,
    pub branch_frac: f64,
    /// Bounds deduced by propagation on the path from the root to this node
    /// (see [`Deductions`]). `None` until something is deduced above or here.
    pub deduced: Option<Arc<Deductions>>,
    /// Propagation has already run here (see [`crate::SolveOptions::propagate_rounds`]).
    /// A node interrupted before its LP is visited again, and propagation is capped at a
    /// number of sweeps, so a second pass could deduce more than the first — which would
    /// make a resumed search explore differently from an uninterrupted one.
    pub propagated: bool,
}

/// One link in the chain of bounds deduced by propagation along a root-to-node path.
///
/// A deduction made at a node stays valid in its whole subtree, so children must
/// inherit it or propagation cannot compound down a path. It is kept as a shared
/// chain rather than folded into [`Node::bound_changes`], because that list is
/// cumulative and is cloned into BOTH children at every branch: recording every
/// deduction there made the lists grow with depth until the search exhausted
/// memory on a model with thousands of variables. A link is allocated once, shared
/// by every descendant, and freed when the last node holding it is closed.
#[derive(Debug)]
pub(crate) struct Deductions {
    /// The rest of the path, shared with the parent node.
    parent: Option<Arc<Deductions>>,
    /// The bounds deduced at this node alone.
    changes: Vec<(usize, f64, f64)>,
}

impl Deductions {
    /// Chain `changes` onto `parent`. Deducing nothing adds no link.
    pub fn chain(
        parent: Option<Arc<Deductions>>,
        changes: Vec<(usize, f64, f64)>,
    ) -> Option<Arc<Deductions>> {
        if changes.is_empty() {
            parent
        } else {
            Some(Arc::new(Deductions { parent, changes }))
        }
    }
}

/// The bounds a node starts from: its branching bounds tightened by every deduction
/// on its path from the root. Both are valid restrictions of the node's subproblem,
/// so they intersect; the result is sorted by var, one entry each.
pub(crate) fn node_bounds(
    changes: &[(usize, f64, f64)],
    deduced: &Option<Arc<Deductions>>,
) -> Vec<(usize, f64, f64)> {
    let mut map: BTreeMap<usize, (f64, f64)> = BTreeMap::new();
    for &(v, lo, hi) in changes {
        map.insert(v, (lo, hi));
    }
    let mut link = deduced.as_deref();
    while let Some(d) = link {
        for &(v, lo, hi) in &d.changes {
            map.entry(v)
                .and_modify(|b| {
                    b.0 = b.0.max(lo);
                    b.1 = b.1.min(hi);
                })
                .or_insert((lo, hi));
        }
        link = d.parent.as_deref();
    }
    map.into_iter().map(|(v, (lo, hi))| (v, lo, hi)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn later_bound_changes_win() {
        let changes = vec![(3, 0.0, 7.0), (1, 0.0, 1.0), (3, 2.0, 5.0)];
        assert_eq!(
            node_bounds(&changes, &None),
            vec![(1, 0.0, 1.0), (3, 2.0, 5.0)]
        );
        assert_eq!(node_bounds(&[], &None), vec![]);
    }

    #[test]
    fn deductions_intersect_along_the_path() {
        // Grandparent deduced x3 <= 6 and x7 >= 2; the parent then deduced x3 <= 4.
        let grandparent = Deductions::chain(None, vec![(3, 0.0, 6.0), (7, 2.0, 9.0)]);
        let parent = Deductions::chain(grandparent, vec![(3, 0.0, 4.0)]);
        // The node branched x3 >= 1 and carries an untouched var of its own.
        let changes = vec![(3, 1.0, 4.0), (5, 0.0, 1.0)];
        assert_eq!(
            node_bounds(&changes, &parent),
            vec![(3, 1.0, 4.0), (5, 0.0, 1.0), (7, 2.0, 9.0)]
        );
    }

    #[test]
    fn the_tighter_of_branch_and_deduction_wins() {
        let deduced = Deductions::chain(None, vec![(2, 3.0, 8.0)]);
        // A branch that is looser on one side than the ancestor deduction keeps
        // the deduction's side: both restrict the same subproblem.
        assert_eq!(node_bounds(&[(2, 0.0, 5.0)], &deduced), vec![(2, 3.0, 5.0)]);
    }

    #[test]
    fn empty_deductions_add_no_link() {
        let chain = Deductions::chain(None, vec![]);
        assert!(chain.is_none());
        let one = Deductions::chain(None, vec![(1, 0.0, 1.0)]);
        let same = Deductions::chain(one.clone(), vec![]);
        assert!(std::sync::Arc::ptr_eq(
            one.as_ref().unwrap(),
            same.as_ref().unwrap()
        ));
    }
}
