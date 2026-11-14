//! The search of one child: principal variation search with late move reductions.

use shakmaty::Position;

use super::reductions::late_move_reduction;
use super::{NodeContext, Search, NULL_WINDOW_WIDTH};

impl Search<'_> {
    /// Value of the child just played (the `index`-th move of the node).
    ///
    /// The first move gets the full window; the others a null window that only asks whether they
    /// improve on the best so far, and the full window again only if they do. `late_quiet` is
    /// `Some` for a move that may be reduced, and carries whether the history favors it.
    pub(super) fn search_child(
        &mut self,
        context: &NodeContext,
        index: usize,
        (alpha, beta): (i32, i32),
        depth_remaining: i32,
        late_quiet: Option<bool>,
    ) -> i32 {
        let ply = context.ply;
        let maximizing = context.maximizing;
        let (low, high) = if maximizing {
            (alpha, alpha + NULL_WINDOW_WIDTH)
        } else {
            (beta - NULL_WINDOW_WIDTH, beta)
        };
        let has_full_window = beta - alpha > NULL_WINDOW_WIDTH;
        if index > 0 && depth_remaining >= 2 {
            // A late quiet move that gives no check is searched shallower first; if even so it
            // improves on the best move, it is searched again to the full depth.
            let reduction_plies = match late_quiet {
                Some(favorable_history) if !self.state.pos().is_check() => {
                    late_move_reduction(depth_remaining, index, favorable_history)
                }
                _ => 0,
            };
            if reduction_plies > 0 {
                let (shallow, _) = self.node(
                    low,
                    high,
                    depth_remaining - 1 - reduction_plies,
                    ply + 1,
                    true,
                );
                if (shallow <= alpha && maximizing) || (shallow >= beta && !maximizing) {
                    return shallow;
                }
            }
            let (scout, _) = self.node(low, high, depth_remaining - 1, ply + 1, true);
            if scout > alpha && scout < beta && !self.truncated && has_full_window {
                return self.node(alpha, beta, depth_remaining - 1, ply + 1, true).0;
            }
            return scout;
        }
        self.node(alpha, beta, depth_remaining - 1, ply + 1, true).0
    }
}
