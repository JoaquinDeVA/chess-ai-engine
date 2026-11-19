//! The moves of a node: try each in order, keep the best, and reward the move that cuts.

use shakmaty::{Move, MoveList};

use super::move_list::NodeMoves;
use super::{NodeContext, NodeResult, Search};
use crate::board::{is_noisy, move_id};
use crate::search::scores::INFINITE;

impl Search<'_> {
    /// Search the moves of the node inside `(alpha, beta)` and return the best value and move.
    /// `futility_value` is the static value for pruning quiet moves, `None` if none is skipped.
    pub(super) fn search_moves(
        &mut self,
        context: &NodeContext,
        node_moves: NodeMoves,
        (mut alpha, mut beta): (i32, i32),
        depth_remaining: i32,
        table_move: Option<u32>,
        futility_value: Option<i32>,
    ) -> NodeResult {
        let NodeMoves {
            mut picker,
            killer_ids,
            killer_count,
        } = node_moves;
        let killers = &killer_ids[..killer_count];
        let (ply, mover, maximizing) = (context.ply, context.mover, context.maximizing);
        let mut best_value = if maximizing { -INFINITE } else { INFINITE };
        let mut best_move = None;
        // Quiet moves searched without a cut, to be penalized when a later move cuts.
        let mut failed = MoveList::new();
        let mut tried = 0;
        while let Some(m) = picker.next(self.state.pos()) {
            let index = tried;
            tried += 1;
            let id = move_id(&m);
            // Reducible: a quiet move, not a killer, in a node that is not in check. `Some` carries
            // whether the history says the move has cut before.
            let late = (!context.in_check && !is_noisy(&m) && !killers.contains(&id))
                .then(|| self.memory.history.score(mover, &m) > 0);
            if index > 0 && !is_noisy(&m) && !killers.contains(&id) {
                if let Some(value) = futility_value {
                    if self.is_futile(context, &m, value, depth_remaining, (alpha, beta)) {
                        continue;
                    }
                }
            }
            let nodes_before = self.nodes;
            self.apply(m);
            let child = self.search_child(context, index, (alpha, beta), depth_remaining, late);
            self.undo();
            if ply == 0 {
                self.root_nodes.push((id, self.nodes - nodes_before));
            }
            if maximizing {
                if child > best_value {
                    best_value = child;
                    best_move = Some(id);
                    alpha = alpha.max(best_value);
                }
            } else if child < best_value {
                best_value = child;
                best_move = Some(id);
                beta = beta.min(best_value);
            }
            if alpha >= beta {
                #[cfg(feature = "search-summary")]
                {
                    self.cutoffs += 1;
                }
                self.reward_cutoff(
                    context,
                    &m,
                    index,
                    depth_remaining,
                    table_move,
                    killers,
                    &failed,
                );
                break;
            }
            if !is_noisy(&m) {
                failed.push(m);
            }
        }
        (best_value, best_move)
    }

    /// A move cut: quiet moves that cut are remembered (killers, history,
    /// continuation_history) to be tried earlier in sibling nodes, and the quiet moves tried before it
    /// are penalized; captures are already ordered first.
    #[allow(clippy::too_many_arguments)]
    fn reward_cutoff(
        &mut self,
        context: &NodeContext,
        m: &Move,
        index: usize,
        depth_remaining: i32,
        table_move: Option<u32>,
        killers: &[u32],
        failed: &MoveList,
    ) {
        let (ply, mover) = (context.ply, context.mover);
        let id = move_id(m);
        let killer_slot = (!m.is_capture() && table_move != Some(id))
            .then(|| killers.iter().position(|&killer| killer == id))
            .flatten();
        self.memory.diagnostics.cutoff(index == 0, killer_slot);
        self.memory.killers.record(ply, *m);
        if !is_noisy(m) {
            self.memory.history.reward(mover, m, depth_remaining);
            for tried in failed {
                self.memory.history.penalize(mover, tried, depth_remaining);
            }
            if let Some(previous) = context.previous_piece_to {
                self.memory
                    .continuation_history
                    .reward(mover, previous, m, depth_remaining);
            }
        }
    }
}
