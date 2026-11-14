//! The moves of a node in the order they are tried.

use shakmaty::MoveList;

use super::{NodeContext, Search};
use crate::board::move_id;
use crate::search::ordering::{move_to_front, order, see_ge, MAX_KILLERS_PER_PLY};

/// The ordered moves of a node and the killers of its ply (they are not reduced or pruned).
pub(super) struct NodeMoves {
    pub(super) moves: MoveList,
    killer_ids: [u32; MAX_KILLERS_PER_PLY],
    killer_count: usize,
}

impl NodeMoves {
    pub(super) fn killers(&self) -> &[u32] {
        &self.killer_ids[..self.killer_count]
    }
}

impl Search<'_> {
    /// The legal `moves` ordered: killers, countermove, history, and the captures that lose
    /// material after the quiet ones. The root keeps only the moves it is allowed; helper threads
    /// rotate the root; everywhere else the table move goes first.
    pub(super) fn node_moves(
        &self,
        context: &NodeContext,
        moves: &MoveList,
        depth_remaining: i32,
        table_move: Option<u32>,
    ) -> NodeMoves {
        let mut killer_ids = [0u32; MAX_KILLERS_PER_PLY];
        let stored = self.memory.killers.get(context.ply);
        killer_ids[..stored.len()].copy_from_slice(stored);
        let killer_count = stored.len();
        let counter = self
            .memory
            .countermoves
            .get(context.mover, context.previous);
        let ordered = order(
            moves,
            &killer_ids[..killer_count],
            counter,
            Some((&self.memory.history, context.mover)),
            context
                .previous_piece_to
                .map(|previous| (&self.memory.continuation, previous)),
        );
        // Captures that lose material come after the quiet moves (the order is otherwise kept).
        let pos = self.state.pos();
        let mut ordered: MoveList = ordered
            .iter()
            .filter(|m| !m.is_capture() || see_ge(pos, m, 0))
            .chain(
                ordered
                    .iter()
                    .filter(|m| m.is_capture() && !see_ge(pos, m, 0)),
            )
            .copied()
            .collect();
        if let (0, Some(allowed)) = (context.ply, &self.options.root_moves) {
            let kept: MoveList = ordered
                .iter()
                .filter(|m| allowed.contains(&move_id(m)))
                .copied()
                .collect();
            // A list that matches no legal move is a mistake of the caller: search everything.
            if !kept.is_empty() {
                ordered = kept;
            }
        }
        // Helper threads rotate the root moves so they do not all search the same subtree; the
        // table move is not put first there for the same reason.
        let helper_root = depth_remaining == self.max_depth && self.control.rotation > 0;
        if helper_root {
            let shift = self.control.rotation % ordered.len();
            ordered.rotate_left(shift);
        } else {
            move_to_front(&mut ordered, table_move);
        }
        NodeMoves {
            moves: ordered,
            killer_ids,
            killer_count,
        }
    }
}
