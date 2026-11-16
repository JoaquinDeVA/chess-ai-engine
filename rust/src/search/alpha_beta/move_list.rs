//! The moves of a node in the order they are tried.

use shakmaty::MoveList;

use super::{NodeContext, Search};
use crate::board::move_id;
use crate::search::ordering::{move_to_front, MovePicker, MAX_KILLERS_PER_PLY};

/// The moves of a node, handed out in the order they are tried, and the killers of its ply (they
/// are not reduced or pruned).
pub(super) struct NodeMoves {
    pub(super) picker: MovePicker,
    pub(super) killer_ids: [u32; MAX_KILLERS_PER_PLY],
    pub(super) killer_count: usize,
}

impl Search<'_> {
    /// The legal `moves` in the order they are tried: table move, captures that do not lose
    /// material, killers, quiet moves by history, and the losing captures. The root
    /// keeps only the moves it is allowed; helper threads rotate the root instead of putting the
    /// table move first.
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
        let restricted = context.ply == 0 && self.options.root_moves.is_some();
        let helper_root = depth_remaining == self.max_depth && self.control.rotation > 0;
        let listed = restricted || helper_root;
        let mut picker = MovePicker::new(
            moves,
            &killer_ids[..killer_count],
            if listed { None } else { table_move },
            (&self.memory.history, context.mover),
            context
                .previous_piece_to
                .map(|previous| (&self.memory.continuation, previous)),
        );
        if listed {
            let pos = self.state.pos();
            let mut ordered: MoveList = std::iter::from_fn(|| picker.next(pos)).collect();
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
            // Helper threads rotate the root moves so they do not all search the same subtree;
            // the table move is not put first there for the same reason.
            if helper_root {
                let shift = self.control.rotation % ordered.len();
                ordered.rotate_left(shift);
            } else {
                move_to_front(&mut ordered, table_move);
            }
            picker.replace_with(ordered);
        }
        NodeMoves {
            picker,
            killer_ids,
            killer_count,
        }
    }
}
