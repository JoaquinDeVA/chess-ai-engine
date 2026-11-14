//! Techniques that skip work: reverse futility, null move and futility pruning.

use shakmaty::{Chess, Color, Move, Position};

use super::reductions::null_move_reduction;
use super::{NodeContext, Outcome, Search, NULL_WINDOW_WIDTH};
use crate::eval::EVAL_LIMIT;

/// The null move is tried from this depth on.
const NULL_MOVE_FROM_DEPTH: i32 = 3;

/// Reverse futility pruning works up to this depth, and prunes when the static value is beyond the
/// bound by `RFP_MARGIN_PER_PLY` per ply left (60 centipawns; it was 80).
const RFP_UP_TO_DEPTH: i32 = 6;
const RFP_MARGIN_PER_PLY: i32 = 60;

/// Futility pruning works up to this depth, with a margin of `FUTILITY_MARGIN_PER_PLY` per ply left
/// (90 centipawns; it was 120).
const SHALLOW_UP_TO_DEPTH: i32 = 3;
const FUTILITY_MARGIN_PER_PLY: i32 = 90;

impl Search<'_> {
    /// Reverse futility: a static value far beyond the bound, in a window of no width and not in
    /// check, is taken as the value of the node.
    pub(super) fn reverse_futility_value(
        &mut self,
        context: &NodeContext,
        depth_remaining: i32,
        (alpha, beta): (i32, i32),
        static_cache: &mut Option<i32>,
    ) -> Option<i32> {
        if context.ply > 0
            && depth_remaining <= RFP_UP_TO_DEPTH
            && !context.in_check
            && beta - alpha <= NULL_WINDOW_WIDTH
            // A bound in the range of the mates is not one a static value can judge.
            && alpha > -EVAL_LIMIT
            && beta < EVAL_LIMIT
        {
            let value = self.cached_static_value(static_cache);
            let margin = RFP_MARGIN_PER_PLY * depth_remaining;
            if (context.maximizing && value - beta >= margin)
                || (!context.maximizing && alpha - value >= margin)
            {
                return Some(value);
            }
        }
        None
    }

    /// Null move: it needs a pass to be safe (not in check, not at the root, not twice in a
    /// row) and pieces on the board, or zugzwang makes passing the best move. Only tried when
    /// the static value already beats the bound, so the reduced search rarely wastes time.
    /// Returns the bound when passing the turn already cuts.
    pub(super) fn null_move_cut(
        &mut self,
        context: &NodeContext,
        depth_remaining: i32,
        (alpha, beta): (i32, i32),
        allow_null: bool,
        static_cache: &mut Option<i32>,
    ) -> Option<Outcome> {
        if !(allow_null
            && context.ply > 0
            && depth_remaining >= NULL_MOVE_FROM_DEPTH
            && !context.in_check
            && has_pieces(self.state.pos(), context.mover))
        {
            return None;
        }
        let value = self.cached_static_value(static_cache);
        let margin = if context.maximizing {
            value - beta
        } else {
            alpha - value
        };
        if margin < 0 {
            return None;
        }
        // If passing the turn already cuts, playing a move will too: prune.
        let reduction = null_move_reduction(depth_remaining, margin);
        let (low, high) = if context.maximizing {
            (beta - NULL_WINDOW_WIDTH, beta)
        } else {
            (alpha, alpha + NULL_WINDOW_WIDTH)
        };
        self.play_null();
        let (value, _) = self.node(
            low,
            high,
            depth_remaining - 1 - reduction,
            context.ply + 1,
            false,
        );
        self.unplay();
        if !self.truncated
            && ((context.maximizing && value >= beta) || (!context.maximizing && value <= alpha))
        {
            return Some((if context.maximizing { beta } else { alpha }, None));
        }
        None
    }

    /// The static value that the shallow pruning of the quiet moves compares against, `None` when
    /// the node is not shallow: in check, too deep, or with bounds in the range of the mates.
    pub(super) fn futility_value(
        &mut self,
        context: &NodeContext,
        depth_remaining: i32,
        (alpha, beta): (i32, i32),
        static_cache: &mut Option<i32>,
    ) -> Option<i32> {
        let shallow = !context.in_check
            && depth_remaining <= SHALLOW_UP_TO_DEPTH
            && alpha > -EVAL_LIMIT
            && beta < EVAL_LIMIT;
        shallow.then(|| self.cached_static_value(static_cache))
    }

    /// Whether the quiet move `action` is skipped: the static value is so far from the (current)
    /// window that a move cannot bring it back, and the move gives no check.
    pub(super) fn is_futile(
        &self,
        context: &NodeContext,
        action: &Move,
        static_value: i32,
        depth_remaining: i32,
        (alpha, beta): (i32, i32),
    ) -> bool {
        let margin = FUTILITY_MARGIN_PER_PLY * depth_remaining;
        let hopeless = if context.maximizing {
            alpha - static_value >= margin
        } else {
            static_value - beta >= margin
        };
        if !hopeless {
            return false;
        }
        // A move that gives check is never skipped.
        let mut after = self.state.pos().clone();
        after.play_unchecked(*action);
        !after.is_check()
    }
}

/// Whether `color` has a piece other than pawns and the king in `pos` (otherwise passing the turn
/// can be the worst thing to do, and a null move is not tried).
fn has_pieces(pos: &Chess, color: Color) -> bool {
    let board = pos.board();
    (board.by_color(color) & !(board.pawns() | board.kings())).any()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::position_from_fen;

    #[test]
    fn the_null_move_is_not_tried_with_only_pawns_and_the_king() {
        // Lasker's zugzwang position: passing the turn would look winning, but nobody may pass.
        let lasker = position_from_fen("8/k7/3p4/p2P1p2/P2P1P2/8/8/K7 w - - 0 1").unwrap();
        assert!(!has_pieces(&lasker, Color::White));
        assert!(!has_pieces(&lasker, Color::Black));
        let start =
            position_from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1").unwrap();
        assert!(has_pieces(&start, Color::White));
    }
}
