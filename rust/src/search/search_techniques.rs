//! Constants and formulas of the search techniques: null move, late move
//! reductions (LMR), reverse futility pruning (RFP), internal iterative reduction (IIR) and
//! futility pruning.

use shakmaty::{Chess, Color, Position};

/// The null move is tried from this depth on.
pub(super) const NULL_MOVE_FROM_DEPTH: i32 = 3;
/// Centipawns beyond the bound that deepen the null-move reduction by one ply.
const NULL_MOVE_MARGIN_STEP: i32 = 200;
/// Internal iterative reduction works from this depth on.
pub(super) const IIR_FROM_DEPTH: i32 = 4;
/// Late move reductions start at this depth and at this position in the move order.
const LMR_FROM_DEPTH: i32 = 3;
const LMR_FROM_INDEX: usize = 3;

/// Reverse futility pruning works up to this depth, and prunes when the static value is beyond the
/// bound by `RFP_MARGIN_PER_PLY` per ply left (60 centipawns; it was 80).
pub(super) const RFP_UP_TO_DEPTH: i32 = 6;
pub(super) const RFP_MARGIN_PER_PLY: i32 = 60;

/// Futility pruning works up to this depth, with a margin of `FUTILITY_MARGIN_PER_PLY` per ply left
/// (90 centipawns; it was 120).
pub(super) const SHALLOW_UP_TO_DEPTH: i32 = 3;
pub(super) const FUTILITY_MARGIN_PER_PLY: i32 = 90;

/// Plies a late quiet move is searched less: `0.75 + ln(depth)·ln(index) / 2.25`, one less when
/// the history says the move has cut before, and never so many that the search falls below one ply.
pub(super) fn late_move_reduction(
    depth_remaining: i32,
    index: usize,
    favorable_history: bool,
) -> i32 {
    if depth_remaining < LMR_FROM_DEPTH || index < LMR_FROM_INDEX {
        return 0;
    }
    let raw = 0.75 + f64::from(depth_remaining).ln() * (index as f64).ln() / 2.25;
    let steps = raw as i32 - i32::from(favorable_history);
    steps.clamp(0, depth_remaining - 2)
}

/// Plies the null-move search is reduced: 2 from depth 3, one more every four plies of depth, and one
/// more per `NULL_MOVE_MARGIN_STEP` by which the static value is beyond the bound (two at most).
pub(super) fn null_move_reduction(depth_remaining: i32, margin: i32) -> i32 {
    2 + depth_remaining / 4 + (margin / NULL_MOVE_MARGIN_STEP).clamp(0, 2)
}

/// Whether `color` has a piece other than pawns and the king in `pos` (otherwise passing the turn
/// can be the worst thing to do, and a null move is not tried).
pub(super) fn has_pieces(pos: &Chess, color: Color) -> bool {
    let board = pos.board();
    (board.by_color(color) & !(board.pawns() | board.kings())).any()
}
