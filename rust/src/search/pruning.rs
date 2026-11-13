//! Constants and formulas of the pruning and reduction techniques: null move, late move
//! reductions (LMR), reverse futility pruning (RFP), internal iterative reduction (IIR) and
//! futility pruning.

use shakmaty::{Chess, Color, Position};

/// The null move is tried from this depth on.
pub(super) const NULL_MOVE_FROM_DEPTH: i32 = 3;
/// Internal iterative reduction works from this depth on.
pub(super) const IIR_FROM_DEPTH: i32 = 4;
/// Late move reductions start at this depth and at this position in the move order.
const LMR_FROM_DEPTH: i32 = 3;
const LMR_FROM_INDEX: usize = 3;

/// Reverse futility pruning works up to this depth, and prunes when the static value is beyond the
/// bound by `RFP_MARGIN_PER_PLY` per ply left (it was 0.46 log-odds, the same 80 centipawns).
pub(super) const RFP_UP_TO_DEPTH: i32 = 6;
pub(super) const RFP_MARGIN_PER_PLY: i32 = 80;

/// Futility pruning works up to this depth, with a margin of `FUTILITY_MARGIN_PER_PLY` per ply left
/// (it was 0.69 log-odds, the same 120 centipawns).
pub(super) const SHALLOW_UP_TO_DEPTH: i32 = 3;
pub(super) const FUTILITY_MARGIN_PER_PLY: i32 = 120;

/// Plies a late quiet move is searched less: `0.75 + ln(depth)·ln(index) / 2.25`, one less when
/// the history says the move has cut before, and never so many that the search falls below one ply.
pub(super) fn reduction(depth_remaining: i32, index: usize, good_history: bool) -> i32 {
    if depth_remaining < LMR_FROM_DEPTH || index < LMR_FROM_INDEX {
        return 0;
    }
    let raw = 0.75 + f64::from(depth_remaining).ln() * (index as f64).ln() / 2.25;
    let steps = raw as i32 - i32::from(good_history);
    steps.clamp(0, depth_remaining - 2)
}

/// Whether `color` has a piece other than pawns and the king in `pos` (otherwise passing the turn
/// can be the worst thing to do, and a null move is not tried).
pub(super) fn has_pieces(pos: &Chess, color: Color) -> bool {
    let board = pos.board();
    (board.by_color(color) & !(board.pawns() | board.kings())).any()
}
