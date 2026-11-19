//! Techniques that search less deep: late move reductions, internal iterative reduction and the
//! reduction of the null move.

/// From this depth on the null-move reduction is one ply deeper.
const NULL_MOVE_DEEP_FROM_DEPTH: i32 = 6;
/// Internal iterative reduction works from this depth on.
const IIR_FROM_DEPTH: i32 = 4;
/// Late move reductions start at this depth and at this position in the move order.
const LMR_FROM_DEPTH: i32 = 3;
const LMR_FROM_INDEX: usize = 3;

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

/// Plies the null-move search is reduced: 2, or 3 from `NULL_MOVE_DEEP_FROM_DEPTH`.
pub(super) fn null_move_reduction(depth_remaining: i32) -> i32 {
    if depth_remaining >= NULL_MOVE_DEEP_FROM_DEPTH {
        3
    } else {
        2
    }
}

/// Internal iterative reduction: with no table move the order is poor, so search one ply
/// less; the next iteration, with the move stored, goes deeper.
pub(super) fn internal_iterative_reduction(
    depth_remaining: i32,
    ply: i32,
    table_move: Option<u32>,
) -> i32 {
    if table_move.is_none() && ply > 0 && depth_remaining >= IIR_FROM_DEPTH {
        depth_remaining - 1
    } else {
        depth_remaining
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn late_move_reductions_follow_the_depth_and_the_position_in_the_order() {
        assert_eq!(
            late_move_reduction(3, 2, false),
            0,
            "too early in the order"
        );
        assert_eq!(late_move_reduction(2, 9, false), 0, "too shallow");
        assert_eq!(late_move_reduction(3, 3, false), 1);
        assert_eq!(late_move_reduction(6, 10, false), 2);
        assert_eq!(
            late_move_reduction(6, 10, true),
            1,
            "favorable history: one ply less"
        );
        assert_eq!(
            late_move_reduction(3, 100, false),
            1,
            "never below one ply of search"
        );
    }

    #[test]
    fn the_null_move_reduction_deepens_from_depth_six() {
        assert_eq!(null_move_reduction(3), 2);
        assert_eq!(null_move_reduction(5), 2);
        assert_eq!(null_move_reduction(6), 3);
        assert_eq!(null_move_reduction(12), 3);
    }
}
