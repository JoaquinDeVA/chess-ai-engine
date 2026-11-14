//! Scores of the search: bounds, mate values and their conversion for the table.

/// A bound beyond every value, the window of the root.
pub const INFINITE: i32 = 32_000;
/// Value of a mate on the board; a mate `n` plies from the root is worth `MATE - n` to the winner,
/// so a quicker mate is preferred one unit per ply.
pub const MATE: i32 = 31_000;
/// Values from here out are mates: no line is longer than `MATE - MATE_BOUND` plies, and every
/// evaluation is within `EVAL_LIMIT`, far below.
pub const MATE_BOUND: i32 = MATE - 1_000;

/// A value as the table keeps it: the distance to a mate counts from the node, not from the root.
/// The map is increasing, so the bounds of the window can go through it as well.
pub(super) fn for_table(value: i32, ply: i32) -> i32 {
    if value >= MATE_BOUND {
        value + ply
    } else if value <= -MATE_BOUND {
        value - ply
    } else {
        value
    }
}

/// The inverse of [`for_table`] for a node at `ply`.
pub(super) fn at_ply(value: i32, ply: i32) -> i32 {
    if value >= MATE_BOUND {
        value - ply
    } else if value <= -MATE_BOUND {
        value + ply
    } else {
        value
    }
}
