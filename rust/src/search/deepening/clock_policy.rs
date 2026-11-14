//! The clock policy of the deepening: how long the soft limit is stretched, and which iterations
//! are not worth starting.

/// Time of an iteration over the time of the one before it, rounded down (measured: median 2.4,
/// 46 % above 2.5, 23 % above 4).
const NEXT_ITERATION_COST: f64 = 3.5;
/// An iteration is not started either when it would end more than this many times past the scaled
/// soft limit: it would spend far more than the move was given (the cut it risks is the hard one).
const SOFT_OVERRUN: f64 = 4.0;
/// First completed depth whose best move scales the soft limit (the first ones are noise).
const SCALE_FROM_DEPTH: usize = 6;
/// Factor of the soft limit by the completed iterations in a row with the same best move: one that
/// has just changed gets more time, one that has held for four or more gets less.
const STABILITY_SCALE: [f64; 5] = [2.5, 1.2, 0.9, 0.8, 0.75];
/// Factor of the soft limit by the share `f` of the nodes spent under the best move:
/// `(NODE_SHARE_PIVOT - f) * NODE_SHARE_SCALE`, from 2.0 (f = 0) to 0.68 (f = 1).
const NODE_SHARE_PIVOT: f64 = 1.5;
const NODE_SHARE_SCALE: f64 = 1.35;

/// Factor of the soft limit after an iteration of `depth` whose best move has held `stable`
/// iterations in a row and took `share` of the nodes.
pub(super) fn soft_scale(depth: usize, stable: usize, share: f64) -> f64 {
    if depth < SCALE_FROM_DEPTH {
        return 1.0;
    }
    STABILITY_SCALE[stable.min(STABILITY_SCALE.len() - 1)]
        * (NODE_SHARE_PIVOT - share)
        * NODE_SHARE_SCALE
}

/// Whether the next iteration is worth starting after one that took `last_seconds`, with `elapsed`
/// seconds spent since the start of the search. An iteration costs about `NEXT_ITERATION_COST` times
/// the last one: one that the hard limit would cut is not started (its work is thrown away), nor one
/// that would end `SOFT_OVERRUN` times past the (scaled) soft limit.
pub(super) fn next_iteration_fits(
    elapsed: f64,
    last_seconds: f64,
    hard_limit: f64,
    soft: f64,
) -> bool {
    let predicted_end = elapsed + last_seconds * NEXT_ITERATION_COST;
    !(predicted_end >= hard_limit || predicted_end > soft * SOFT_OVERRUN)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_soft_limit_grows_for_a_new_move_and_shrinks_for_a_settled_one() {
        assert_eq!(soft_scale(SCALE_FROM_DEPTH - 1, 0, 0.0), 1.0);
        let fresh = soft_scale(SCALE_FROM_DEPTH, 0, 0.3);
        let settled = soft_scale(SCALE_FROM_DEPTH, 9, 0.9);
        assert!((fresh - 2.5 * 1.2 * 1.35).abs() < 1e-9, "{fresh}");
        assert!(settled < 0.65 && settled < fresh, "{settled}");
    }
}
