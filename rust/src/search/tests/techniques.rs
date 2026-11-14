//! The pruning and reduction techniques must not lose what a plain search finds.

use crate::board::uci_string;
use crate::search::*;

use super::helpers::*;

#[test]
fn the_search_techniques_keep_the_forced_mate_and_the_hanging_queen() {
    // LMR, RFP, IIR and futility all work in these searches: none may lose a mate in two or a
    // free queen.
    assert_eq!(quiescent(MATE_IN_TWO, 6).value, MATE - 3);
    for depth in [3, 5] {
        assert_eq!(
            quiescent(HANGING_QUEEN, depth)
                .best
                .as_ref()
                .map(uci_string)
                .as_deref(),
            Some("d2d5")
        );
    }
}
