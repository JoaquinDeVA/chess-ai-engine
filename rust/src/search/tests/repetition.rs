//! Search behavior when a draw can be claimed by the next move.

use super::helpers::{control, material};
use crate::board::GameState;
use crate::search::{alpha_beta, Options};

#[test]
fn the_root_searches_when_a_move_can_claim_threefold() {
    let shuffle = ["g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6", "f3g1"];
    let mut state = GameState::from_moves(
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        &shuffle.map(String::from),
    )
    .unwrap();
    let mut memory = crate::search::Memory::default();
    let result = alpha_beta(
        &mut state,
        &material(),
        &mut memory,
        &Options::default(),
        1,
        &control(f64::INFINITY, &super::helpers::no_time),
    );

    assert!(result.nodes > 1);
    assert!(
        result.best.is_some(),
        "root search must select a legal move"
    );
}
