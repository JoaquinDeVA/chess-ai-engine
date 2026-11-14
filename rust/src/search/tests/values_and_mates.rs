//! Values, mates and quiescence: what a fixed-depth search returns.

use crate::board::uci_string;
use crate::search::*;

use super::helpers::*;

#[test]
fn a_quicker_win_is_worth_more_than_a_slower_one() {
    let one = fixed(MATE_IN_ONE, 3);
    assert_eq!(one.best.as_ref().map(uci_string).as_deref(), Some("a1a8"));
    assert_eq!(one.value, MATE - 1);
    // Depth 6, not 4: late move reductions search the quiet mating rook move a ply shallower.
    let two = fixed(MATE_IN_TWO, 6);
    assert!(
        two.value > MATE_BOUND && two.value < one.value,
        "{}",
        two.value
    );
}

#[test]
#[cfg(feature = "search-summary")]
fn check_extends_the_search_by_one_ply() {
    let result = fixed("4k3/8/8/8/8/8/4r3/4K3 w - - 0 1", 1);
    assert!(result.best.is_some());
    assert!(result.report.max_ply >= 2);
}

#[test]
fn quiescence_searches_evasions_when_in_check() {
    let result = quiescent("4k3/8/8/8/8/8/4r3/4K3 w - - 0 1", 0);
    assert!(result.nodes > 1);
}

#[test]
fn a_mate_keeps_its_distance_when_it_comes_back_from_the_table() {
    // The deepening reads the mates of the previous iteration from the table at other plies.
    let two = fixed(MATE_IN_TWO, 6);
    assert_eq!(two.value, MATE - 3);
}

#[test]
fn an_entry_that_the_fifty_move_clock_could_change_is_not_stored() {
    let stored = |fen: &str| {
        let mut game = state(fen);
        let key = game.key();
        let mut memory = Memory::default();
        alpha_beta(
            &mut game,
            &material(),
            &mut memory,
            &Options::default(),
            4,
            &control(f64::INFINITY, &no_time),
        );
        memory.table.get(&key).is_some()
    };
    assert!(stored("4k3/8/8/8/8/8/8/R3K3 w - - 0 1"));
    assert!(stored("4k3/8/8/8/8/8/8/R3K3 w - - 90 80"));
    assert!(!stored("4k3/8/8/8/8/8/8/R3K3 w - - 97 80"));
}

#[test]
fn the_state_is_restored_after_every_search() {
    let mut game = state(START);
    let before = game.key();
    let options = Options {
        quiescence_plies: 4,
        ..Options::default()
    };
    alpha_beta(
        &mut game,
        &material(),
        &mut Memory::default(),
        &options,
        3,
        &control(f64::INFINITY, &no_time),
    );
    assert_eq!(game.key(), before);
}

#[test]
fn quiescence_sees_the_recapture_at_the_horizon() {
    // White can take a protected pawn with the queen: a depth-1 search without quiescence sees
    // only the gain, with quiescence it sees the loss of the queen.
    let fen = "4k3/8/3p4/4p3/8/8/4Q3/4K3 w - - 0 1";
    let plain = fixed(fen, 1);
    let with = quiescent(fen, 1);
    assert_ne!(
        plain.best.as_ref().map(uci_string),
        with.best.as_ref().map(uci_string)
    );
    assert_ne!(with.best.as_ref().map(uci_string).as_deref(), Some("e2e5"));
}

#[test]
fn see_keeps_the_obvious_capture() {
    // A queen hangs: the search takes it.
    let result = quiescent(HANGING_QUEEN, 3);
    assert_eq!(
        result.best.as_ref().map(uci_string).as_deref(),
        Some("d2d5")
    );
}
