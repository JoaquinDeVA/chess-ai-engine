//! The root restricted to a list of moves.

use crate::board::uci_string;
use crate::search::*;

use super::helpers::*;

#[test]
fn the_root_only_searches_the_moves_it_is_given() {
    let fen = MATE_IN_ONE;
    let search_with = |root_moves: Option<Vec<u32>>| {
        let options = Options {
            root_moves,
            ..Options::default()
        };
        alpha_beta(
            &mut state(fen),
            &material(),
            &mut Memory::default(),
            &options,
            2,
            &control(f64::INFINITY, &no_time),
        )
        .best
        .as_ref()
        .map(uci_string)
    };
    assert_eq!(search_with(None).as_deref(), Some("a1a8"));
    // Mate is left out of the list, so another listed move is played.
    let allowed = ids(fen, &["a1a2", "a1b1"]);
    assert_eq!(allowed.len(), 2);
    let best = search_with(Some(allowed)).unwrap();
    assert!(best == "a1a2" || best == "a1b1");
    // A list that matches no legal move is ignored.
    assert_eq!(search_with(Some(vec![u32::MAX])).as_deref(), Some("a1a8"));
}

#[test]
fn a_table_entry_does_not_bring_back_a_move_the_root_list_excludes() {
    let fen = MATE_IN_ONE;
    let mut memory = Memory::default();
    let run = |memory: &mut Memory, root_moves: Option<Vec<u32>>| {
        let options = Options {
            root_moves,
            ..Options::default()
        };
        alpha_beta(
            &mut state(fen),
            &material(),
            memory,
            &options,
            3,
            &control(f64::INFINITY, &no_time),
        )
        .best
        .as_ref()
        .map(uci_string)
    };
    // The unrestricted search stores the mate in the table.
    assert_eq!(run(&mut memory, None).as_deref(), Some("a1a8"));
    let best = run(&mut memory, Some(ids(fen, &["a1a2", "a1b1"]))).unwrap();
    assert!(best == "a1a2" || best == "a1b1", "{best}");
}
