//! The opening book and the endgame tables.

use std::path::PathBuf;

use _core::board::{position_from_fen, uci_string};
use _core::knowledge::book::OpeningBook;
use _core::knowledge::tablebase::EndgameTablebase;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..")
}

#[test]
fn the_book_returns_the_entry_of_a_position_and_nothing_else() {
    let start = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
    let key = _core::knowledge::book::position_key(start);
    let book = OpeningBook::from_entries([(key, "e2e4".to_string())]);
    let pos = position_from_fen(start).unwrap();
    assert_eq!(
        book.lookup(&pos).map(|m| uci_string(&m)).as_deref(),
        Some("e2e4")
    );
    let other = position_from_fen("4k3/8/8/8/8/8/8/4K3 w - - 0 1").unwrap();
    assert!(book.lookup(&other).is_none());
}

/// The moves the tables leave to the search, in UCI.
fn root_moves(tables: &EndgameTablebase, fen: &str) -> Option<Vec<String>> {
    let pos = position_from_fen(fen).unwrap();
    let moves = tables.root_moves(&pos)?;
    Some(moves.iter().map(uci_string).collect())
}

#[test]
fn the_three_piece_tables_keep_only_the_moves_that_hold_the_result() {
    let tables = EndgameTablebase::open(&root().join("tests/data/syzygy")).unwrap();
    // The rook wins; Rb2 hands it to the king (a draw); only the fastest win is left.
    let moves = root_moves(&tables, "8/8/8/8/8/2k5/R7/4K3 w - - 0 1").unwrap();
    assert!(moves.contains(&"a2a4".to_string()), "{moves:?}");
    assert!(!moves.contains(&"a2b2".to_string()), "{moves:?}");
    // Lost for black: no move changes it, none is stalemate, and the king can walk away.
    let moves = root_moves(&tables, "8/k7/8/8/8/8/3KQ3/8 b - - 0 1").unwrap();
    assert!(moves.contains(&"a7b6".to_string()), "{moves:?}");
}

#[test]
fn a_mate_in_one_is_the_only_move_left() {
    let tables = EndgameTablebase::open(&root().join("tests/data/syzygy")).unwrap();
    let moves = root_moves(&tables, "7k/8/6K1/8/8/8/Q7/8 w - - 0 1").unwrap();
    assert_eq!(moves, ["a2a8"]);
}

#[test]
fn positions_the_tables_do_not_cover_give_no_moves() {
    let tables = EndgameTablebase::open(&root().join("tests/data/syzygy")).unwrap();
    assert!(root_moves(
        &tables,
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    )
    .is_none());
}

#[test]
fn a_winning_capture_stays_among_the_moves_and_a_draw_does_not() {
    // Game FHRCwTDL, move 69: Qxa2 wins and zeroes the counter; Qb8 draws.
    let tables = EndgameTablebase::open(&root().join("tests/data/syzygy")).unwrap();
    let moves = root_moves(&tables, "8/8/8/2k5/8/8/pQ5K/8 w - - 2 69").unwrap();
    assert!(moves.contains(&"b2a2".to_string()), "{moves:?}");
    assert!(!moves.contains(&"b2b8".to_string()), "{moves:?}");
}

#[test]
fn a_winning_move_that_throws_a_queen_away_is_not_among_the_fastest() {
    // Game FHRCwTDL, move 65: Qxa3+ and Qc7+ both win, but Qc7+ Kxc7 throws a queen away and counts
    // as progress only because the capture zeroes the counter. Qxa3+ is the one left.
    let tables = EndgameTablebase::open(&root().join("tests/data/syzygy")).unwrap();
    let moves = root_moves(&tables, "6Q1/8/3k4/8/8/p7/7K/2Q5 w - - 3 65").unwrap();
    assert_eq!(moves, ["c1a3"]);
}
