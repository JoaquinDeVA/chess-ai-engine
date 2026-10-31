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

#[test]
fn the_three_piece_tables_play_the_winning_move() {
    let tables = EndgameTablebase::open(&root().join("tests/data/syzygy")).unwrap();
    for (fen, expected) in [
        ("8/6R1/8/8/8/8/3k1K2/8 w - - 0 1", "g7g3"),
        ("8/k7/8/8/8/8/3KQ3/8 b - - 0 1", "a7b7"),
    ] {
        let pos = position_from_fen(fen).unwrap();
        let got = tables.best_move(&pos).map(|m| uci_string(&m));
        assert_eq!(got.as_deref(), Some(expected), "{fen}");
    }
}
