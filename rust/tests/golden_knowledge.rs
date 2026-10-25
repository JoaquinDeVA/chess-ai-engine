//! The opening book and the endgame tables against the golden files made by the Python engine.

use std::fs;
use std::path::PathBuf;

use _core::board::{position_from_fen, uci_string};
use _core::knowledge::book::OpeningBook;
use _core::knowledge::tablebase::EndgameTablebase;
use serde_json::Value;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..")
}

fn read(path: PathBuf) -> Value {
    serde_json::from_str(&fs::read_to_string(&path).expect("file")).expect("valid JSON")
}

#[test]
fn the_book_matches_golden() {
    let book = OpeningBook::load(&root().join("data/opening_book.json")).unwrap();
    let golden = read(root().join("tests/golden/book.json"));
    assert_eq!(book.len() as u64, golden["size"].as_u64().unwrap());
    let cases = golden["cases"].as_array().unwrap();
    assert!(cases.len() > 300);
    for case in cases {
        let pos = position_from_fen(case["fen"].as_str().unwrap()).unwrap();
        let got = book.lookup(&pos).map(|m| uci_string(&m));
        assert_eq!(got.as_deref(), case["move"].as_str(), "{}", case["fen"]);
    }
}

fn check_tablebase(directory: PathBuf, cases: &Value) {
    let tables = EndgameTablebase::open(&directory).unwrap();
    let cases = cases.as_array().unwrap();
    assert!(!cases.is_empty());
    for case in cases {
        let pos = position_from_fen(case["fen"].as_str().unwrap()).unwrap();
        let got = tables.best_move(&pos).map(|m| uci_string(&m));
        assert_eq!(got.as_deref(), case["move"].as_str(), "{}", case["fen"]);
    }
}

#[test]
fn the_three_piece_tables_match_golden() {
    let golden = read(root().join("tests/golden/tablebase.json"));
    check_tablebase(root().join("tests/data/syzygy"), &golden["three_piece"]);
}

#[test]
fn the_five_piece_tables_match_golden_when_present() {
    let directory = root().join("data/knowledge/endgame");
    if !directory.is_dir() {
        eprintln!("skipped: no endgame tables in {}", directory.display());
        return;
    }
    let golden = read(root().join("tests/golden/tablebase.json"));
    check_tablebase(directory, &golden["five_piece"]);
}
