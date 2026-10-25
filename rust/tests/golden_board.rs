//! The terminal test of the Rust rules against the golden file made by the Python engine.

use std::fs;
use std::path::PathBuf;

use _core::board::GameState;
use serde_json::Value;
use shakmaty::Color;

fn golden(name: &str) -> Value {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../tests/golden")
        .join(name);
    serde_json::from_str(&fs::read_to_string(&path).expect("golden file")).expect("valid JSON")
}

#[test]
fn terminal_matches_golden() {
    let file = golden("terminal.json");
    let cases = file["cases"].as_array().unwrap();
    assert!(cases.len() > 2000);
    let mut over_count = 0;
    for case in cases {
        let moves: Vec<String> = case["moves"]
            .as_array()
            .unwrap()
            .iter()
            .map(|m| m.as_str().unwrap().to_string())
            .collect();
        let state = GameState::from_moves(case["root"].as_str().unwrap(), &moves).unwrap();
        let legal = _core::board::ordered_moves(state.pos());
        let outcome = state.outcome(&legal);
        let over = case["over"].as_bool().unwrap();
        assert_eq!(outcome.is_some(), over, "over differs in {}", case["fen"]);
        if over {
            over_count += 1;
            let winner = match outcome.unwrap() {
                Some(Color::White) => Some("white"),
                Some(Color::Black) => Some("black"),
                None => None,
            };
            assert_eq!(
                winner,
                case["winner"].as_str(),
                "winner differs in {}",
                case["fen"]
            );
        }
    }
    assert!(over_count > 50);
}

#[test]
fn the_final_fen_matches_after_replaying_the_moves() {
    let file = golden("positions.json");
    for entry in file["positions"].as_array().unwrap().iter().take(500) {
        let moves: Vec<String> = entry["moves"]
            .as_array()
            .unwrap()
            .iter()
            .map(|m| m.as_str().unwrap().to_string())
            .collect();
        let state = GameState::from_moves(entry["root"].as_str().unwrap(), &moves).unwrap();
        let fen = _core::board::fen_of(state.pos());
        // python-chess prints the en passant square only when a capture is possible, as we do
        assert_eq!(fen, entry["fen"].as_str().unwrap());
    }
}
