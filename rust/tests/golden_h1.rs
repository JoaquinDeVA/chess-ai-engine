//! The heuristic `h1` against the golden features and values made by the Python engine.

use std::fs;
use std::path::PathBuf;

use _core::board::position_from_fen;
use _core::eval::h1::{features, H1};
use _core::eval::Evaluator;
use serde_json::Value;
use shakmaty::Color;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..")
}

fn read(path: PathBuf) -> Value {
    serde_json::from_str(&fs::read_to_string(&path).expect("file")).expect("valid JSON")
}

fn weights() -> [f64; 6] {
    let file = read(root().join("src/chess_engine/models/h1_weights.json"));
    let names = _core::eval::h1::FEATURE_NAMES;
    let mut out = [0.0; 6];
    for (slot, name) in names.iter().enumerate() {
        out[slot] = file["weights"][name].as_f64().expect("weight");
    }
    out
}

#[test]
fn h1_features_and_values_match_golden() {
    let positions = read(root().join("tests/golden/positions.json"));
    let golden_features = read(root().join("tests/golden/features_h1.json"));
    let golden_values = read(root().join("tests/golden/eval_h1.json"));
    let h1 = H1::new(weights());
    let positions = positions["positions"].as_array().unwrap();
    let rows = golden_features["rows"].as_array().unwrap();
    let values = golden_values["rows"].as_array().unwrap();
    assert_eq!(positions.len(), rows.len());
    for (i, entry) in positions.iter().enumerate() {
        let pos = position_from_fen(entry["fen"].as_str().unwrap()).unwrap();
        let got = features(&pos);
        for (k, expected) in rows[i].as_array().unwrap().iter().enumerate() {
            let expected = expected.as_f64().unwrap();
            assert!(
                (got[k] - expected).abs() < 1e-12,
                "feature {k} of {}: {} vs {expected}",
                entry["fen"],
                got[k]
            );
        }
        let white = h1.evaluate(&pos, Color::White);
        let black = h1.evaluate(&pos, Color::Black);
        let want = values[i].as_array().unwrap();
        assert!(
            (white - want[0].as_f64().unwrap()).abs() < 1e-12,
            "white value of {}",
            entry["fen"]
        );
        assert!(
            (black - want[1].as_f64().unwrap()).abs() < 1e-12,
            "black value of {}",
            entry["fen"]
        );
    }
}
