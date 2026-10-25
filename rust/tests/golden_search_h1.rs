//! Fixed-depth searches of the `h1` player against the golden file made by the Python engine.

use std::fs;
use std::path::PathBuf;

use _core::board::{uci_string, GameState};
use _core::eval::h1::{FEATURE_NAMES, H1};
use _core::search::{alpha_beta, Control, Memory, Options};
use serde_json::Value;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..")
}

fn read(path: PathBuf) -> Value {
    serde_json::from_str(&fs::read_to_string(&path).expect("file")).expect("valid JSON")
}

fn h1() -> H1 {
    let file = read(root().join("src/chess_engine/evaluation/h1_weights.json"));
    let mut weights = [0.0; 6];
    for (slot, name) in FEATURE_NAMES.iter().enumerate() {
        weights[slot] = file["weights"][name].as_f64().unwrap();
    }
    H1::new(weights)
}

#[test]
fn fixed_depth_search_matches_golden() {
    let positions = read(root().join("tests/golden/positions.json"));
    let golden = read(root().join("tests/golden/search_h1.json"));
    let positions = positions["positions"].as_array().unwrap();
    let eval = h1();
    let clock = || 0.0;
    let mut node_mismatches = 0;
    let results = golden["results"].as_array().unwrap();
    assert!(results.len() > 200);
    for case in results {
        let entry = &positions[case["index"].as_u64().unwrap() as usize];
        let moves: Vec<String> = entry["moves"]
            .as_array()
            .unwrap()
            .iter()
            .map(|m| m.as_str().unwrap().to_string())
            .collect();
        let mut state = GameState::from_moves(entry["root"].as_str().unwrap(), &moves).unwrap();
        let options = Options {
            quiescence_plies: case["quiescence"].as_u64().unwrap() as u32,
            ..Options::default()
        };
        let depth = case["depth"].as_i64().unwrap() as i32;
        let mut memory = Memory::default();
        let control = Control {
            start: 0.0,
            hard_limit: f64::INFINITY,
            clock: &clock,
            stop: None,
            rotation: 0,
        };
        let found = alpha_beta(&mut state, &eval, &mut memory, &options, depth, &control);
        let (value, best, nodes, prunings, max_ply) = (
            found.value,
            found.best,
            found.nodes,
            found.prunings,
            found.max_ply,
        );
        let label = format!("{} depth {} q {}", entry["fen"], depth, case["quiescence"]);
        assert!(
            (value - case["value"].as_f64().unwrap()).abs() < 1e-9,
            "value of {label}: {value}"
        );
        assert_eq!(
            best.as_ref().map(uci_string).as_deref(),
            case["move"].as_str(),
            "move of {label}"
        );
        assert_eq!(
            max_ply as i64,
            case["max_ply"].as_i64().unwrap(),
            "max ply of {label}"
        );
        if nodes != case["nodes"].as_u64().unwrap()
            || prunings != case["prunings"].as_u64().unwrap()
        {
            node_mismatches += 1;
            if node_mismatches <= 8 {
                eprintln!(
                    "{label}: nodes {nodes} vs {}, prunings {prunings} vs {}",
                    case["nodes"], case["prunings"]
                );
            }
        }
    }
    assert_eq!(
        node_mismatches, 0,
        "the counters differ in {node_mismatches} searches"
    );
}
