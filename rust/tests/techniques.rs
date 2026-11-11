//! The optional techniques prune or reduce the search: they must keep the forced results and most
//! of the moves of the default search. Positions: tests/data/positions.json; evaluation: the
//! compiled-in network.

use std::fs;
use std::path::PathBuf;

use _core::board::GameState;
use _core::eval::sparse_net::Net;
use _core::search::{alpha_beta, Control, Memory, Options, Techniques};
use serde_json::Value;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..")
}

fn read(path: PathBuf) -> Value {
    serde_json::from_str(&fs::read_to_string(&path).expect("file")).expect("valid JSON")
}

/// The network the UCI binary compiles in.
fn net() -> Net {
    let text = fs::read_to_string(root().join("rust/models/net_pst_w64.json")).expect("model");
    Net::from_json(&text).expect("valid model")
}

/// Values and moves of the first `count` positions at depth `depth`, one search each.
fn plain_values(techniques: Techniques, depth: i32, count: usize) -> Vec<(f64, Option<String>)> {
    let positions = read(root().join("tests/data/positions.json"));
    let net = net();
    let clock = || 0.0;
    let options = Options {
        quiescence_plies: 4,
        techniques,
        ..Options::default()
    };
    let mut out = Vec::new();
    for entry in positions["positions"]
        .as_array()
        .unwrap()
        .iter()
        .take(count)
    {
        let moves: Vec<String> = entry["moves"]
            .as_array()
            .unwrap()
            .iter()
            .map(|m| m.as_str().unwrap().to_string())
            .collect();
        let mut state = GameState::from_moves(entry["root"].as_str().unwrap(), &moves).unwrap();
        let legal = _core::board::ordered_moves(state.pos());
        if state.outcome(&legal).is_some() {
            continue;
        }
        let control = Control {
            start: 0.0,
            hard_limit: f64::INFINITY,
            clock: &clock,
            stop: None,
            progress: None,
            rotation: 0,
        };
        let found = alpha_beta(
            &mut state,
            &net,
            &mut Memory::default(),
            &options,
            depth,
            &control,
        );
        out.push((
            found.value,
            found.best.map(|m| _core::board::uci_string(&m)),
        ));
    }
    out
}

#[test]
#[cfg_attr(
    debug_assertions,
    ignore = "slow without optimisation: cargo test --release"
)]
fn the_optional_techniques_keep_the_forced_results_and_most_moves() {
    let reference = plain_values(Techniques::default(), 5, 120);
    let with = plain_values(Techniques::all(), 5, 120);
    assert_eq!(reference.len(), with.len());
    let (mut forced, mut kept, mut same_move) = (0, 0, 0);
    for ((a, move_a), (b, move_b)) in reference.iter().zip(&with) {
        same_move += usize::from(move_a == move_b);
        if *a > 0.97 || *a < 0.03 {
            forced += 1;
            kept += usize::from((*a > 0.5) == (*b > 0.5) && (b - 0.5).abs() > 0.4);
        }
    }
    eprintln!(
        "forced {forced}, kept {kept}, same move {same_move} of {}",
        reference.len()
    );
    assert!(
        forced >= 3,
        "the sample has too few forced results ({forced})"
    );
    assert!(
        kept as f64 >= 0.9 * forced as f64,
        "{kept} of {forced} forced results kept"
    );
    // The five techniques together change the move in about a third of the positions (74 of 115
    // equal when written); the forced results are what must not change.
    assert!(
        same_move as f64 >= 0.6 * reference.len() as f64,
        "{same_move} equal moves"
    );
}
