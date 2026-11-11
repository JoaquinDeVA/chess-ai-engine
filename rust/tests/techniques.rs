//! The techniques that only change the order or the windows of the search must give the value of
//! the reference search. Positions: tests/data/positions.json; evaluation: the compiled-in network.

use std::fs;
use std::path::PathBuf;

use _core::board::GameState;
use _core::eval::sparse_net::Net;
use _core::eval::Eval;
use _core::search::{alpha_beta, Control, Memory, Options, Techniques};
use serde_json::Value;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..")
}

fn read(path: PathBuf) -> Value {
    serde_json::from_str(&fs::read_to_string(&path).expect("file")).expect("valid JSON")
}

/// The network the UCI binary compiles in.
fn net() -> Eval {
    let text = fs::read_to_string(root().join("rust/models/net_pst_w64.json")).expect("model");
    Eval::Net(Net::from_json(&text).expect("valid model"))
}

/// Values of the first `count` positions at `depth` with the given techniques (iterative
/// deepening from depth 1, so that the table has something to use).
fn values(techniques: Techniques, depth: i32, count: usize) -> Vec<f64> {
    let positions = read(root().join("tests/data/positions.json"));
    let eval = net();
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
        let moves = _core::board::ordered_moves(state.pos());
        if state.outcome(&moves).is_some() {
            continue;
        }
        let mut memory = Memory::default();
        let control = Control {
            start: 0.0,
            hard_limit: f64::INFINITY,
            clock: &clock,
            stop: None,
            progress: None,
            rotation: 0,
        };
        let mut last = 0.0;
        for d in 1..=depth {
            last = alpha_beta(&mut state, &eval, &mut memory, &options, d, &control).value;
        }
        out.push(last);
    }
    out
}

fn same_value_as_the_reference(techniques: Techniques) {
    let reference = values(Techniques::default(), 3, 80);
    let with = values(techniques, 3, 80);
    assert_eq!(reference.len(), with.len());
    assert!(reference.len() > 60);
    for (i, (a, b)) in reference.iter().zip(&with).enumerate() {
        assert!(
            (a - b).abs() < 1e-9,
            "position {i}: {a} against {b} with {techniques:?}"
        );
    }
}

#[test]
#[cfg_attr(
    debug_assertions,
    ignore = "slow without optimisation: cargo test --release"
)]
fn the_table_move_does_not_change_the_value() {
    same_value_as_the_reference(Techniques {
        tt_move: true,
        ..Techniques::default()
    });
}

/// Values of the first `count` positions at depth `depth`, one search each (no deepening).
fn plain_values(techniques: Techniques, depth: i32, count: usize) -> Vec<(f64, Option<String>)> {
    let positions = read(root().join("tests/data/positions.json"));
    let eval = net();
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
            &eval,
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
fn the_selective_techniques_keep_the_forced_results_and_most_moves() {
    let selective = Techniques {
        null_move: true,
        ..Techniques::default()
    };
    let reference = plain_values(Techniques::default(), 5, 120);
    let with = plain_values(selective, 5, 120);
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
    assert!(
        same_move as f64 >= 0.7 * reference.len() as f64,
        "{same_move} equal moves"
    );
}

#[test]
#[cfg_attr(
    debug_assertions,
    ignore = "slow without optimisation: cargo test --release"
)]
fn the_null_move_is_not_used_in_pawn_endings() {
    // Lasker's zugzwang position: passing the turn would look winning, but nobody may pass.
    let fen = "8/k7/3p4/p2P1p2/P2P1P2/8/8/K7 w - - 0 1";
    let eval = net();
    let clock = || 0.0;
    let mut values = Vec::new();
    for null_move in [false, true] {
        let options = Options {
            quiescence_plies: 4,
            techniques: Techniques {
                null_move,
                ..Techniques::default()
            },
            ..Options::default()
        };
        let mut state = GameState::from_moves(fen, &[]).unwrap();
        let control = Control {
            start: 0.0,
            hard_limit: f64::INFINITY,
            clock: &clock,
            stop: None,
            progress: None,
            rotation: 0,
        };
        values.push(
            alpha_beta(
                &mut state,
                &eval,
                &mut Memory::default(),
                &options,
                7,
                &control,
            )
            .value,
        );
    }
    assert!((values[0] - values[1]).abs() < 1e-9, "{values:?}");
}
