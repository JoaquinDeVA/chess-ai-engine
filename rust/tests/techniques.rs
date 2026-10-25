//! The techniques that only change the order or the windows of the search must give the value of
//! the reference search. Positions: the golden ones; evaluation: `h1`.

use std::fs;
use std::path::PathBuf;

use _core::board::GameState;
use _core::eval::h1::{FEATURE_NAMES, H1};
use _core::search::{alpha_beta, Control, Memory, Options, Techniques};
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

/// Values of the first `count` positions at `depth` with the given techniques (iterative
/// deepening from depth 1, so that the table and the history have something to use).
fn values(techniques: Techniques, depth: i32, count: usize) -> Vec<f64> {
    let positions = read(root().join("tests/golden/positions.json"));
    let eval = h1();
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
fn the_table_move_does_not_change_the_value() {
    same_value_as_the_reference(Techniques {
        tt_move: true,
        ..Techniques::default()
    });
}

#[test]
fn the_history_does_not_change_the_value() {
    same_value_as_the_reference(Techniques {
        history: true,
        ..Techniques::default()
    });
}

#[test]
fn pvs_does_not_change_the_value() {
    same_value_as_the_reference(Techniques {
        pvs: true,
        ..Techniques::default()
    });
}

#[test]
fn aspiration_windows_do_not_change_the_value() {
    // the iterations of this helper are searched by hand, so the windows are tested through the
    // deepening below
    let positions = read(root().join("tests/golden/positions.json"));
    let eval = h1();
    let clock = || 0.0;
    let mut checked = 0;
    for entry in positions["positions"].as_array().unwrap().iter().take(40) {
        let moves: Vec<String> = entry["moves"]
            .as_array()
            .unwrap()
            .iter()
            .map(|m| m.as_str().unwrap().to_string())
            .collect();
        let fen = entry["root"].as_str().unwrap();
        let mut results = Vec::new();
        for aspiration in [false, true] {
            let options = Options {
                quiescence_plies: 4,
                min_depth: 8,
                estimator: "never_stop".into(),
                techniques: Techniques {
                    aspiration,
                    ..Techniques::default()
                },
                ..Options::default()
            };
            let mut state = GameState::from_moves(fen, &moves).unwrap();
            let legal = _core::board::ordered_moves(state.pos());
            if state.outcome(&legal).is_some() {
                break;
            }
            let control = Control {
                start: 0.0,
                hard_limit: f64::INFINITY,
                clock: &clock,
                stop: None,
                progress: None,
                rotation: 0,
            };
            let (best, stats) = _core::search::iterative_deepening(
                &mut state,
                &eval,
                &mut Memory::default(),
                &options,
                5,
                1e9,
                &control,
            )
            .unwrap();
            results.push((
                best.map(|m| _core::board::uci_string(&m)),
                stats.completed_depth,
            ));
        }
        if results.len() == 2 {
            assert_eq!(results[0].1, results[1].1);
            checked += usize::from(results[0].0 == results[1].0);
        }
    }
    assert!(checked >= 36, "{checked} of 40 moves equal");
}
