//! Tests of the search.

use std::sync::atomic::{AtomicU64, Ordering};

use shakmaty::{Chess, Color, Position, Role};

use super::*;
use crate::board::{position_from_fen, uci_string, GameState};
use crate::eval::Evaluator;

/// Material balance as an expected score; every call advances a fake clock by `tick` seconds.
struct Material<'a> {
    clock: &'a FakeClock,
    tick: f64,
}

impl Evaluator for Material<'_> {
    fn evaluate(&self, pos: &Chess, player: Color) -> f64 {
        self.clock.advance(self.tick);
        let board = pos.board();
        let value = |role: Role| match role {
            Role::Pawn => 1.0,
            Role::Knight | Role::Bishop => 3.0,
            Role::Rook => 5.0,
            Role::Queen => 9.0,
            Role::King => 0.0,
        };
        let mut white = 0.0;
        for role in Role::ALL {
            white +=
                value(role) * (board.by_role(role) & board.by_color(Color::White)).count() as f64;
            white -=
                value(role) * (board.by_role(role) & board.by_color(Color::Black)).count() as f64;
        }
        let score = 0.5 + white / 80.0;
        if player == Color::White {
            score
        } else {
            1.0 - score
        }
    }
}

#[derive(Default)]
struct FakeClock {
    bits: AtomicU64,
}

impl FakeClock {
    fn now(&self) -> f64 {
        f64::from_bits(self.bits.load(Ordering::Relaxed))
    }
    fn advance(&self, seconds: f64) {
        self.bits
            .store((self.now() + seconds).to_bits(), Ordering::Relaxed);
    }
}

fn state(fen: &str) -> GameState {
    GameState::new(position_from_fen(fen).unwrap())
}

fn run(
    fen: &str,
    options: &Options,
    max_depth: usize,
    time_limit: f64,
    hard_limit: f64,
    tick: f64,
) -> (Option<String>, Stats, Memory) {
    let clock = FakeClock::default();
    let eval = Material {
        clock: &clock,
        tick,
    };
    let now = || clock.now();
    let control = Control {
        start: 0.0,
        hard_limit,
        clock: &now,
        stop: None,
        progress: None,
        rotation: 0,
    };
    let mut memory = Memory::default();
    let mut game = state(fen);
    let (best, stats) = iterative_deepening(
        &mut game,
        &eval,
        &mut memory,
        options,
        max_depth,
        time_limit,
        &control,
    )
    .unwrap();
    (best.as_ref().map(uci_string), stats, memory)
}

const START: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
const MATE_IN_ONE: &str = "6k1/5ppp/8/8/8/8/8/R3K3 w - - 0 1";
const MATE_IN_TWO: &str = "7k/8/5K2/8/8/8/8/6R1 w - - 0 1";

/// A fixed-depth search without quiescence.
fn fixed(fen: &str, depth: i32) -> SearchResult {
    let clock = FakeClock::default();
    let eval = Material {
        clock: &clock,
        tick: 0.0,
    };
    let now = || clock.now();
    let control = Control {
        start: 0.0,
        hard_limit: f64::INFINITY,
        clock: &now,
        stop: None,
        progress: None,
        rotation: 0,
    };
    alpha_beta(
        &mut state(fen),
        &eval,
        &mut Memory::default(),
        &Options {
            quiescence_plies: 0,
            ..Options::default()
        },
        depth,
        &control,
    )
}

#[test]
fn a_quicker_win_is_worth_more_than_a_slower_one() {
    let one = fixed(MATE_IN_ONE, 3);
    assert_eq!(one.best.as_ref().map(uci_string).as_deref(), Some("a1a8"));
    assert!((one.value - (1.0 - DEPTH_PREFERENCE_EPSILON)).abs() < 1e-12);
    let two = fixed(MATE_IN_TWO, 4);
    assert!(two.value > 0.99 && two.value < one.value, "{}", two.value);
}

#[test]
fn the_state_is_restored_after_every_search() {
    let mut game = state(START);
    let before = game.key();
    let clock = FakeClock::default();
    let eval = Material {
        clock: &clock,
        tick: 0.0,
    };
    let now = || clock.now();
    let control = Control {
        start: 0.0,
        hard_limit: f64::INFINITY,
        clock: &now,
        stop: None,
        progress: None,
        rotation: 0,
    };
    let options = Options {
        quiescence_plies: 4,
        ..Options::default()
    };
    alpha_beta(
        &mut game,
        &eval,
        &mut Memory::default(),
        &options,
        3,
        &control,
    );
    assert_eq!(game.key(), before);
}

#[test]
fn quiescence_sees_the_recapture_at_the_horizon() {
    // White can take a protected pawn with the queen: a depth-1 search without quiescence sees
    // only the gain, with quiescence it sees the loss of the queen.
    let fen = "4k3/8/3p4/4p3/8/8/4Q3/4K3 w - - 0 1";
    let plain = fixed(fen, 1);
    let with = {
        let clock = FakeClock::default();
        let eval = Material {
            clock: &clock,
            tick: 0.0,
        };
        let now = || clock.now();
        let control = Control {
            start: 0.0,
            hard_limit: f64::INFINITY,
            clock: &now,
            stop: None,
            progress: None,
            rotation: 0,
        };
        let options = Options {
            quiescence_plies: 4,
            ..Options::default()
        };
        alpha_beta(
            &mut state(fen),
            &eval,
            &mut Memory::default(),
            &options,
            1,
            &control,
        )
    };
    assert_ne!(
        plain.best.as_ref().map(uci_string),
        with.best.as_ref().map(uci_string)
    );
    assert_ne!(with.best.as_ref().map(uci_string).as_deref(), Some("e2e5"));
}

/// Fixed-depth search with quiescence of 4 plies and the given techniques.
fn quiescent(fen: &str, depth: i32, techniques: Techniques) -> SearchResult {
    let clock = FakeClock::default();
    let eval = Material {
        clock: &clock,
        tick: 0.0,
    };
    let now = || clock.now();
    let control = Control {
        start: 0.0,
        hard_limit: f64::INFINITY,
        clock: &now,
        stop: None,
        progress: None,
        rotation: 0,
    };
    let options = Options {
        quiescence_plies: 4,
        techniques,
        ..Options::default()
    };
    alpha_beta(
        &mut state(fen),
        &eval,
        &mut Memory::default(),
        &options,
        depth,
        &control,
    )
}

#[test]
fn quiescence_evasions_see_the_fork_after_a_check() {
    // Nc7+ forks king and queen. After it the quiescence search stands pat and sees a lost
    // queen for White; with evasions the king must move and Nxa8 follows (a pawn keeps the
    // ending from being a draw by insufficient material).
    let fen = "q3k3/8/8/1N6/8/8/7P/6K1 w - - 0 1";
    let plain = quiescent(fen, 1, Techniques::default());
    let with = quiescent(
        fen,
        1,
        Techniques {
            qs_evasions: true,
            ..Techniques::default()
        },
    );
    assert_eq!(with.best.as_ref().map(uci_string).as_deref(), Some("b5c7"));
    assert!(
        with.value > plain.value + 0.1,
        "{} {}",
        with.value,
        plain.value
    );
}

#[test]
fn the_horizon_table_entries_do_not_change_the_value() {
    for fen in [
        START,
        "4k3/8/3p4/4p3/8/8/4Q3/4K3 w - - 0 1",
        "r3k2r/8/8/3pP3/8/8/8/R3K2R w KQkq d6 0 1",
    ] {
        let plain = quiescent(fen, 3, Techniques::default());
        let with = quiescent(
            fen,
            3,
            Techniques {
                qs_tt: true,
                ..Techniques::default()
            },
        );
        assert!(
            (plain.value - with.value).abs() < 1e-9,
            "{fen}: {} {}",
            plain.value,
            with.value
        );
    }
}

#[test]
fn see_keeps_the_obvious_capture() {
    // A queen hangs: with or without SEE the search takes it.
    let fen = "4k3/8/8/3q4/8/8/3R4/4K3 w - - 0 1";
    for see in [false, true] {
        let result = quiescent(
            fen,
            3,
            Techniques {
                see,
                ..Techniques::default()
            },
        );
        assert_eq!(
            result.best.as_ref().map(uci_string).as_deref(),
            Some("d2d5")
        );
    }
}

#[test]
fn a_search_cut_by_the_clock_stores_nothing() {
    let clock = FakeClock::default();
    let eval = Material {
        clock: &clock,
        tick: 0.0,
    };
    let now = || clock.now();
    let control = Control {
        start: 0.0,
        hard_limit: -1.0,
        clock: &now,
        stop: None,
        progress: None,
        rotation: 0,
    };
    let mut memory = Memory::default();
    let result = alpha_beta(
        &mut state(START),
        &eval,
        &mut memory,
        &Options::default(),
        3,
        &control,
    );
    assert!(result.truncated && result.best.is_none());
    assert!(memory.table.is_empty());
}

#[test]
fn the_deepening_reports_the_depth_limit() {
    let options = Options {
        estimator: "never_stop".into(),
        ..Options::default()
    };
    let (best, stats, _) = run(START, &options, 4, 1e6, 1e6, 0.0);
    assert!(best.is_some());
    assert_eq!(
        (stats.completed_depth, stats.stop_reason),
        (4, StopReason::MaxDepth)
    );
    assert_eq!(stats.iterations.len(), 4);
    assert!(stats.iterations.iter().all(|it| !it.truncated));
}

#[test]
fn the_estimate_cannot_stop_the_deepening_before_min_depth_but_does_after() {
    // every evaluation takes a long fake time, so any estimate exceeds the budget
    let options = Options {
        estimator: "fixed_factor".into(),
        ..Options::default()
    };
    let (_, stats, _) = run(START, &options, 8, 1.0, 1e12, 1e-3);
    assert!(
        stats.completed_depth >= options.min_depth,
        "{}",
        stats.completed_depth
    );
    assert_eq!(stats.stop_reason, StopReason::Estimate, "{stats:?}");
    assert!(stats.stop_predicted > 0.0);
}

#[test]
fn an_exhausted_budget_still_completes_min_depth_and_then_stops() {
    let options = Options {
        estimator: "never_stop".into(),
        ..Options::default()
    };
    let (_, stats, _) = run(START, &options, 8, 0.0, 1e12, 1e-3);
    assert_eq!(
        (stats.completed_depth, stats.stop_reason),
        (options.min_depth, StopReason::Budget)
    );
}

#[test]
fn a_single_legal_move_is_played_after_the_first_iteration() {
    // Kxb2 is the only move; with the rule the deepening stops at once, without it it goes on.
    let fen = "k7/8/8/8/8/8/1q6/K7 w - - 0 1";
    let base = Options {
        estimator: "never_stop".into(),
        ..Options::default()
    };
    let single = Options {
        time_rules: TimeRules {
            single: true,
            ..TimeRules::default()
        },
        ..base.clone()
    };
    let (best, stats, _) = run(fen, &single, 6, 1e9, 1e9, 1e-3);
    assert_eq!(best.as_deref(), Some("a1b2"));
    assert_eq!(stats.completed_depth, 1, "{stats:?}");
    let (_, stats, _) = run(fen, &base, 6, 1e9, 1e9, 1e-3);
    assert_eq!(stats.completed_depth, 6, "{stats:?}");
}

#[test]
fn half_the_budget_stops_the_next_iteration() {
    let half = Options {
        estimator: "never_stop".into(),
        time_rules: TimeRules {
            half: true,
            ..TimeRules::default()
        },
        ..Options::default()
    };
    // depth 1 and 2 take 0.063 fake seconds in total: past half of 0.1, but inside the whole
    let (_, with, _) = run(START, &half, 8, 0.1, 1e9, 1e-3);
    let (_, without, _) = run(
        START,
        &Options {
            time_rules: TimeRules::default(),
            ..half.clone()
        },
        8,
        0.1,
        1e9,
        1e-3,
    );
    assert!(
        with.completed_depth < without.completed_depth,
        "{with:?} {without:?}"
    );
}

#[test]
fn the_stability_and_node_factors_scale_the_budget() {
    let stab = TimeRules::from_names(&["stab"]).unwrap();
    assert!((stab.scale(0, 0.0) - 1.2).abs() < 1e-9);
    assert!((stab.scale(5, 0.0) - 1.0).abs() < 1e-9);
    assert!((stab.scale(50, 0.0) - 0.8).abs() < 1e-9);
    let nodes = TimeRules::from_names(&["nodes"]).unwrap();
    assert!((nodes.scale(0, 0.9) - 0.6).abs() < 1e-9);
    assert!((nodes.scale(0, 0.99) - 0.5).abs() < 1e-9);
    assert!((nodes.scale(0, 0.2) - 2.0).abs() < 1e-9);
    assert_eq!(TimeRules::default().scale(9, 0.1), 1.0);
    assert!(TimeRules::from_names(&["nope"]).is_err());
}

#[test]
fn an_iteration_cut_by_the_hard_limit_is_discarded() {
    let options = Options {
        estimator: "never_stop".into(),
        ..Options::default()
    };
    // depth 1 and 2 finish within 0.15 fake seconds, depth 3 does not
    let (best, stats, _) = run(START, &options, 6, 1e9, 0.15, 1e-3);
    assert_eq!(stats.stop_reason, StopReason::Truncated);
    assert_eq!(stats.completed_depth, 2, "{stats:?}");
    assert!(stats.iterations.last().unwrap().truncated);
    assert!(best.is_some() && !stats.truncated);
}

#[test]
fn the_first_iteration_may_be_partial_when_nothing_else_exists() {
    let options = Options::default();
    let (best, stats, _) = run(START, &options, 6, 1e9, -1.0, 0.0);
    assert!(best.is_none() && stats.truncated && stats.completed_depth == 0);
}

#[test]
fn several_threads_return_a_legal_move_and_do_not_lose_nodes() {
    let positions = [
        START,
        "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
        "r3k2r/pp3ppp/2n1bn2/q1pp4/3P4/2N1PN2/PP2BPPP/R2QK2R w KQkq - 0 1",
        "8/2k5/8/3Pp3/8/8/8/4K3 w - e6 0 1",
    ];
    for fen in positions {
        let clock = FakeClock::default();
        let eval = Material {
            clock: &clock,
            tick: 0.0,
        };
        let now = || clock.now();
        let control = Control {
            start: 0.0,
            hard_limit: 1e9,
            clock: &now,
            stop: None,
            progress: None,
            rotation: 0,
        };
        let options = Options {
            estimator: "never_stop".into(),
            techniques: Techniques {
                lmr: false,
                ..Techniques::all()
            },
            ..Options::default()
        };
        let mut memory = Memory::with_threads(1000, 4, 1);
        let mut game = state(fen);
        let (best, stats) =
            iterative_deepening(&mut game, &eval, &mut memory, &options, 4, 1e9, &control).unwrap();
        let legal = crate::board::ordered_moves(game.pos());
        assert!(
            legal
                .iter()
                .any(|m| Some(uci_string(m)) == best.as_ref().map(uci_string)),
            "{fen}"
        );
        assert!(stats.nodes_visited > 0);
    }
}

mod incremental {
    use crate::board::{position_from_fen, GameState};
    use crate::eval::sparse_features::{FeatureSet, Perspective};
    use crate::eval::sparse_net::{Layer, Net};
    use crate::search::*;

    fn layer(fan_in: usize, fan_out: usize, seed: u64) -> Layer {
        let mut state = seed;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) as f64 / (1u64 << 31) as f64) - 0.5
        };
        Layer {
            fan_in,
            fan_out,
            weights: (0..fan_in * fan_out).map(|_| next() * 0.3).collect(),
            bias: (0..fan_out).map(|_| next() * 0.2).collect(),
        }
    }

    #[test]
    fn the_incremental_evaluation_gives_the_same_search() {
        let net = Net::new(
            FeatureSet::Slotted,
            Perspective::SideToMove,
            vec![layer(768, 32, 1), layer(32, 16, 2), layer(16, 1, 3)],
        )
        .unwrap();
        let fens = [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
            "r3k2r/pp3ppp/2n1bn2/q1pp4/3P4/2N1PN2/PP2BPPP/R2QK2R w KQkq - 0 1",
            "8/2k5/8/3Pp3/8/8/8/4K3 w - e6 0 1",
        ];
        for fen in fens {
            let mut found = Vec::new();
            for incremental in [false, true] {
                let options = Options {
                    quiescence_plies: 4,
                    techniques: Techniques {
                        incremental,
                        ..Techniques::default()
                    },
                    ..Options::default()
                };
                let clock = || 0.0;
                let control = Control {
                    start: 0.0,
                    hard_limit: f64::INFINITY,
                    clock: &clock,
                    stop: None,
                    progress: None,
                    rotation: 0,
                };
                let mut game = GameState::new(position_from_fen(fen).unwrap());
                found.push(alpha_beta(
                    &mut game,
                    &net,
                    &mut Memory::default(),
                    &options,
                    3,
                    &control,
                ));
            }
            assert!((found[0].value - found[1].value).abs() < 1e-9, "{fen}");
            assert_eq!(found[0].best, found[1].best, "{fen}");
            // a different rounding can break a tie between equal values: the counts agree to 1 %
            let (a, b) = (found[0].nodes as f64, found[1].nodes as f64);
            assert!((a - b).abs() <= 0.01 * a, "{fen}: {a} nodes against {b}");
        }
    }
}
