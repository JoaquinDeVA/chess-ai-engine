//! Tests of the search.

use std::sync::atomic::{AtomicU64, Ordering};

use shakmaty::Color;

use super::iteration_time::{fixed_factor, never_stop};
use super::pruning::has_pieces;
use super::*;
use crate::board::{position_from_fen, uci_string, GameState};
use crate::eval::FeatureSet;
use crate::eval::{Layer, Net};

/// A network that sees only material: the side to move's expected score is
/// `sigmoid((own - opponent) / 20)`, with pawn 1, knight and bishop 3, rook 5 and queen 9.
fn material() -> Net {
    let value = |role: usize| [1.0, 3.0, 3.0, 5.0, 9.0, 0.0][role];
    // First layer 768 -> 2: unit 0 adds the material of the side to move, unit 1 the opponent's
    // (inputs 0..384 and 384..768, 64 squares per piece type), divided by 40 to stay below the
    // clip at 1.
    let mut weights = vec![0.0; 768 * 2];
    for input in 0..768 {
        let (unit, role) = (input / 384, (input % 384) / 64);
        weights[input * 2 + unit] = value(role) / 40.0;
    }
    let first = Layer {
        fan_in: 768,
        fan_out: 2,
        weights,
        bias: vec![0.0, 0.0],
    };
    let last = Layer {
        fan_in: 2,
        fan_out: 1,
        weights: vec![2.0, -2.0],
        bias: vec![0.0],
    };
    Net::new(FeatureSet::Slotted, vec![first, last]).unwrap()
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

fn no_time() -> f64 {
    0.0
}

/// A control of one search with no stop flag and no helpers.
fn control(hard_limit: f64, clock: Clock<'_>) -> Control<'_> {
    Control {
        start: 0.0,
        hard_limit,
        clock,
        stop: None,
        progress: None,
        rotation: 0,
    }
}

fn state(fen: &str) -> GameState {
    GameState::new(position_from_fen(fen).unwrap())
}

/// Iterative deepening with the material network; every reading of the clock advances it by
/// `tick` seconds.
fn run(
    fen: &str,
    options: &Options,
    max_depth: usize,
    time_limit: f64,
    hard_limit: f64,
    tick: f64,
) -> (Option<String>, Stats) {
    let clock = FakeClock::default();
    let now = || {
        clock.advance(tick);
        clock.now()
    };
    let (best, stats) = iterative_deepening(
        &mut state(fen),
        &material(),
        &mut Memory::default(),
        options,
        max_depth,
        time_limit,
        &control(hard_limit, &now),
    )
    .unwrap();
    (best.as_ref().map(uci_string), stats)
}

/// One search of `depth` plies with the material network, `quiescence_plies` and `techniques`.
fn search(fen: &str, depth: i32, quiescence_plies: u32, techniques: Techniques) -> SearchResult {
    let options = Options {
        quiescence_plies,
        techniques,
        ..Options::default()
    };
    alpha_beta(
        &mut state(fen),
        &material(),
        &mut Memory::default(),
        &options,
        depth,
        &control(f64::INFINITY, &no_time),
    )
}

/// A fixed-depth search without quiescence.
fn fixed(fen: &str, depth: i32) -> SearchResult {
    search(fen, depth, 0, Techniques::default())
}

/// Fixed-depth search with quiescence of 4 plies and the given techniques.
fn quiescent(fen: &str, depth: i32, techniques: Techniques) -> SearchResult {
    search(fen, depth, 4, techniques)
}

const START: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
const MATE_IN_ONE: &str = "6k1/5ppp/8/8/8/8/8/R3K3 w - - 0 1";
const MATE_IN_TWO: &str = "7k/8/5K2/8/8/8/8/6R1 w - - 0 1";
const HANGING_QUEEN: &str = "4k3/8/8/3q4/8/8/3R4/4K3 w - - 0 1";

#[test]
fn a_quicker_win_is_worth_more_than_a_slower_one() {
    let one = fixed(MATE_IN_ONE, 3);
    assert_eq!(one.best.as_ref().map(uci_string).as_deref(), Some("a1a8"));
    assert!((one.value - (1.0 - DEPTH_PREFERENCE_EPSILON)).abs() < 1e-12);
    let two = fixed(MATE_IN_TWO, 4);
    assert!(two.value > 0.99 && two.value < one.value, "{}", two.value);
}

#[test]
fn a_mate_keeps_its_distance_when_it_comes_back_from_the_table() {
    // The deepening reads the mates of the previous iteration from the table at other plies.
    let two = fixed(MATE_IN_TWO, 6);
    assert!(
        (two.value - (1.0 - 3.0 * DEPTH_PREFERENCE_EPSILON)).abs() < 1e-12,
        "{}",
        two.value
    );
}

#[test]
fn an_entry_that_the_fifty_move_clock_could_change_is_not_stored() {
    let stored = |fen: &str| {
        let mut game = state(fen);
        let key = game.key();
        let mut memory = Memory::default();
        alpha_beta(
            &mut game,
            &material(),
            &mut memory,
            &Options::default(),
            4,
            &control(f64::INFINITY, &no_time),
        );
        memory.table.get(&key).is_some()
    };
    assert!(stored("4k3/8/8/8/8/8/8/R3K3 w - - 0 1"));
    assert!(stored("4k3/8/8/8/8/8/8/R3K3 w - - 90 80"));
    assert!(!stored("4k3/8/8/8/8/8/8/R3K3 w - - 97 80"));
}

#[test]
fn the_state_is_restored_after_every_search() {
    let mut game = state(START);
    let before = game.key();
    let options = Options {
        quiescence_plies: 4,
        ..Options::default()
    };
    alpha_beta(
        &mut game,
        &material(),
        &mut Memory::default(),
        &options,
        3,
        &control(f64::INFINITY, &no_time),
    );
    assert_eq!(game.key(), before);
}

#[test]
fn quiescence_sees_the_recapture_at_the_horizon() {
    // White can take a protected pawn with the queen: a depth-1 search without quiescence sees
    // only the gain, with quiescence it sees the loss of the queen.
    let fen = "4k3/8/3p4/4p3/8/8/4Q3/4K3 w - - 0 1";
    let plain = fixed(fen, 1);
    let with = quiescent(fen, 1, Techniques::default());
    assert_ne!(
        plain.best.as_ref().map(uci_string),
        with.best.as_ref().map(uci_string)
    );
    assert_ne!(with.best.as_ref().map(uci_string).as_deref(), Some("e2e5"));
}

#[test]
fn see_keeps_the_obvious_capture() {
    // A queen hangs: the search takes it.
    let result = quiescent(HANGING_QUEEN, 3, Techniques::default());
    assert_eq!(
        result.best.as_ref().map(uci_string).as_deref(),
        Some("d2d5")
    );
}

#[test]
fn the_null_move_is_not_tried_with_only_pawns_and_the_king() {
    // Lasker's zugzwang position: passing the turn would look winning, but nobody may pass.
    let lasker = position_from_fen("8/k7/3p4/p2P1p2/P2P1P2/8/8/K7 w - - 0 1").unwrap();
    assert!(!has_pieces(&lasker, Color::White));
    assert!(!has_pieces(&lasker, Color::Black));
    let start = position_from_fen(START).unwrap();
    assert!(has_pieces(&start, Color::White));
}

#[test]
fn a_search_cut_by_the_clock_stores_nothing() {
    let mut memory = Memory::default();
    let result = alpha_beta(
        &mut state(START),
        &material(),
        &mut memory,
        &Options::default(),
        3,
        &control(-1.0, &no_time),
    );
    assert!(result.truncated && result.best.is_none());
    assert!(memory.table.is_empty());
}

#[test]
fn the_deepening_reports_the_depth_limit() {
    let options = Options {
        estimator: never_stop,
        ..Options::default()
    };
    let (best, stats) = run(START, &options, 4, 1e6, 1e6, 0.0);
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
    // Every reading of the clock takes 0.1 fake seconds, so after the first iteration the
    // estimate of the next one exceeds what is left of the budget.
    let options = Options {
        estimator: fixed_factor,
        ..Options::default()
    };
    let (_, stats) = run(START, &options, 8, 1.0, 1e12, 0.1);
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
        estimator: never_stop,
        ..Options::default()
    };
    let (_, stats) = run(START, &options, 8, 0.0, 1e12, 1e-3);
    assert_eq!(
        (stats.completed_depth, stats.stop_reason),
        (options.min_depth, StopReason::Budget)
    );
}

#[test]
fn an_iteration_cut_by_the_hard_limit_is_discarded() {
    let options = Options {
        estimator: never_stop,
        ..Options::default()
    };
    // Every reading of the clock takes 0.1 fake seconds, and an iteration of under 2,048 nodes
    // reads it about five times (the deepening's own readings and the first node): depths 1 to 4
    // end before 2.2 seconds and depth 5 passes the limit at its first node.
    let (best, stats) = run(START, &options, 6, 1e9, 2.2, 0.1);
    assert_eq!(stats.stop_reason, StopReason::Truncated);
    assert_eq!(stats.completed_depth, 4, "{stats:?}");
    assert!(stats.iterations.last().unwrap().truncated);
    assert!(best.is_some() && !stats.truncated);
}

#[test]
fn the_first_iteration_may_be_partial_when_nothing_else_exists() {
    let options = Options::default();
    let (best, stats) = run(START, &options, 6, 1e9, -1.0, 0.0);
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
        let options = Options {
            estimator: never_stop,
            techniques: Techniques::all(),
            ..Options::default()
        };
        let mut memory = Memory::with_threads(1000, 4, 1);
        let mut game = state(fen);
        let (best, stats) = iterative_deepening(
            &mut game,
            &material(),
            &mut memory,
            &options,
            4,
            1e9,
            &control(1e9, &no_time),
        )
        .unwrap();
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

#[test]
fn late_move_reductions_follow_the_depth_and_the_position_in_the_order() {
    use super::pruning::reduction;
    assert_eq!(reduction(3, 2, false), 0, "too early in the order");
    assert_eq!(reduction(2, 9, false), 0, "too shallow");
    assert_eq!(reduction(3, 3, false), 1);
    assert_eq!(reduction(6, 10, false), 2);
    assert_eq!(reduction(6, 10, true), 1, "good history: one ply less");
    assert_eq!(reduction(3, 100, false), 1, "never below one ply of search");
}

#[test]
fn late_move_reductions_still_find_the_forced_mate() {
    let with = Techniques {
        lmr: true,
        ..Techniques::all()
    };
    let result = quiescent(MATE_IN_TWO, 6, with);
    let expected = 1.0 - 3.0 * DEPTH_PREFERENCE_EPSILON;
    assert!((result.value - expected).abs() < 1e-12, "{}", result.value);
}

#[test]
fn reverse_futility_keeps_the_forced_mate_and_the_hanging_queen() {
    let with = Techniques {
        rfp: true,
        ..Techniques::all()
    };
    let result = quiescent(MATE_IN_TWO, 6, with);
    let expected = 1.0 - 3.0 * DEPTH_PREFERENCE_EPSILON;
    assert!((result.value - expected).abs() < 1e-12, "{}", result.value);
    let queen = quiescent(HANGING_QUEEN, 3, with);
    let without = quiescent(HANGING_QUEEN, 3, Techniques::all());
    assert_eq!(queen.best, without.best);
}

#[test]
fn the_reverse_futility_margin_means_the_same_at_every_level() {
    use super::pruning::log_odds;
    // 0.5 -> 0.61 and 0.9 -> 0.934 are the same distance in log-odds (about 0.46).
    assert!((log_odds(0.61) - log_odds(0.5) - 0.447).abs() < 0.02);
    assert!((log_odds(0.934) - log_odds(0.9) - 0.472).abs() < 0.03);
}

#[test]
fn shallow_pruning_keeps_the_forced_mate_and_the_hanging_queen() {
    let with = Techniques {
        futility: true,
        lmp: true,
        ..Techniques::all()
    };
    let result = quiescent(MATE_IN_TWO, 6, with);
    let expected = 1.0 - 3.0 * DEPTH_PREFERENCE_EPSILON;
    assert!((result.value - expected).abs() < 1e-12, "{}", result.value);
    assert_eq!(
        quiescent(HANGING_QUEEN, 3, with).best,
        quiescent(HANGING_QUEEN, 3, Techniques::default()).best
    );
}

#[test]
fn late_move_pruning_allows_more_quiet_moves_the_deeper_the_node() {
    use super::pruning::late_move_limit;
    assert_eq!(late_move_limit(1), 4);
    assert_eq!(late_move_limit(2), 7);
    assert_eq!(late_move_limit(3), 12);
}

#[test]
fn internal_iterative_reduction_keeps_the_forced_mate_and_the_hanging_queen() {
    let with = Techniques::all();
    let without = Techniques {
        iir: false,
        ..Techniques::all()
    };
    let result = quiescent(MATE_IN_TWO, 6, with);
    let expected = 1.0 - 3.0 * DEPTH_PREFERENCE_EPSILON;
    assert!((result.value - expected).abs() < 1e-12, "{}", result.value);
    assert_eq!(
        quiescent(HANGING_QUEEN, 5, with).best,
        quiescent(HANGING_QUEEN, 5, without).best
    );
}
