//! Tests of the search.

use std::sync::atomic::{AtomicU64, Ordering};

use shakmaty::Color;

use super::search_techniques::has_pieces;
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
        dynamic_hard_limit: None,
        dynamic_soft_limit: None,
        dynamic_budget_start: None,
        clock,
        stop: None,
        progress: None,
        rotation: 0,
    }
}

#[test]
fn dynamic_limits_are_visible_to_an_active_search() {
    let hard = AtomicU64::new(f64::INFINITY.to_bits());
    let soft = AtomicU64::new(f64::INFINITY.to_bits());
    let control = Control {
        start: 0.0,
        hard_limit: f64::INFINITY,
        dynamic_hard_limit: Some(&hard),
        dynamic_soft_limit: Some(&soft),
        dynamic_budget_start: None,
        clock: &no_time,
        stop: None,
        progress: None,
        rotation: 0,
    };
    hard.store(2.0f64.to_bits(), Ordering::Release);
    soft.store(1.0f64.to_bits(), Ordering::Release);
    assert_eq!(control.current_hard_limit(), 2.0);
    assert_eq!(control.current_soft_limit(f64::INFINITY), 1.0);
}

#[test]
fn the_scale_applies_to_the_budget_and_not_to_the_time_before_it() {
    let hard = AtomicU64::new(f64::INFINITY.to_bits());
    let soft = AtomicU64::new(3.8f64.to_bits());
    let begin = AtomicU64::new(2.3f64.to_bits());
    let control = Control {
        start: 0.0,
        hard_limit: f64::INFINITY,
        dynamic_hard_limit: Some(&hard),
        dynamic_soft_limit: Some(&soft),
        dynamic_budget_start: Some(&begin),
        clock: &no_time,
        stop: None,
        progress: None,
        rotation: 0,
    };
    // Budget 1.5 s from second 2.3.
    assert!((control.scaled_soft_limit(f64::INFINITY, 1.0) - 3.8).abs() < 1e-9);
    assert!((control.scaled_soft_limit(f64::INFINITY, 2.0) - 5.3).abs() < 1e-9);
    assert!((control.scaled_soft_limit(f64::INFINITY, 0.5) - 3.05).abs() < 1e-9);
    // Without a start the whole limit is the budget.
    let plain = Control {
        dynamic_budget_start: None,
        ..control
    };
    assert!((plain.scaled_soft_limit(f64::INFINITY, 0.5) - 1.9).abs() < 1e-9);
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

/// One search of `depth` plies with the material network and `quiescence_plies`.
fn search(fen: &str, depth: i32, quiescence_plies: u32) -> SearchResult {
    let options = Options {
        quiescence_plies,
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
    search(fen, depth, 0)
}

/// Fixed-depth search with quiescence of 4 plies.
fn quiescent(fen: &str, depth: i32) -> SearchResult {
    search(fen, depth, 4)
}

const START: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
const MATE_IN_ONE: &str = "6k1/5ppp/8/8/8/8/8/R3K3 w - - 0 1";
const MATE_IN_TWO: &str = "7k/8/5K2/8/8/8/8/6R1 w - - 0 1";
const HANGING_QUEEN: &str = "4k3/8/8/3q4/8/8/3R4/4K3 w - - 0 1";

#[test]
fn a_quicker_win_is_worth_more_than_a_slower_one() {
    let one = fixed(MATE_IN_ONE, 3);
    assert_eq!(one.best.as_ref().map(uci_string).as_deref(), Some("a1a8"));
    assert_eq!(one.value, MATE - 1);
    // Depth 6, not 4: late move reductions search the quiet mating rook move a ply shallower.
    let two = fixed(MATE_IN_TWO, 6);
    assert!(
        two.value > MATE_BOUND && two.value < one.value,
        "{}",
        two.value
    );
}

#[test]
#[cfg(feature = "search-summary")]
fn check_extends_the_search_by_one_ply() {
    let result = fixed("4k3/8/8/8/8/8/4r3/4K3 w - - 0 1", 1);
    assert!(result.best.is_some());
    assert!(result.report.max_ply >= 2);
}

#[test]
fn quiescence_searches_evasions_when_in_check() {
    let result = quiescent("4k3/8/8/8/8/8/4r3/4K3 w - - 0 1", 0);
    assert!(result.nodes > 1);
}

#[test]
fn a_mate_keeps_its_distance_when_it_comes_back_from_the_table() {
    // The deepening reads the mates of the previous iteration from the table at other plies.
    let two = fixed(MATE_IN_TWO, 6);
    assert_eq!(two.value, MATE - 3);
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
    let with = quiescent(fen, 1);
    assert_ne!(
        plain.best.as_ref().map(uci_string),
        with.best.as_ref().map(uci_string)
    );
    assert_ne!(with.best.as_ref().map(uci_string).as_deref(), Some("e2e5"));
}

#[test]
fn see_keeps_the_obvious_capture() {
    // A queen hangs: the search takes it.
    let result = quiescent(HANGING_QUEEN, 3);
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
    let options = Options::default();
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
fn an_exhausted_budget_still_completes_min_depth_and_then_stops() {
    let options = Options::default();
    let (_, stats) = run(START, &options, 8, 0.0, 1e12, 1e-3);
    assert_eq!(
        (stats.completed_depth, stats.stop_reason),
        (options.min_depth, StopReason::Budget)
    );
}

#[test]
fn an_iteration_the_hard_limit_would_cut_is_not_started() {
    let options = Options::default();
    // Every reading of the clock takes 0.1 fake seconds, and an iteration of under 2,048 nodes
    // reads it about four times: depths 1 to 5 end before 2.2 seconds. After the last one, 3.5
    // times its duration no longer fits under the limit, so depth 6 is not started and nothing
    // is thrown away.
    let (best, stats) = run(START, &options, 6, 1e9, 2.2, 0.1);
    assert_eq!(stats.stop_reason, StopReason::Budget);
    assert!(stats.completed_depth >= options.min_depth, "{stats:?}");
    assert!(stats.iterations.iter().all(|it| !it.truncated));
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
        let options = Options::default();
        let mut memory = Memory::new(4, 1);
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
    use super::search_techniques::late_move_reduction;
    assert_eq!(
        late_move_reduction(3, 2, false),
        0,
        "too early in the order"
    );
    assert_eq!(late_move_reduction(2, 9, false), 0, "too shallow");
    assert_eq!(late_move_reduction(3, 3, false), 1);
    assert_eq!(late_move_reduction(6, 10, false), 2);
    assert_eq!(
        late_move_reduction(6, 10, true),
        1,
        "favorable history: one ply less"
    );
    assert_eq!(
        late_move_reduction(3, 100, false),
        1,
        "never below one ply of search"
    );
}

#[test]
fn the_search_techniques_keep_the_forced_mate_and_the_hanging_queen() {
    // LMR, RFP, IIR and futility all work in these searches: none may lose a mate in two or a
    // free queen.
    assert_eq!(quiescent(MATE_IN_TWO, 6).value, MATE - 3);
    for depth in [3, 5] {
        assert_eq!(
            quiescent(HANGING_QUEEN, depth)
                .best
                .as_ref()
                .map(uci_string)
                .as_deref(),
            Some("d2d5")
        );
    }
}

#[test]
fn an_iteration_that_would_end_far_past_the_soft_limit_is_not_started() {
    let options = Options::default();
    // Same fake clock, a soft limit of 0.2 s and a far hard limit: after min_depth the soft limit
    // is passed, so the search stops there, and no iteration is cut.
    let (best, stats) = run(START, &options, 8, 0.2, 1e9, 0.1);
    assert_eq!(stats.stop_reason, StopReason::Budget);
    assert!(stats.iterations.iter().all(|it| !it.truncated));
    assert!(best.is_some());
}

/// Ids of the legal moves of `fen` whose UCI string is in `names`.
fn ids(fen: &str, names: &[&str]) -> Vec<u32> {
    let state = state(fen);
    crate::board::ordered_moves(state.pos())
        .iter()
        .filter(|m| names.contains(&uci_string(m).as_str()))
        .map(crate::board::move_id)
        .collect()
}

#[test]
fn the_root_only_searches_the_moves_it_is_given() {
    let fen = MATE_IN_ONE;
    let search_with = |root_moves: Option<Vec<u32>>| {
        let options = Options {
            root_moves,
            ..Options::default()
        };
        alpha_beta(
            &mut state(fen),
            &material(),
            &mut Memory::default(),
            &options,
            2,
            &control(f64::INFINITY, &no_time),
        )
        .best
        .as_ref()
        .map(uci_string)
    };
    assert_eq!(search_with(None).as_deref(), Some("a1a8"));
    // Mate is left out of the list, so another listed move is played.
    let allowed = ids(fen, &["a1a2", "a1b1"]);
    assert_eq!(allowed.len(), 2);
    let best = search_with(Some(allowed)).unwrap();
    assert!(best == "a1a2" || best == "a1b1");
    // A list that matches no legal move is ignored.
    assert_eq!(search_with(Some(vec![u32::MAX])).as_deref(), Some("a1a8"));
}

#[test]
fn a_table_entry_does_not_bring_back_a_move_the_root_list_excludes() {
    let fen = MATE_IN_ONE;
    let mut memory = Memory::default();
    let run = |memory: &mut Memory, root_moves: Option<Vec<u32>>| {
        let options = Options {
            root_moves,
            ..Options::default()
        };
        alpha_beta(
            &mut state(fen),
            &material(),
            memory,
            &options,
            3,
            &control(f64::INFINITY, &no_time),
        )
        .best
        .as_ref()
        .map(uci_string)
    };
    // The unrestricted search stores the mate in the table.
    assert_eq!(run(&mut memory, None).as_deref(), Some("a1a8"));
    let best = run(&mut memory, Some(ids(fen, &["a1a2", "a1b1"]))).unwrap();
    assert!(best == "a1a2" || best == "a1b1", "{best}");
}

#[test]
fn the_null_move_reduction_grows_with_the_depth_and_the_margin() {
    use super::search_techniques::null_move_reduction;
    assert_eq!(null_move_reduction(3, 0), 2);
    assert_eq!(null_move_reduction(4, 0), 3);
    assert_eq!(null_move_reduction(8, 0), 4);
    assert_eq!(null_move_reduction(3, 200), 3);
    assert_eq!(
        null_move_reduction(3, 5_000),
        4,
        "the margin adds two plies at most"
    );
}
