//! What the tests share: a material-only network, a fake clock and ways to run a search.

use std::sync::atomic::{AtomicU64, Ordering};

use crate::board::{position_from_fen, uci_string, GameState};
use crate::eval::{FeatureSet, Layer, Net};
use crate::search::*;

/// A network that sees only material: the side to move's expected score is
/// `sigmoid((own - opponent) / 20)`, with pawn 1, knight and bishop 3, rook 5 and queen 9.
pub(super) fn material() -> Net {
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

pub(super) fn no_time() -> f64 {
    0.0
}

/// A control of one search with no stop flag and no helpers.
pub(super) fn control(hard_limit: f64, clock: Clock<'_>) -> Control<'_> {
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

pub(super) fn state(fen: &str) -> GameState {
    GameState::new(position_from_fen(fen).unwrap())
}

/// Iterative deepening with the material network; every reading of the clock advances it by
/// `tick` seconds.
pub(super) fn run(
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
pub(super) fn search(fen: &str, depth: i32, quiescence_plies: u32) -> SearchResult {
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
pub(super) fn fixed(fen: &str, depth: i32) -> SearchResult {
    search(fen, depth, 0)
}

/// Fixed-depth search with quiescence of 4 plies.
pub(super) fn quiescent(fen: &str, depth: i32) -> SearchResult {
    search(fen, depth, 4)
}

pub(super) const START: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
pub(super) const MATE_IN_ONE: &str = "6k1/5ppp/8/8/8/8/8/R3K3 w - - 0 1";
pub(super) const MATE_IN_TWO: &str = "7k/8/5K2/8/8/8/8/6R1 w - - 0 1";
pub(super) const HANGING_QUEEN: &str = "4k3/8/8/3q4/8/8/3R4/4K3 w - - 0 1";

/// Ids of the legal moves of `fen` whose UCI string is in `names`.
pub(super) fn ids(fen: &str, names: &[&str]) -> Vec<u32> {
    let state = state(fen);
    crate::board::ordered_moves(state.pos())
        .iter()
        .filter(|m| names.contains(&uci_string(m).as_str()))
        .map(crate::board::move_id)
        .collect()
}
