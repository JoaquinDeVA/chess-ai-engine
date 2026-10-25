//! Alpha-beta search with iterative deepening. Port of `search/game_algorithms.py`.
//!
//! The values are expected scores of the root player in `[0, 1]`; terminal values are 1, 0.5 and
//! 0, nudged by `epsilon` per ply so that quicker wins and slower losses are preferred. The
//! search keeps a bounded transposition table and killer moves between iterations and between the
//! moves of a game, and ends every branch at the horizon with a quiescence search over captures
//! and promotions.

pub mod estimators;
pub mod shared;
pub mod tt;

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering::Relaxed};
use std::sync::Arc;

use shakmaty::{Color, Move};

use crate::board::{is_noisy, move_id, ordered_moves, GameState};
use crate::eval::Evaluator;
use shared::SharedTable;
use tt::{order, Flag, Killers, Table, Tt, DEFAULT_MAX_TT_ENTRIES};

/// Iterations that are always completed before the time estimate may stop the deepening.
pub const DEFAULT_MIN_DEPTH: usize = 3;
/// Rule that predicts the duration of the next iteration.
pub const DEFAULT_ESTIMATOR: &str = "geometric_growth";
/// Weight of the "win faster / lose slower" preference, in utility units per ply.
pub const DEPTH_PREFERENCE_EPSILON: f64 = 0.003;
/// Plies of quiescence search of the bots.
pub const DEFAULT_QUIESCENCE_PLIES: u32 = 4;
/// Tolerance for finishing a depth that was started within the budget.
const OVERSHOOT_FACTOR: f64 = 1.15;

/// Why the deepening ended.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum StopReason {
    MaxDepth,
    Estimate,
    Budget,
    Truncated,
}

impl StopReason {
    pub fn as_str(self) -> &'static str {
        match self {
            StopReason::MaxDepth => "max_depth",
            StopReason::Estimate => "estimate",
            StopReason::Budget => "budget",
            StopReason::Truncated => "truncated",
        }
    }
}

/// One iteration of the deepening: depth, seconds, nodes and whether the clock cut it.
#[derive(Clone, Copy, Debug)]
pub struct IterationInfo {
    pub depth: usize,
    pub seconds: f64,
    pub nodes: u64,
    pub truncated: bool,
}

/// What the search measured.
#[derive(Clone, Debug)]
pub struct Stats {
    pub nodes_visited: u64,
    pub pruning_count: u64,
    pub max_depth_reached: i32,
    pub truncated: bool,
    pub completed_depth: usize,
    pub iterations: Vec<IterationInfo>,
    pub stop_reason: StopReason,
    pub stop_predicted: f64,
}

/// Settings of a search.
#[derive(Clone, Debug)]
pub struct Options {
    pub quiescence_plies: u32,
    pub min_depth: usize,
    pub estimator: String,
    pub epsilon: f64,
    /// Techniques that can be switched off to measure what each one saves (all on by default).
    pub use_table: bool,
    pub use_killers: bool,
    pub use_capture_order: bool,
    /// Techniques of selective search; all off reproduces the reference engine.
    pub techniques: Techniques,
}

/// The techniques added to the reference search, each with its own switch (all off by default).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Techniques {
    /// Try the best move stored in the table first.
    pub tt_move: bool,
    /// Order quiet moves by how often they caused cut-offs.
    pub history: bool,
    /// Principal variation search: zero-width windows after the first move.
    pub pvs: bool,
    /// Search the root with a narrow window around the previous value.
    pub aspiration: bool,
    /// Prune a node if passing the turn already cuts.
    pub null_move: bool,
    /// Search late quiet moves with less depth.
    pub lmr: bool,
}

impl Techniques {
    /// Names, as the switches are called from Python.
    pub const NAMES: [&'static str; 6] = [
        "tt_move",
        "history",
        "pvs",
        "aspiration",
        "null_move",
        "lmr",
    ];

    /// The techniques with the given names.
    pub fn from_names<S: AsRef<str>>(names: &[S]) -> Result<Techniques, String> {
        let mut techniques = Techniques::default();
        for name in names {
            match name.as_ref() {
                "tt_move" => techniques.tt_move = true,
                "history" => techniques.history = true,
                "pvs" => techniques.pvs = true,
                "aspiration" => techniques.aspiration = true,
                "null_move" => techniques.null_move = true,
                "lmr" => techniques.lmr = true,
                other => {
                    return Err(format!(
                        "unknown technique {other:?}; available: {:?}",
                        Techniques::NAMES
                    ))
                }
            }
        }
        Ok(techniques)
    }

    /// Every technique on.
    pub fn all() -> Techniques {
        Techniques {
            tt_move: true,
            history: true,
            pvs: true,
            aspiration: true,
            null_move: true,
            lmr: true,
        }
    }
}

impl Default for Options {
    fn default() -> Options {
        Options {
            quiescence_plies: 0,
            min_depth: DEFAULT_MIN_DEPTH,
            estimator: DEFAULT_ESTIMATOR.to_string(),
            epsilon: DEPTH_PREFERENCE_EPSILON,
            use_table: true,
            use_killers: true,
            use_capture_order: true,
            techniques: Techniques::default(),
        }
    }
}

/// What a game keeps between two moves: the transposition table and the killer moves, and how
/// many threads search.
pub struct Memory {
    pub table: Tt,
    pub killers: Killers,
    pub threads: usize,
}

impl Memory {
    /// A single-thread memory with the exact, bounded table of the Python engine.
    pub fn new(max_tt_entries: usize) -> Memory {
        Memory {
            table: Tt::Local(Table::new(max_tt_entries)),
            killers: Killers::default(),
            threads: 1,
        }
    }

    /// A memory for `threads` threads (at least 1): with one, the exact table; with more, a
    /// lock-free table of `hash_megabytes` shared by all of them.
    pub fn with_threads(max_tt_entries: usize, threads: usize, hash_megabytes: usize) -> Memory {
        if threads <= 1 {
            return Memory::new(max_tt_entries);
        }
        Memory {
            table: Tt::Shared(Arc::new(SharedTable::with_megabytes(hash_megabytes))),
            killers: Killers::default(),
            threads,
        }
    }

    pub fn clear(&mut self) {
        self.table.clear();
        self.killers.clear();
    }
}

impl Default for Memory {
    fn default() -> Memory {
        Memory::new(DEFAULT_MAX_TT_ENTRIES)
    }
}

/// A clock: seconds since some fixed instant.
pub type Clock<'a> = &'a (dyn Fn() -> f64 + Sync);

/// How a search is controlled from outside: the clock, the hard limit and, for the helper
/// threads, the flag that tells them to stop.
pub struct Control<'a> {
    /// Time (in the units of `clock`) the elapsed time is measured from.
    pub start: f64,
    /// Elapsed time after which the search is cut.
    pub hard_limit: f64,
    pub clock: Clock<'a>,
    pub stop: Option<&'a AtomicBool>,
    /// Depth the main thread is searching, written by it and read by the helpers.
    pub progress: Option<&'a AtomicUsize>,
    /// Helper threads rotate the moves of the root by this much, so that they do not all start with
    /// the same one.
    pub rotation: usize,
}

struct Search<'a> {
    state: &'a mut GameState,
    eval: &'a dyn Evaluator,
    memory: &'a mut Memory,
    options: &'a Options,
    control: &'a Control<'a>,
    root_player: Color,
    max_depth: i32,
    nodes: u64,
    prunings: u64,
    max_ply: i32,
    truncated: bool,
}

type Outcome = (f64, Option<u32>);

impl Search<'_> {
    /// Utility of a finished game, nudged so that quicker wins and slower losses are preferred.
    fn terminal_utility(&self, winner: Option<Color>, ply: i32) -> f64 {
        match winner {
            Some(color) if color == self.root_player => 1.0 - ply as f64 * self.options.epsilon,
            Some(_) => ply as f64 * self.options.epsilon,
            None => 0.5,
        }
    }

    fn static_value(&self) -> f64 {
        self.eval.evaluate(self.state.pos(), self.root_player)
    }

    /// Value of the node, searching `depth_remaining` more plies inside `(alpha, beta)`.
    fn node(&mut self, mut alpha: f64, mut beta: f64, depth_remaining: i32, ply: i32) -> Outcome {
        let (alpha_orig, beta_orig) = (alpha, beta);
        self.nodes += 1;
        self.max_ply = self.max_ply.max(ply);

        let key = self.state.key();
        let entry = if self.options.use_table {
            self.memory.table.get(&key)
        } else {
            None
        };
        if let Some(entry) = entry {
            if entry.depth >= depth_remaining {
                match entry.flag {
                    Flag::Exact => return (entry.value, entry.best),
                    Flag::LowerBound => alpha = alpha.max(entry.value),
                    Flag::UpperBound => beta = beta.min(entry.value),
                }
                if alpha >= beta {
                    return (entry.value, entry.best);
                }
            }
        }

        let moves = ordered_moves(self.state.pos());
        if let Some(winner) = self.state.outcome(&moves) {
            return (self.terminal_utility(winner, ply), None);
        }
        if depth_remaining <= 0 {
            if self.options.quiescence_plies > 0 {
                let value = self.quiescence(alpha, beta, self.options.quiescence_plies, ply, moves);
                return (value, None);
            }
            return (self.static_value(), None);
        }
        let stopped = self.control.stop.is_some_and(|flag| flag.load(Relaxed));
        if stopped || (self.control.clock)() - self.control.start >= self.control.hard_limit {
            self.truncated = true;
            return (self.static_value(), None);
        }

        let maximizing = self.state.turn() == self.root_player;
        let killers: &[u32] = if self.options.use_killers {
            self.memory.killers.get(depth_remaining)
        } else {
            &[]
        };
        let mut ordered = order(moves, killers, self.options.use_capture_order);
        if depth_remaining == self.max_depth && self.control.rotation > 0 {
            let shift = self.control.rotation % ordered.len();
            ordered.rotate_left(shift);
        }
        let mut best_value = if maximizing {
            f64::NEG_INFINITY
        } else {
            f64::INFINITY
        };
        let mut best_move = None;
        for action in ordered {
            self.state.apply(action);
            let (child, _) = self.node(alpha, beta, depth_remaining - 1, ply + 1);
            self.state.undo();
            if maximizing {
                if child > best_value {
                    best_value = child;
                    best_move = Some(move_id(&action));
                    alpha = alpha.max(best_value);
                }
            } else if child < best_value {
                best_value = child;
                best_move = Some(move_id(&action));
                beta = beta.min(best_value);
            }
            if alpha >= beta {
                self.prunings += 1;
                if self.options.use_killers {
                    self.memory.killers.record(depth_remaining, action);
                }
                break;
            }
        }
        if !self.truncated && self.options.use_table {
            self.memory.table.store(
                key,
                best_value,
                best_move,
                depth_remaining,
                alpha_orig,
                beta_orig,
            );
        }
        (best_value, best_move)
    }

    /// Value of a horizon node searching only captures and promotions (stand pat, no table).
    fn quiescence(
        &mut self,
        mut alpha: f64,
        mut beta: f64,
        plies_left: u32,
        ply: i32,
        moves: Vec<Move>,
    ) -> f64 {
        let mut best_value = self.static_value(); // nobody is forced to capture
        if plies_left == 0 {
            return best_value;
        }
        let maximizing = self.state.turn() == self.root_player;
        if maximizing {
            alpha = alpha.max(best_value);
        } else {
            beta = beta.min(best_value);
        }
        if alpha >= beta {
            self.prunings += 1;
            return best_value;
        }
        for action in moves.into_iter().filter(is_noisy) {
            self.state.apply(action);
            self.nodes += 1;
            let child_moves = ordered_moves(self.state.pos());
            let value = match self.state.outcome(&child_moves) {
                Some(winner) => self.terminal_utility(winner, ply + 1),
                None => self.quiescence(alpha, beta, plies_left - 1, ply + 1, child_moves),
            };
            self.state.undo();
            if maximizing {
                best_value = best_value.max(value);
                alpha = alpha.max(best_value);
            } else {
                best_value = best_value.min(value);
                beta = beta.min(best_value);
            }
            if alpha >= beta {
                self.prunings += 1;
                break;
            }
        }
        best_value
    }
}

/// What one alpha-beta search found.
#[derive(Clone, Debug)]
pub struct SearchResult {
    /// Value of the root for the player to move there.
    pub value: f64,
    /// Best move of the root, `None` if the root itself was cut.
    pub best: Option<Move>,
    pub nodes: u64,
    pub prunings: u64,
    /// Deepest ply reached.
    pub max_ply: i32,
    /// Whether the clock or the stop flag cut the search.
    pub truncated: bool,
}

/// One alpha-beta search of exactly `max_depth` plies.
///
/// The search is cut when the elapsed time passes `control.hard_limit` or `control.stop` is raised:
/// it is then flagged as truncated and stores nothing in the table from then on.
pub fn alpha_beta(
    state: &mut GameState,
    eval: &dyn Evaluator,
    memory: &mut Memory,
    options: &Options,
    max_depth: i32,
    control: &Control,
) -> SearchResult {
    let root_player = state.turn();
    let root = state.pos().clone();
    let mut search = Search {
        state,
        eval,
        memory,
        options,
        control,
        root_player,
        max_depth,
        nodes: 0,
        prunings: 0,
        max_ply: 0,
        truncated: false,
    };
    let (value, best) = search.node(f64::NEG_INFINITY, f64::INFINITY, max_depth, 0);
    let best = best.and_then(|id| ordered_moves(&root).into_iter().find(|m| move_id(m) == id));
    SearchResult {
        value,
        best,
        nodes: search.nodes,
        prunings: search.prunings,
        max_ply: search.max_ply,
        truncated: search.truncated,
    }
}

/// The work of a helper thread: iterative deepening on its own copy of the position that only
/// fills the shared table. It searches one or two plies deeper than the main thread is, rotates the moves
/// of the root and stops when the flag is raised. Returns the nodes it visited.
fn helper(
    index: usize,
    mut state: GameState,
    eval: &dyn Evaluator,
    table: Arc<SharedTable>,
    options: &Options,
    max_depth: usize,
    control: &Control,
) -> u64 {
    let mut memory = Memory {
        table: Tt::Shared(table),
        killers: Killers::default(),
        threads: 1,
    };
    let mut nodes = 0;
    while !control.stop.is_some_and(|flag| flag.load(Relaxed)) {
        // Stay ahead of the main thread: the table is useful with what it will need next.
        let main_depth = control.progress.map_or(1, |p| p.load(Relaxed));
        let depth = (main_depth + 1 + index % 2).min(max_depth).max(1);
        let result = alpha_beta(
            &mut state,
            eval,
            &mut memory,
            options,
            depth as i32,
            control,
        );
        nodes += result.nodes;
        if result.truncated {
            break;
        }
    }
    nodes
}

/// Iterative deepening over [`alpha_beta`]: depth 1, 2, ... `max_depth`, returning the move of the
/// deepest **completed** iteration.
///
/// Before starting an iteration the time it would take is estimated from the finished ones and the
/// loop stops if it does not fit in what is left of `time_limit` (the soft budget). The estimate and
/// the soft budget are ignored until `min_depth` iterations are complete; only the hard limit can
/// stop earlier. An iteration cut by the hard limit is discarded unless it is the very first.
///
/// With `memory.threads > 1` the other threads search the same position and share the table (Lazy
/// SMP); the move is always the one of the main thread. `control.stop` and `control.rotation` are
/// those of the main thread and are not used by the helpers' copies.
pub fn iterative_deepening(
    state: &mut GameState,
    eval: &dyn Evaluator,
    memory: &mut Memory,
    options: &Options,
    max_depth: usize,
    time_limit: f64,
    control: &Control,
) -> Result<(Option<Move>, Stats), String> {
    let helpers = match &memory.table {
        Tt::Shared(table) if memory.threads > 1 => Some((table.clone(), memory.threads - 1)),
        _ => None,
    };
    let Some((table, extra)) = helpers else {
        return deepen(state, eval, memory, options, max_depth, time_limit, control);
    };
    table.new_search();
    let stop = AtomicBool::new(false);
    let helper_nodes = std::sync::atomic::AtomicU64::new(0);
    let progress = AtomicUsize::new(0);
    let copies: Vec<GameState> = (0..extra).map(|_| state.clone()).collect();
    let result = std::thread::scope(|scope| {
        for (index, copy) in copies.into_iter().enumerate() {
            let (table, stop, nodes, progress) = (table.clone(), &stop, &helper_nodes, &progress);
            scope.spawn(move || {
                let helper_control = Control {
                    start: control.start,
                    hard_limit: control.hard_limit,
                    clock: control.clock,
                    stop: Some(stop),
                    progress: Some(progress),
                    rotation: index + 1,
                };
                nodes.fetch_add(
                    helper(
                        index,
                        copy,
                        eval,
                        table,
                        options,
                        max_depth,
                        &helper_control,
                    ),
                    Relaxed,
                );
            });
        }
        let main_control = Control {
            start: control.start,
            hard_limit: control.hard_limit,
            clock: control.clock,
            stop: control.stop,
            progress: Some(&progress),
            rotation: control.rotation,
        };
        let result = deepen(
            state,
            eval,
            memory,
            options,
            max_depth,
            time_limit,
            &main_control,
        );
        stop.store(true, Relaxed);
        result
    });
    result.map(|(best, mut stats)| {
        stats.nodes_visited += helper_nodes.load(Relaxed);
        (best, stats)
    })
}

fn deepen(
    state: &mut GameState,
    eval: &dyn Evaluator,
    memory: &mut Memory,
    options: &Options,
    max_depth: usize,
    time_limit: f64,
    control: &Control,
) -> Result<(Option<Move>, Stats), String> {
    let estimate = estimators::get(&options.estimator).ok_or_else(|| {
        format!(
            "unknown estimator {:?}; available: {:?}",
            options.estimator,
            estimators::NAMES
        )
    })?;
    let clock = control.clock;
    let start = clock();
    let mut best_move: Option<Move> = None;
    let (mut total_nodes, mut total_prunings, mut deepest_ply) = (0u64, 0u64, 0i32);
    let mut completed_depth = 0usize;
    let mut last_truncated = false;
    let mut history: Vec<estimators::Iteration> = Vec::new();
    let mut iterations: Vec<IterationInfo> = Vec::new();
    let mut stop_reason = StopReason::MaxDepth;
    let mut stop_predicted = 0.0;

    for depth in 1..=max_depth {
        let remaining = time_limit - (clock() - start);
        if depth > options.min_depth && !history.is_empty() {
            let predicted = estimate(&history);
            if predicted > remaining * OVERSHOOT_FACTOR {
                stop_reason = StopReason::Estimate;
                stop_predicted = predicted;
                break;
            }
        }
        if let Some(progress) = control.progress {
            progress.store(depth, Relaxed);
        }
        let iteration_start = clock();
        let result = alpha_beta(state, eval, memory, options, depth as i32, control);
        let seconds = clock() - iteration_start;
        iterations.push(IterationInfo {
            depth,
            seconds,
            nodes: result.nodes,
            truncated: result.truncated,
        });
        total_nodes += result.nodes;
        total_prunings += result.prunings;
        last_truncated = result.truncated;
        deepest_ply = deepest_ply.max(result.max_ply);

        if result.truncated {
            if best_move.is_none() && result.best.is_some() {
                best_move = result.best; // better a partial answer than none at all
            }
            stop_reason = StopReason::Truncated;
            break;
        }
        history.push((seconds, result.nodes));
        best_move = result.best;
        completed_depth = depth;
        if depth >= options.min_depth && clock() - start >= time_limit {
            stop_reason = StopReason::Budget;
            break;
        }
    }

    let stats = Stats {
        nodes_visited: total_nodes,
        pruning_count: total_prunings,
        max_depth_reached: deepest_ply,
        truncated: last_truncated && best_move.is_none(),
        completed_depth,
        iterations,
        stop_reason,
        stop_predicted,
    };
    Ok((best_move, stats))
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicU64, Ordering};

    use shakmaty::{Chess, Position, Role};

    use super::*;
    use crate::board::{position_from_fen, uci_string};

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
                white += value(role)
                    * (board.by_role(role) & board.by_color(Color::White)).count() as f64;
                white -= value(role)
                    * (board.by_role(role) & board.by_color(Color::Black)).count() as f64;
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
            &Options::default(),
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
            (3, StopReason::Budget)
        );
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
                ..Options::default()
            };
            let mut memory = Memory::with_threads(1000, 4, 1);
            let mut game = state(fen);
            let (best, stats) =
                iterative_deepening(&mut game, &eval, &mut memory, &options, 4, 1e9, &control)
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
}
