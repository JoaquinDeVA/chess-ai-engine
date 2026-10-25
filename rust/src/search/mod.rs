//! Alpha-beta search with iterative deepening. Port of `search/game_algorithms.py`.
//!
//! The values are expected scores of the root player in `[0, 1]`; terminal values are 1, 0.5 and
//! 0, nudged by `epsilon` per ply so that quicker wins and slower losses are preferred. The
//! search keeps a bounded transposition table and killer moves between iterations and between the
//! moves of a game, and ends every branch at the horizon with a quiescence search over captures
//! and promotions.

pub mod estimators;
pub mod tt;

use shakmaty::{Color, Move};

use crate::board::{is_noisy, ordered_moves, GameState};
use crate::eval::Evaluator;
use tt::{order, Flag, Killers, Table, DEFAULT_MAX_TT_ENTRIES};

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
        }
    }
}

/// What a game keeps between two moves: the transposition table and the killer moves.
pub struct Memory {
    pub table: Table,
    pub killers: Killers,
}

impl Memory {
    pub fn new(max_tt_entries: usize) -> Memory {
        Memory {
            table: Table::new(max_tt_entries),
            killers: Killers::default(),
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
pub type Clock<'a> = &'a dyn Fn() -> f64;

struct Search<'a> {
    state: &'a mut GameState,
    eval: &'a dyn Evaluator,
    memory: &'a mut Memory,
    options: &'a Options,
    root_player: Color,
    max_depth: i32,
    start: f64,
    hard_limit: f64,
    clock: Clock<'a>,
    nodes: u64,
    prunings: u64,
    max_ply: i32,
    truncated: bool,
}

type Outcome = (f64, Option<Move>);

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
    fn node(&mut self, mut alpha: f64, mut beta: f64, depth_remaining: i32) -> Outcome {
        let (alpha_orig, beta_orig) = (alpha, beta);
        self.nodes += 1;
        let ply = self.max_depth - depth_remaining;
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
        if (self.clock)() - self.start >= self.hard_limit {
            self.truncated = true;
            return (self.static_value(), None);
        }

        let maximizing = self.state.turn() == self.root_player;
        let killers: &[u32] = if self.options.use_killers {
            self.memory.killers.get(depth_remaining)
        } else {
            &[]
        };
        let ordered = order(moves, killers, self.options.use_capture_order);
        let mut best_value = if maximizing {
            f64::NEG_INFINITY
        } else {
            f64::INFINITY
        };
        let mut best_move = None;
        for action in ordered {
            self.state.apply(action);
            let (child, _) = self.node(alpha, beta, depth_remaining - 1);
            self.state.undo();
            if maximizing {
                if child > best_value {
                    best_value = child;
                    best_move = Some(action);
                    alpha = alpha.max(best_value);
                }
            } else if child < best_value {
                best_value = child;
                best_move = Some(action);
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

/// One alpha-beta search of exactly `max_depth` plies.
///
/// `hard_limit` is the elapsed time (since `start`, in the units of `clock`) after which the
/// search is cut: it is then flagged as truncated and stores nothing in the table from then on.
/// Returns the value of the root, its best move (`None` if the root itself was cut) and the
/// counters `(nodes, prunings, deepest ply, truncated)`.
#[allow(clippy::too_many_arguments)]
pub fn alpha_beta(
    state: &mut GameState,
    eval: &dyn Evaluator,
    memory: &mut Memory,
    options: &Options,
    max_depth: i32,
    start: f64,
    hard_limit: f64,
    clock: Clock,
) -> (f64, Option<Move>, u64, u64, i32, bool) {
    let root_player = state.turn();
    let mut search = Search {
        state,
        eval,
        memory,
        options,
        root_player,
        max_depth,
        start,
        hard_limit,
        clock,
        nodes: 0,
        prunings: 0,
        max_ply: 0,
        truncated: false,
    };
    let (value, best) = search.node(f64::NEG_INFINITY, f64::INFINITY, max_depth);
    (
        value,
        best,
        search.nodes,
        search.prunings,
        search.max_ply,
        search.truncated,
    )
}

/// Iterative deepening over [`alpha_beta`]: depth 1, 2, ... `max_depth`, returning the move of the
/// deepest **completed** iteration.
///
/// Before starting an iteration the time it would take is estimated from the finished ones and the
/// loop stops if it does not fit in what is left of `time_limit` (the soft budget). The estimate and
/// the soft budget are ignored until `min_depth` iterations are complete; only `hard_limit` can
/// stop earlier. An iteration cut by the hard limit is discarded unless it is the very first.
#[allow(clippy::too_many_arguments)]
pub fn iterative_deepening(
    state: &mut GameState,
    eval: &dyn Evaluator,
    memory: &mut Memory,
    options: &Options,
    max_depth: usize,
    time_limit: f64,
    hard_limit: f64,
    clock: Clock,
) -> Result<(Option<Move>, Stats), String> {
    let estimate = estimators::get(&options.estimator).ok_or_else(|| {
        format!(
            "unknown estimator {:?}; available: {:?}",
            options.estimator,
            estimators::NAMES
        )
    })?;
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
        let iteration_start = clock();
        let (_, mv, nodes, prunings, ply, truncated) = alpha_beta(
            state,
            eval,
            memory,
            options,
            depth as i32,
            start,
            hard_limit,
            clock,
        );
        let seconds = clock() - iteration_start;
        iterations.push(IterationInfo {
            depth,
            seconds,
            nodes,
            truncated,
        });
        total_nodes += nodes;
        total_prunings += prunings;
        last_truncated = truncated;
        deepest_ply = deepest_ply.max(ply);

        if truncated {
            if best_move.is_none() && mv.is_some() {
                best_move = mv; // better a partial answer than none at all
            }
            stop_reason = StopReason::Truncated;
            break;
        }
        history.push((seconds, nodes));
        best_move = mv;
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
