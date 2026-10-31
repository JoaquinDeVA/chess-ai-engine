//! The search of one node: alpha-beta with its pruning techniques, and the quiescence search.

use std::sync::atomic::Ordering::Relaxed;

use shakmaty::{Color, Move, Position};

use super::options::*;
use super::tt::{move_to_front, order, Flag};
use crate::board::{is_noisy, move_id, ordered_moves, GameState};
use crate::eval::{EvalState, Evaluator};

/// Late quiet moves are reduced from this move on (counting from 0) and at least this deep.
const LMR_FROM_MOVE: usize = 5;
const LMR_FROM_DEPTH: i32 = 4;
/// From this move on the reduction is two plies.
const LMR_DEEPER_FROM_MOVE: usize = 15;
/// The null move is tried from this depth on.
const NULL_MOVE_FROM_DEPTH: i32 = 3;
/// Width of the window of the scouts of the principal variation search.
const SCOUT_EPSILON: f64 = 1e-9;

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
    /// Evaluation that follows the moves, when the technique is on and the evaluator supports it.
    incremental: Option<Box<dyn EvalState + 'a>>,
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
        match &self.incremental {
            Some(state) => state.evaluate(self.state.pos(), self.root_player),
            None => self.eval.evaluate(self.state.pos(), self.root_player),
        }
    }

    /// Play `m` in the game state and in the incremental evaluation.
    fn play(&mut self, m: Move) {
        if let Some(state) = &mut self.incremental {
            state.apply(self.state.turn(), &m);
        }
        self.state.apply(m);
    }

    /// Pass the turn in the game state and in the incremental evaluation.
    fn play_null(&mut self) {
        if let Some(state) = &mut self.incremental {
            state.apply_null();
        }
        self.state.apply_null();
    }

    /// Whether `color` has a piece other than pawns and the king (otherwise passing the turn can
    /// be the worst thing to do, and a null move is not tried).
    fn has_pieces(&self, color: Color) -> bool {
        let board = self.state.pos().board();
        (board.by_color(color) & !(board.pawns() | board.kings())).any()
    }

    /// Value of the child just played, searched as the techniques say: late quiet moves with less
    /// depth first, the moves after the first with a window of no width, and again with the whole
    /// window and depth when the cheaper search says the move is better than the best so far.
    fn child_value(
        &mut self,
        index: usize,
        late_quiet: bool,
        (alpha, beta): (f64, f64),
        depth_remaining: i32,
        ply: i32,
        maximizing: bool,
    ) -> f64 {
        let (low, high) = if maximizing {
            (alpha, alpha + SCOUT_EPSILON)
        } else {
            (beta - SCOUT_EPSILON, beta)
        };
        let narrow = beta - alpha > SCOUT_EPSILON;
        if late_quiet && narrow {
            let reduction = if index >= LMR_DEEPER_FROM_MOVE && depth_remaining >= 5 {
                2
            } else {
                1
            };
            let (value, _) = self.node(low, high, depth_remaining - 1 - reduction, ply + 1, true);
            let better = if maximizing {
                value > alpha
            } else {
                value < beta
            };
            if !better || self.truncated {
                return value;
            }
        }
        if self.options.techniques.pvs && index > 0 && depth_remaining >= 2 && narrow {
            let (scout, _) = self.node(low, high, depth_remaining - 1, ply + 1, true);
            if scout > alpha && scout < beta && !self.truncated {
                return self.node(alpha, beta, depth_remaining - 1, ply + 1, true).0;
            }
            return scout;
        }
        self.node(alpha, beta, depth_remaining - 1, ply + 1, true).0
    }

    /// Take back the last move played with [`Search::play`].
    fn unplay(&mut self) {
        self.state.undo();
        if let Some(state) = &mut self.incremental {
            state.undo();
        }
    }

    /// Value of the node, searching `depth_remaining` more plies inside `(alpha, beta)`.
    fn node(
        &mut self,
        mut alpha: f64,
        mut beta: f64,
        depth_remaining: i32,
        ply: i32,
        allow_null: bool,
    ) -> Outcome {
        let (alpha_orig, beta_orig) = (alpha, beta);
        self.nodes += 1;
        self.max_ply = self.max_ply.max(ply);

        let key = self.state.key();
        let entry = if self.options.use_table {
            self.memory.table.get(&key)
        } else {
            None
        };
        let table_move = entry.and_then(|e| e.best);
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
        let killers: Vec<u32> = if self.options.use_killers {
            self.memory.killers.get(depth_remaining).to_vec()
        } else {
            Vec::new()
        };
        let mover = self.state.turn();
        let history = if self.options.techniques.history {
            Some((&self.memory.history, mover))
        } else {
            None
        };
        let mut ordered = order(moves, &killers, self.options.use_capture_order, history);
        let helper_root = depth_remaining == self.max_depth && self.control.rotation > 0;
        if helper_root {
            let shift = self.control.rotation % ordered.len();
            ordered.rotate_left(shift);
        }
        if self.options.techniques.tt_move && !helper_root {
            move_to_front(&mut ordered, table_move);
        }
        let in_check = self.state.pos().is_check();
        if self.options.techniques.null_move
            && allow_null
            && ply > 0
            && depth_remaining >= NULL_MOVE_FROM_DEPTH
            && !in_check
            && self.has_pieces(mover)
        {
            let enough = if maximizing {
                self.static_value() >= beta
            } else {
                self.static_value() <= alpha
            };
            if enough {
                // If passing the turn already cuts, playing a move will too: prune.
                let reduction = if depth_remaining >= 6 { 3 } else { 2 };
                let (low, high) = if maximizing {
                    (beta - SCOUT_EPSILON, beta)
                } else {
                    (alpha, alpha + SCOUT_EPSILON)
                };
                self.play_null();
                let (value, _) =
                    self.node(low, high, depth_remaining - 1 - reduction, ply + 1, false);
                self.unplay();
                if !self.truncated
                    && ((maximizing && value >= beta) || (!maximizing && value <= alpha))
                {
                    return (if maximizing { beta } else { alpha }, None);
                }
            }
        }
        let mut best_value = if maximizing {
            f64::NEG_INFINITY
        } else {
            f64::INFINITY
        };
        let mut best_move = None;
        for (index, action) in ordered.into_iter().enumerate() {
            self.play(action);
            let late_quiet = self.options.techniques.lmr
                && index >= LMR_FROM_MOVE
                && depth_remaining >= LMR_FROM_DEPTH
                && !in_check
                && !action.is_capture()
                && !action.is_promotion()
                && !killers.contains(&move_id(&action))
                && !self.state.pos().is_check();
            let child = self.child_value(
                index,
                late_quiet,
                (alpha, beta),
                depth_remaining,
                ply,
                maximizing,
            );
            self.unplay();
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
                if self.options.techniques.history && !action.is_capture() && !action.is_promotion()
                {
                    self.memory.history.reward(mover, &action, depth_remaining);
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
            self.play(action);
            self.nodes += 1;
            let child_moves = ordered_moves(self.state.pos());
            let value = match self.state.outcome(&child_moves) {
                Some(winner) => self.terminal_utility(winner, ply + 1),
                None => self.quiescence(alpha, beta, plies_left - 1, ply + 1, child_moves),
            };
            self.unplay();
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
    alpha_beta_window(
        state,
        eval,
        memory,
        options,
        max_depth,
        control,
        (f64::NEG_INFINITY, f64::INFINITY),
    )
}

/// [`alpha_beta`] with a window other than the whole interval: a value inside it is exact, one on
/// its edge (or outside) only says that the true value is on that side.
pub fn alpha_beta_window(
    state: &mut GameState,
    eval: &dyn Evaluator,
    memory: &mut Memory,
    options: &Options,
    max_depth: i32,
    control: &Control,
    window: (f64, f64),
) -> SearchResult {
    let root_player = state.turn();
    let root = state.pos().clone();
    let incremental = if options.techniques.incremental {
        eval.new_state(&root)
    } else {
        None
    };
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
        incremental,
    };
    let (value, best) = search.node(window.0, window.1, max_depth, 0, true);
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
