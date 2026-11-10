//! The search of one node: alpha-beta with its pruning techniques, and the quiescence search.

use std::sync::atomic::Ordering::Relaxed;

use shakmaty::{Color, Move, MoveList, Position};

use super::memory::*;
use super::options::*;
use super::ordering::{move_to_front, order, order_noisy, see_ge, MAX_KILLERS_PER_DEPTH};
use super::tt::Flag;
use crate::board::{move_id, ordered_moves, GameState};
use crate::eval::{EvalState, Evaluator};

/// The null move is tried from this depth on.
const NULL_MOVE_FROM_DEPTH: i32 = 3;
/// The clock is read once every this many nodes: reading it costs about as much as a node.
const CLOCK_EVERY_NODES: u64 = 2048;
/// Width of the window of the scouts of the principal variation search.
const SCOUT_EPSILON: f64 = 1e-9;
/// Late move reductions start at this depth and at this position in the move order.
const LMR_FROM_DEPTH: i32 = 3;
const LMR_FROM_INDEX: usize = 3;

/// Reverse futility pruning works up to this depth, and prunes when the static value is beyond the
/// bound by `RFP_MARGIN_PER_PLY` log-odds per ply left: 80 centipawns of the plan on the usual
/// logistic scale (`80 · ln 10 / 400`).
const RFP_UP_TO_DEPTH: i32 = 6;
const RFP_MARGIN_PER_PLY: f64 = 0.46;

/// Log-odds of an expected score, to measure distances that mean the same at 0.5 and at 0.9.
pub(super) fn log_odds(p: f64) -> f64 {
    (p / (1.0 - p)).ln()
}

/// Futility pruning and late move pruning work up to this depth. The futility margin is
/// `FUTILITY_MARGIN_PER_PLY` log-odds per ply left (120 centipawns on the usual logistic scale,
/// `120 · ln 10 / 400`); late move pruning skips the quiet moves from the `3 + depth²`-th on.
const SHALLOW_UP_TO_DEPTH: i32 = 3;
const FUTILITY_MARGIN_PER_PLY: f64 = 0.69;

/// Quiet moves a node of `depth_remaining` plies tries before late move pruning skips the rest.
pub(super) fn late_move_limit(depth_remaining: i32) -> usize {
    3 + (depth_remaining * depth_remaining) as usize
}

/// Plies a late quiet move is searched less: `0.75 + ln(depth)·ln(index) / 2.25`, one less when
/// the history says the move has cut before, and never so many that the search falls below one ply.
pub(super) fn reduction(depth_remaining: i32, index: usize, good_history: bool) -> i32 {
    if depth_remaining < LMR_FROM_DEPTH || index < LMR_FROM_INDEX {
        return 0;
    }
    let raw = 0.75 + f64::from(depth_remaining).ln() * (index as f64).ln() / 2.25;
    let steps = raw as i32 - i32::from(good_history);
    steps.clamp(0, depth_remaining - 2)
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
    /// Node count from which the clock is read again (it is read every `CLOCK_EVERY_NODES`).
    next_clock_check: u64,
    /// Evaluation that follows the moves, when the technique is on and the evaluator supports it.
    incremental: Option<Box<dyn EvalState + 'a>>,
}

type Outcome = (f64, Option<u32>);

impl Search<'_> {
    /// A value as the table keeps it: the distance to a mate counts from the node, not from the
    /// root. The evaluations are clipped to `0.01 ..= 0.99` and a mate is farther out (the
    /// preference per ply is small enough for that), so what lies outside is a mate. The map is
    /// increasing, so the bounds of the window can go through it as well.
    fn for_table(&self, value: f64, ply: i32) -> f64 {
        let shift = ply as f64 * self.options.epsilon;
        if value > 0.99 {
            value + shift
        } else if value < 0.01 {
            value - shift
        } else {
            value
        }
    }

    /// The inverse of [`Search::for_table`] for a node at `ply`.
    fn at_ply(&self, value: f64, ply: i32) -> f64 {
        let shift = ply as f64 * self.options.epsilon;
        if value > 0.99 {
            value - shift
        } else if value < 0.01 {
            value + shift
        } else {
            value
        }
    }

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
            state.apply(self.state.pos(), &m);
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

    /// Value of the child just played, searched as the techniques say: the moves after the first
    /// with a window of no width, and again with the whole window when the cheaper search says the
    /// move is better than the best so far.
    fn child_value(
        &mut self,
        index: usize,
        (alpha, beta): (f64, f64),
        depth_remaining: i32,
        ply: i32,
        maximizing: bool,
        late: Option<bool>,
    ) -> f64 {
        let (low, high) = if maximizing {
            (alpha, alpha + SCOUT_EPSILON)
        } else {
            (beta - SCOUT_EPSILON, beta)
        };
        let narrow = beta - alpha > SCOUT_EPSILON;
        if index > 0 && depth_remaining >= 2 && narrow {
            // A late quiet move that gives no check is searched shallower first; if even so it
            // improves on the best move, it is searched again to the full depth.
            let reduction = match late {
                Some(good_history) if !self.state.pos().is_check() => {
                    reduction(depth_remaining, index, good_history)
                }
                _ => 0,
            };
            if reduction > 0 {
                let (shallow, _) =
                    self.node(low, high, depth_remaining - 1 - reduction, ply + 1, true);
                if (shallow <= alpha && maximizing) || (shallow >= beta && !maximizing) {
                    return shallow;
                }
            }
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
        // The window as it came in: the table flag is decided against it, not against the one
        // narrowed below by the table and by the moves.
        let (alpha_orig, beta_orig) = (alpha, beta);
        self.nodes += 1;
        self.max_ply = self.max_ply.max(ply);

        // Order of the checks: the table first (it can end the node for free), then game end and
        // horizon, then the clock, and only then the techniques that cost a search.
        let key = self.state.key();
        // A value searched `depth_remaining` plies deep can depend on the fifty-move clock only if
        // the clock can reach 100 inside that search; then the entry gives its move and nothing
        // else. A position with a repetition behind it is a possible draw: the table is not asked.
        let clock_free = self.state.pos().halfmoves() as i32 + depth_remaining < 100;
        let entry = if self.state.has_repeated() {
            None
        } else {
            self.memory.table.get(&key)
        };
        let table_move = entry.and_then(|e| e.best);
        self.memory.counters.table_probes += 1;
        if let Some(entry) = entry {
            self.memory.counters.table_hits += 1;
            if clock_free && entry.depth >= depth_remaining {
                let value = self.at_ply(entry.value, ply);
                match entry.flag {
                    Flag::Exact => {
                        self.memory.counters.table_cuts += 1;
                        return (value, entry.best);
                    }
                    Flag::LowerBound => alpha = alpha.max(value),
                    Flag::UpperBound => beta = beta.min(value),
                }
                if alpha >= beta {
                    self.memory.counters.table_cuts += 1;
                    return (value, entry.best);
                }
            }
        }
        let repetitions_before = self.memory.counters.repetition_draws;

        let moves = ordered_moves(self.state.pos());
        if let Some(winner) = self.state.outcome(&moves) {
            let pos = self.state.pos();
            if winner.is_none()
                && !moves.is_empty()
                && !pos.is_insufficient_material()
                && pos.halfmoves() < 100
            {
                self.memory.counters.repetition_draws += 1;
            }
            return (self.terminal_utility(winner, ply), None);
        }
        if depth_remaining <= 0 {
            if self.options.quiescence_plies > 0 {
                let value = self.quiescence(alpha, beta, self.options.quiescence_plies, ply, moves);
                return (value, None);
            }
            return (self.static_value(), None);
        }
        // Checked after the horizon test so a cut never hides a quiescence value, and before any
        // move is searched so a cut node stores nothing in the table.
        let stopped = self.control.stop.is_some_and(|flag| flag.load(Relaxed));
        let out_of_nodes = self
            .options
            .node_limit
            .is_some_and(|limit| self.nodes >= limit);
        let out_of_time = self.nodes >= self.next_clock_check && {
            self.next_clock_check = self.nodes + CLOCK_EVERY_NODES;
            (self.control.clock)() - self.control.start >= self.control.hard_limit
        };
        if stopped || out_of_nodes || out_of_time || self.truncated {
            self.truncated = true;
            return (self.static_value(), None);
        }

        let maximizing = self.state.turn() == self.root_player;
        let mut killer_ids = [0u32; MAX_KILLERS_PER_DEPTH];
        let stored = self.memory.killers.get(depth_remaining);
        killer_ids[..stored.len()].copy_from_slice(stored);
        let killers = &killer_ids[..stored.len()];
        let mover = self.state.turn();
        let mut ordered = order(&moves, killers, Some((&self.memory.history, mover)));
        if self.options.techniques.see {
            // Captures that lose material come after the quiet moves (the order is otherwise kept).
            let pos = self.state.pos();
            let mut reordered = MoveList::new();
            reordered.extend(
                ordered
                    .iter()
                    .filter(|m| !m.is_capture() || see_ge(pos, m, 0))
                    .copied(),
            );
            reordered.extend(
                ordered
                    .iter()
                    .filter(|m| m.is_capture() && !see_ge(pos, m, 0))
                    .copied(),
            );
            ordered = reordered;
        }
        // Helper threads rotate the root moves so they do not all search the same subtree; the
        // table move is not put first there for the same reason.
        let helper_root = depth_remaining == self.max_depth && self.control.rotation > 0;
        if helper_root {
            let shift = self.control.rotation % ordered.len();
            ordered.rotate_left(shift);
        }
        if self.options.techniques.tt_move && !helper_root {
            move_to_front(&mut ordered, table_move);
        }
        let in_check = self.state.pos().is_check();
        // Reverse futility: a static value far beyond the bound, in a window of no width and not in
        // check, is taken as the value of the node.
        if self.options.techniques.rfp
            && ply > 0
            && depth_remaining <= RFP_UP_TO_DEPTH
            && !in_check
            && beta - alpha <= SCOUT_EPSILON
            // A bound in the range of the mates is not one a static value can judge.
            && alpha > 0.01
            && beta < 0.99
        {
            let value = self.static_value();
            let margin = RFP_MARGIN_PER_PLY * f64::from(depth_remaining);
            if (maximizing && value >= beta && log_odds(value) - log_odds(beta) >= margin)
                || (!maximizing && value <= alpha && log_odds(alpha) - log_odds(value) >= margin)
            {
                return (value, None);
            }
        }
        // Null move: it needs a pass to be safe (not in check, not at the root, not twice in a
        // row) and pieces on the board, or zugzwang makes passing the best move. Only tried when
        // the static value already beats the bound, so the reduced search rarely wastes time.
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
        // Shallow pruning of the quiet moves: not in check, bounds outside the mate range.
        let shallow =
            !in_check && depth_remaining <= SHALLOW_UP_TO_DEPTH && alpha > 0.01 && beta < 0.99;
        let futility_static =
            (shallow && self.options.techniques.futility).then(|| self.static_value());
        let mut best_value = if maximizing {
            f64::NEG_INFINITY
        } else {
            f64::INFINITY
        };
        let mut best_move = None;
        for (index, action) in ordered.into_iter().enumerate() {
            // Reducible: a quiet move, not a killer, in a node that is not in check. `Some` carries
            // whether the history says the move has cut before.
            let late = (self.options.techniques.lmr
                && !in_check
                && !action.is_capture()
                && !action.is_promotion()
                && !killers.contains(&move_id(&action)))
            .then(|| self.memory.history.score(mover, &action) > 0);
            if shallow
                && index > 0
                && !action.is_capture()
                && !action.is_promotion()
                && !killers.contains(&move_id(&action))
            {
                let by_count =
                    self.options.techniques.lmp && index >= late_move_limit(depth_remaining);
                let by_margin = futility_static.is_some_and(|value| {
                    let margin = FUTILITY_MARGIN_PER_PLY * f64::from(depth_remaining);
                    if maximizing {
                        log_odds(alpha) - log_odds(value) >= margin
                    } else {
                        log_odds(value) - log_odds(beta) >= margin
                    }
                });
                if by_count || by_margin {
                    // A move that gives check is never skipped.
                    self.play(action);
                    let gives_check = self.state.pos().is_check();
                    self.unplay();
                    if !gives_check {
                        continue;
                    }
                }
            }
            self.play(action);
            let child =
                self.child_value(index, (alpha, beta), depth_remaining, ply, maximizing, late);
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
            // Cutoff: quiet moves that cut are remembered (killers, history) to be tried earlier
            // in sibling nodes; captures are already ordered first.
            if alpha >= beta {
                self.prunings += 1;
                let counters = &mut self.memory.counters;
                counters.cuts += 1;
                counters.first_move_cuts += u64::from(index == 0);
                let id = move_id(&action);
                if !action.is_capture() && table_move != Some(id) {
                    if let Some(slot) = killers.iter().position(|&killer| killer == id) {
                        counters.killer_cuts[slot] += 1;
                    }
                }
                self.memory.killers.record(depth_remaining, action);
                if !action.is_capture() && !action.is_promotion() {
                    self.memory.history.reward(mover, &action, depth_remaining);
                }
                break;
            }
        }
        // A cut search has partial values, so it must not be stored as if it were exact.
        if !self.truncated && clock_free {
            // Mates are stored by their distance from this node, not from the root.
            let dropped = self.memory.table.store(
                key,
                self.for_table(best_value, ply),
                best_move,
                depth_remaining,
                self.for_table(alpha_orig, ply),
                self.for_table(beta_orig, ply),
            );
            let counters = &mut self.memory.counters;
            counters.stores += 1;
            counters.stores_over_repetition +=
                u64::from(counters.repetition_draws > repetitions_before);
            if let Some(depth) = dropped {
                counters.evictions += 1;
                counters.deeper_evictions += u64::from(depth > depth_remaining);
            }
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
        moves: MoveList,
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
        let see = self.options.techniques.see;
        for action in order_noisy(&moves) {
            if see && !see_ge(self.state.pos(), &action, 0) {
                continue;
            }
            self.play(action);
            self.nodes += 1;
            self.memory.counters.quiescence_nodes += 1;
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
        next_clock_check: 0,
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
