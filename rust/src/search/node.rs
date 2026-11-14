//! The search of one node: alpha-beta with the table, the move order and the techniques of
//! `search_techniques.rs`.

use std::sync::atomic::Ordering::Relaxed;

use shakmaty::{Color, Move, MoveList, Position};

use super::memory::*;
use super::options::*;
use super::ordering::{move_to_front, order, piece_to, see_ge, MAX_KILLERS_PER_PLY};
use super::search_techniques::*;
use super::tt::Flag;
use crate::board::{move_id, ordered_moves, GameState};
use crate::eval::{Net, NetState, EVAL_LIMIT};

/// The clock is read once every this many nodes: reading it costs about as much as a node.
const CLOCK_EVERY_NODES: u64 = 2048;
/// Width of the null window used by principal variation search: one unit of the score.
const NULL_WINDOW_WIDTH: i32 = 1;

/// A bound beyond every value, the window of the root.
pub const INFINITE: i32 = 32_000;
/// Value of a mate on the board; a mate `n` plies from the root is worth `MATE - n` to the winner,
/// so a quicker mate is preferred one unit per ply.
pub const MATE: i32 = 31_000;
/// Values from here out are mates: no line is longer than `MATE - MATE_BOUND` plies, and every
/// evaluation is within `EVAL_LIMIT`, far below.
pub const MATE_BOUND: i32 = MATE - 1_000;

pub(super) struct Search<'a> {
    pub(super) state: &'a mut GameState,
    pub(super) memory: &'a mut Memory,
    pub(super) options: &'a Options,
    pub(super) control: &'a Control<'a>,
    pub(super) root_player: Color,
    pub(super) max_depth: i32,
    pub(super) nodes: u64,
    pub(super) prunings: u64,
    pub(super) max_ply: i32,
    pub(super) truncated: bool,
    /// Node count from which the clock is read again (it is read every `CLOCK_EVERY_NODES`).
    pub(super) next_clock_check: u64,
    /// Evaluation that follows the moves of the search.
    pub(super) eval: NetState<'a>,
    /// Nodes spent under each root move, by move id (the clock reads how settled the best one is).
    pub(super) root_nodes: Vec<(u32, u64)>,
    /// The moves played from the root to this node, `None` for a passed turn.
    pub(super) played: Vec<Option<(u32, usize)>>,
}

type Outcome = (i32, Option<u32>);

impl Search<'_> {
    /// Value of a finished game for the root player: a mate counts the plies to it, so that
    /// quicker wins and slower losses are preferred.
    pub(super) fn terminal_utility(&self, winner: Option<Color>, ply: i32) -> i32 {
        match winner {
            Some(color) if color == self.root_player => MATE - ply,
            Some(_) => ply - MATE,
            None => 0,
        }
    }

    pub(super) fn static_value(&mut self) -> i32 {
        self.eval.score(self.state.pos(), self.root_player)
    }

    /// Play `m` in the game state and in the incremental evaluation.
    pub(super) fn play(&mut self, m: Move) {
        self.eval.apply(self.state.pos(), &m);
        self.played.push(Some((move_id(&m), piece_to(&m))));
        self.state.apply(m);
    }

    /// Pass the turn in the game state and in the incremental evaluation.
    fn play_null(&mut self) {
        self.eval.apply_null();
        self.played.push(None);
        self.state.apply_null();
    }

    /// Search a child with principal variation search and late move reductions.
    fn search_child(
        &mut self,
        index: usize,
        (alpha, beta): (i32, i32),
        depth_remaining: i32,
        ply: i32,
        maximizing: bool,
        late_quiet: Option<bool>,
    ) -> i32 {
        let (low, high) = if maximizing {
            (alpha, alpha + NULL_WINDOW_WIDTH)
        } else {
            (beta - NULL_WINDOW_WIDTH, beta)
        };
        let has_full_window = beta - alpha > NULL_WINDOW_WIDTH;
        if index > 0 && depth_remaining >= 2 {
            // A late quiet move that gives no check is searched shallower first; if even so it
            // improves on the best move, it is searched again to the full depth.
            let reduction_plies = match late_quiet {
                Some(favorable_history) if !self.state.pos().is_check() => {
                    late_move_reduction(depth_remaining, index, favorable_history)
                }
                _ => 0,
            };
            if reduction_plies > 0 {
                let (shallow, _) = self.node(
                    low,
                    high,
                    depth_remaining - 1 - reduction_plies,
                    ply + 1,
                    true,
                );
                if (shallow <= alpha && maximizing) || (shallow >= beta && !maximizing) {
                    return shallow;
                }
            }
            let (scout, _) = self.node(low, high, depth_remaining - 1, ply + 1, true);
            if scout > alpha && scout < beta && !self.truncated && has_full_window {
                return self.node(alpha, beta, depth_remaining - 1, ply + 1, true).0;
            }
            return scout;
        }
        self.node(alpha, beta, depth_remaining - 1, ply + 1, true).0
    }

    /// Take back the last move played with [`Search::play`].
    pub(super) fn unplay(&mut self) {
        self.state.undo();
        self.eval.undo();
        self.played.pop();
    }

    /// Value of the node, searching `depth_remaining` more plies inside `(alpha, beta)`.
    fn node(
        &mut self,
        mut alpha: i32,
        mut beta: i32,
        depth_remaining: i32,
        ply: i32,
        allow_null: bool,
    ) -> Outcome {
        // The window as it came in: the table flag is decided against it, not against the one
        // narrowed below by the table and by the moves.
        let (alpha_orig, beta_orig) = (alpha, beta);
        self.nodes += 1;
        self.max_ply = self.max_ply.max(ply);

        let in_check = self.state.pos().is_check();
        let depth_remaining = if in_check && depth_remaining > 0 {
            depth_remaining + 1
        } else {
            depth_remaining
        };
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
            // A restricted root is never cut by the table: its move may be one the list excludes.
            let restricted_root = ply == 0 && self.options.root_moves.is_some();
            if clock_free && entry.depth >= depth_remaining && !restricted_root {
                let value = at_ply(entry.value, ply);
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
        let mut killer_ids = [0u32; MAX_KILLERS_PER_PLY];
        let stored = self.memory.killers.get(ply);
        killer_ids[..stored.len()].copy_from_slice(stored);
        let killers = &killer_ids[..stored.len()];
        let mover = self.state.turn();
        let (previous, previous_piece_to) = match self.played.last().copied().flatten() {
            Some((id, piece_to)) => (Some(id), Some(piece_to)),
            None => (None, None),
        };
        let counter = self.memory.countermoves.get(mover, previous);
        let ordered = order(
            &moves,
            killers,
            counter,
            Some((&self.memory.history, mover)),
            previous_piece_to.map(|previous| (&self.memory.continuation, previous)),
        );
        // Captures that lose material come after the quiet moves (the order is otherwise kept).
        let pos = self.state.pos();
        let mut ordered: MoveList = ordered
            .iter()
            .filter(|m| !m.is_capture() || see_ge(pos, m, 0))
            .chain(
                ordered
                    .iter()
                    .filter(|m| m.is_capture() && !see_ge(pos, m, 0)),
            )
            .copied()
            .collect();
        if let (0, Some(allowed)) = (ply, &self.options.root_moves) {
            let kept: MoveList = ordered
                .iter()
                .filter(|m| allowed.contains(&move_id(m)))
                .copied()
                .collect();
            // A list that matches no legal move is a mistake of the caller: search everything.
            if !kept.is_empty() {
                ordered = kept;
            }
        }
        // Helper threads rotate the root moves so they do not all search the same subtree; the
        // table move is not put first there for the same reason.
        let helper_root = depth_remaining == self.max_depth && self.control.rotation > 0;
        if helper_root {
            let shift = self.control.rotation % ordered.len();
            ordered.rotate_left(shift);
        }
        if !helper_root {
            move_to_front(&mut ordered, table_move);
        }
        // Reverse futility: a static value far beyond the bound, in a window of no width and not in
        // check, is taken as the value of the node.
        if ply > 0
            && depth_remaining <= RFP_UP_TO_DEPTH
            && !in_check
            && beta - alpha <= NULL_WINDOW_WIDTH
            // A bound in the range of the mates is not one a static value can judge.
            && alpha > -EVAL_LIMIT
            && beta < EVAL_LIMIT
        {
            let value = self.static_value();
            let margin = RFP_MARGIN_PER_PLY * depth_remaining;
            if (maximizing && value - beta >= margin) || (!maximizing && alpha - value >= margin) {
                return (value, None);
            }
        }
        // Null move: it needs a pass to be safe (not in check, not at the root, not twice in a
        // row) and pieces on the board, or zugzwang makes passing the best move. Only tried when
        // the static value already beats the bound, so the reduced search rarely wastes time.
        if allow_null
            && ply > 0
            && depth_remaining >= NULL_MOVE_FROM_DEPTH
            && !in_check
            && has_pieces(self.state.pos(), mover)
        {
            let value = self.static_value();
            let margin = if maximizing {
                value - beta
            } else {
                alpha - value
            };
            if margin >= 0 {
                // If passing the turn already cuts, playing a move will too: prune.
                let reduction = null_move_reduction(depth_remaining, margin);
                let (low, high) = if maximizing {
                    (beta - NULL_WINDOW_WIDTH, beta)
                } else {
                    (alpha, alpha + NULL_WINDOW_WIDTH)
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
        // Internal iterative reduction: with no table move the order is poor, so search one ply
        // less; the next iteration, with the move stored, goes deeper.
        let depth_remaining =
            if table_move.is_none() && ply > 0 && depth_remaining >= IIR_FROM_DEPTH {
                depth_remaining - 1
            } else {
                depth_remaining
            };
        // Shallow pruning of the quiet moves: not in check, bounds outside the mate range.
        let shallow = !in_check
            && depth_remaining <= SHALLOW_UP_TO_DEPTH
            && alpha > -EVAL_LIMIT
            && beta < EVAL_LIMIT;
        let futility_static = shallow.then(|| self.static_value());
        let mut best_value = if maximizing { -INFINITE } else { INFINITE };
        let mut best_move = None;
        for (index, action) in ordered.into_iter().enumerate() {
            // Reducible: a quiet move, not a killer, in a node that is not in check. `Some` carries
            // whether the history says the move has cut before.
            let late = (!in_check
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
                let futile = futility_static.is_some_and(|value| {
                    let margin = FUTILITY_MARGIN_PER_PLY * depth_remaining;
                    if maximizing {
                        alpha - value >= margin
                    } else {
                        value - beta >= margin
                    }
                });
                if futile {
                    // A move that gives check is never skipped.
                    self.play(action);
                    let gives_check = self.state.pos().is_check();
                    self.unplay();
                    if !gives_check {
                        continue;
                    }
                }
            }
            let nodes_before = self.nodes;
            self.play(action);
            let child =
                self.search_child(index, (alpha, beta), depth_remaining, ply, maximizing, late);
            self.unplay();
            if ply == 0 {
                self.root_nodes
                    .push((move_id(&action), self.nodes - nodes_before));
            }
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
                self.memory.killers.record(ply, action);
                if !action.is_capture() && !action.is_promotion() {
                    self.memory.history.reward(mover, &action, depth_remaining);
                    self.memory.countermoves.record(mover, previous, &action);
                    if let Some(previous) = previous_piece_to {
                        self.memory
                            .continuation
                            .reward(mover, previous, &action, depth_remaining);
                    }
                }
                break;
            }
        }
        // A cut search has partial values, so it must not be stored as if it were exact.
        if !self.truncated && clock_free {
            // Mates are stored by their distance from this node, not from the root.
            let dropped = self.memory.table.store(
                key,
                for_table(best_value, ply),
                best_move,
                depth_remaining,
                for_table(alpha_orig, ply),
                for_table(beta_orig, ply),
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
}

/// A value as the table keeps it: the distance to a mate counts from the node, not from the root.
/// The map is increasing, so the bounds of the window can go through it as well.
fn for_table(value: i32, ply: i32) -> i32 {
    if value >= MATE_BOUND {
        value + ply
    } else if value <= -MATE_BOUND {
        value - ply
    } else {
        value
    }
}

/// The inverse of [`for_table`] for a node at `ply`.
fn at_ply(value: i32, ply: i32) -> i32 {
    if value >= MATE_BOUND {
        value - ply
    } else if value <= -MATE_BOUND {
        value + ply
    } else {
        value
    }
}

/// What one alpha-beta search found.
#[derive(Clone, Debug)]
pub struct SearchResult {
    /// Value of the root for the player to move there: an evaluation within `±EVAL_LIMIT`, or a
    /// mate beyond `±MATE_BOUND`.
    pub value: i32,
    /// Best move of the root, `None` if the root itself was cut.
    pub best: Option<Move>,
    pub nodes: u64,
    pub prunings: u64,
    /// Deepest ply reached.
    pub max_ply: i32,
    /// Whether the clock or the stop flag cut the search.
    pub truncated: bool,
    /// Share of the nodes spent under the best move of the root (0 without one).
    pub best_share: f64,
}

/// One alpha-beta search of exactly `max_depth` plies.
///
/// The search is cut when the elapsed time passes `control.hard_limit` or `control.stop` is raised:
/// it is then flagged as truncated and stores nothing in the table from then on.
pub fn alpha_beta(
    state: &mut GameState,
    net: &Net,
    memory: &mut Memory,
    options: &Options,
    max_depth: i32,
    control: &Control,
) -> SearchResult {
    let root_player = state.turn();
    let root = state.pos().clone();
    let mut search = Search {
        state,
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
        eval: NetState::new(net, &root),
        root_nodes: Vec::new(),
        played: Vec::new(),
    };
    let (value, best) = search.node(-INFINITE, INFINITE, max_depth, 0, true);
    let best_nodes: u64 = search
        .root_nodes
        .iter()
        .filter(|&&(id, _)| Some(id) == best)
        .map(|&(_, nodes)| nodes)
        .sum();
    let best_share = best_nodes as f64 / search.nodes.max(1) as f64;
    let best = best.and_then(|id| ordered_moves(&root).into_iter().find(|m| move_id(m) == id));
    SearchResult {
        value,
        best,
        nodes: search.nodes,
        prunings: search.prunings,
        max_ply: search.max_ply,
        truncated: search.truncated,
        best_share,
    }
}
