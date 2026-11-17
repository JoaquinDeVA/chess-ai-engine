//! Alpha-beta search of a position to a fixed depth.
//!
//! `Search` holds the state of one search; the stages of a node live in one file each:
//! `node.rs` (the order of the stages), `table_access.rs`, `limits.rs`, `move_list.rs`,
//! `pruning.rs` and `reductions.rs` (the techniques), `move_loop.rs` and `child_search.rs`
//! (the moves of the node), and `quiescence.rs` (the horizon).

mod child_search;
mod limits;
mod move_list;
mod move_loop;
mod node;
mod pruning;
mod quiescence;
mod reductions;
mod table_access;

use shakmaty::{Color, Move};

use super::control::Control;
use super::memory::Memory;
use super::options::Options;
use super::ordering::piece_to;
use super::scores::{INFINITE, MATE};
#[cfg(feature = "search-summary")]
use super::stats::SearchReport;
use crate::board::{move_id, ordered_moves, GameState};
use crate::eval::{Net, NetState};

/// Width of the null window used by principal variation search: one unit of the score.
const NULL_WINDOW_WIDTH: i32 = 1;

/// A value and the best move found for it (`None` when the node has none to give).
type Outcome = (i32, Option<u32>);

/// What the stages of one node need to know about who moves and what was just played.
struct NodeContext {
    ply: i32,
    in_check: bool,
    mover: Color,
    /// Whether the mover is the root player (the value is then to be maximized).
    maximizing: bool,
    /// Piece-and-target of the move that led here (`None` at the root and after a pass).
    previous_piece_to: Option<usize>,
}

struct Search<'a> {
    state: &'a mut GameState,
    memory: &'a mut Memory,
    options: &'a Options,
    control: &'a Control<'a>,
    root_player: Color,
    max_depth: i32,
    nodes: u64,
    #[cfg(feature = "search-summary")]
    prunings: u64,
    #[cfg(feature = "search-summary")]
    max_ply: i32,
    truncated: bool,
    /// Node count from which the clock is read again (it is read every `CLOCK_EVERY_NODES`).
    next_clock_check: u64,
    /// Evaluation that follows the moves of the search.
    eval: NetState<'a>,
    /// Nodes spent under each root move, by move id (the clock reads how settled the best one is).
    root_nodes: Vec<(u32, u64)>,
    /// The moves played from the root to this node, `None` for a passed turn.
    played: Vec<Option<(u32, usize)>>,
}

impl Search<'_> {
    /// Value of a finished game for the root player: a mate counts the plies to it, so that
    /// quicker wins and slower losses are preferred.
    fn terminal_utility(&self, winner: Option<Color>, ply: i32) -> i32 {
        match winner {
            Some(color) if color == self.root_player => MATE - ply,
            Some(_) => ply - MATE,
            None => 0,
        }
    }

    fn static_value(&mut self) -> i32 {
        self.eval.score(self.state.pos(), self.root_player)
    }

    /// The static value of the node, evaluated once for all the techniques that ask for it.
    fn cached_static_value(&mut self, cache: &mut Option<i32>) -> i32 {
        *cache.get_or_insert_with(|| self.static_value())
    }

    /// Play `m` in the game state and in the incremental evaluation.
    fn play(&mut self, m: Move) {
        let child_key = crate::board::zobrist_after(self.state.pos(), self.state.key(), &m);
        self.memory.table.prefetch(&child_key);
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

    /// Take back the last move played with [`Search::play`] or [`Search::play_null`].
    fn unplay(&mut self) {
        self.state.undo();
        self.eval.undo();
        self.played.pop();
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
    /// Whether the clock or the stop flag cut the search.
    pub truncated: bool,
    /// Share of the nodes spent under the best move of the root (0 without one).
    pub best_share: f64,
    #[cfg(feature = "search-summary")]
    pub report: SearchReport,
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
        #[cfg(feature = "search-summary")]
        prunings: 0,
        #[cfg(feature = "search-summary")]
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
        truncated: search.truncated,
        best_share,
        #[cfg(feature = "search-summary")]
        report: SearchReport {
            prunings: search.prunings,
            max_ply: search.max_ply,
        },
    }
}
