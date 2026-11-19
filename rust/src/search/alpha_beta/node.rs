//! The search of one node: the stages in the order they run. Each stage lives in its own file.

use shakmaty::{MoveList, Position};

use super::reductions::internal_iterative_reduction;
use super::{NodeContext, Outcome, Search};
use crate::board::legal_moves;

impl Search<'_> {
    /// Value of the node, searching `depth_remaining` more plies inside `(alpha, beta)`.
    ///
    /// The order of the checks matters: the table first (it can end the node for free), then game
    /// end and horizon, then the clock, and only then the techniques that cost a search.
    pub(super) fn node(
        &mut self,
        mut alpha: i32,
        mut beta: i32,
        depth_remaining: i32,
        ply: i32,
        allow_null: bool,
    ) -> Outcome {
        // The window as it came in: the table flag is decided against it, not against the one
        // narrowed below by the table and by the moves.
        let window_in = (alpha, beta);
        self.nodes += 1;
        #[cfg(feature = "search-summary")]
        {
            self.seldepth = self.seldepth.max(ply);
        }

        let in_check = self.state.pos().is_check();
        let depth_remaining = if in_check && depth_remaining > 0 {
            depth_remaining + 1
        } else {
            depth_remaining
        };
        let key = self.state.key();
        let probe = self.probe_table(key, depth_remaining, ply, &mut alpha, &mut beta);
        if let Some(cut) = probe.cut {
            return cut;
        }
        let repetitions_before = self.memory.diagnostics.repetition_draws();

        // Inner nodes out of check with no repetition: the draws that need no moves now, the
        // legal moves (and stalemate) only after reverse futility and the null move.
        let deferred = depth_remaining > 0 && ply > 0 && !in_check && !self.state.has_repeated();
        let mut moves = MoveList::new();
        if deferred {
            let pos = self.state.pos();
            if pos.is_insufficient_material() || pos.halfmoves() >= 100 {
                return (self.terminal_utility(None, ply), None);
            }
        } else {
            moves = legal_moves(self.state.pos());
            if let Some(value) = self.finished_game_value(&moves, ply) {
                return (value, None);
            }
            if depth_remaining <= 0 {
                return (self.horizon_value(alpha, beta, ply, &moves), None);
            }
        }
        // Checked after the horizon test so a cut never hides a quiescence value, and before any
        // move is searched so a cut node stores nothing in the table.
        if self.out_of_budget() {
            self.truncated = true;
            return (self.static_value(), None);
        }

        let context = self.node_context(ply, in_check);
        let mut static_cache: Option<i32> = None;
        if let Some(value) =
            self.reverse_futility_value(&context, depth_remaining, (alpha, beta), &mut static_cache)
        {
            return (value, None);
        }
        if let Some(cut) = self.null_move_cut(
            &context,
            depth_remaining,
            (alpha, beta),
            allow_null,
            &mut static_cache,
        ) {
            return cut;
        }
        if deferred {
            moves = legal_moves(self.state.pos());
            if moves.is_empty() {
                return (self.terminal_utility(None, ply), None);
            }
        }
        let node_moves = self.node_moves(&context, &moves, depth_remaining, probe.best);
        let depth_remaining = internal_iterative_reduction(depth_remaining, ply, probe.best);
        let futility_value =
            self.futility_value(&context, depth_remaining, (alpha, beta), &mut static_cache);
        let (best_value, best_move) = self.search_moves(
            &context,
            node_moves,
            (alpha, beta),
            depth_remaining,
            probe.best,
            futility_value,
        );
        // A cut search has partial values, so it must not be stored as if it were exact.
        if !self.truncated && probe.clock_free {
            self.store_result(
                key,
                (best_value, best_move),
                depth_remaining,
                window_in,
                ply,
                repetitions_before,
            );
        }
        (best_value, best_move)
    }

    /// Value of a node whose game is over (mate, stalemate, draw), `None` if it goes on.
    fn finished_game_value(&mut self, moves: &MoveList, ply: i32) -> Option<i32> {
        let winner = if ply == 0 {
            self.state.outcome_at_root(moves)?
        } else {
            self.state.outcome(moves)?
        };
        let pos = self.state.pos();
        if winner.is_none()
            && !moves.is_empty()
            && !pos.is_insufficient_material()
            && pos.halfmoves() < 100
        {
            self.memory.diagnostics.repetition_draw();
        }
        Some(self.terminal_utility(winner, ply))
    }

    /// Value of a node at the horizon: the quiescence search, or the static value without one.
    fn horizon_value(&mut self, alpha: i32, beta: i32, ply: i32, moves: &MoveList) -> i32 {
        if self.options.quiescence_plies > 0 {
            self.quiescence(alpha, beta, self.options.quiescence_plies, ply, moves)
        } else {
            self.static_value()
        }
    }

    fn node_context(&self, ply: i32, in_check: bool) -> NodeContext {
        let mover = self.state.turn();
        let previous_piece_to = self
            .played
            .last()
            .copied()
            .flatten()
            .map(|(_, piece_to)| piece_to);
        NodeContext {
            ply,
            in_check,
            mover,
            maximizing: mover == self.root_player,
            previous_piece_to,
        }
    }
}
