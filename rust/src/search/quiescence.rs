//! The quiescence search: at the horizon, only captures and promotions that do not lose material.

use shakmaty::MoveList;

use super::node::Search;
use super::ordering::{order_noisy, see_ge};
use crate::board::ordered_moves;

impl Search<'_> {
    /// Value of a horizon node searching only captures and promotions (stand pat, no table).
    pub(super) fn quiescence(
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
        for action in order_noisy(&moves) {
            // Captures that lose material are not searched.
            if !see_ge(self.state.pos(), &action, 0) {
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
