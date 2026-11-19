//! The quiescence search: at the horizon, only captures and promotions that do not lose material.

use shakmaty::{Move, MoveList, Position};

use super::Search;
use crate::board::legal_moves;
use crate::search::ordering::{order_noisy, see_ge};
use crate::search::scores::INFINITE;

impl Search<'_> {
    /// Value of a horizon node: forced evasions in check, otherwise captures and promotions.
    pub(super) fn quiescence(
        &mut self,
        mut alpha: i32,
        mut beta: i32,
        plies_left: u32,
        ply: i32,
        moves: &MoveList,
    ) -> i32 {
        let in_check = self.state.pos().is_check();
        let maximizing = self.state.turn() == self.root_player;
        let mut best_value = if in_check {
            if maximizing {
                -INFINITE
            } else {
                INFINITE
            }
        } else {
            let value = self.static_value();
            if plies_left == 0 {
                return value;
            }
            if maximizing {
                alpha = alpha.max(value);
            } else {
                beta = beta.min(value);
            }
            if alpha >= beta {
                #[cfg(feature = "search-summary")]
                {
                    self.cutoffs += 1;
                }
                return value;
            }
            value
        };
        let noisy;
        let candidates: &[Move] = if in_check {
            moves
        } else {
            noisy = order_noisy(moves);
            &noisy
        };
        for &m in candidates {
            if !in_check && !see_ge(self.state.pos(), &m, 0) {
                continue;
            }
            self.apply(m);
            self.nodes += 1;
            self.memory.diagnostics.quiescence_node();
            let child_moves = legal_moves(self.state.pos());
            let value = match self.state.outcome(&child_moves) {
                Some(winner) => self.terminal_value(winner, ply + 1),
                None => self.quiescence(
                    alpha,
                    beta,
                    plies_left.saturating_sub(1),
                    ply + 1,
                    &child_moves,
                ),
            };
            self.undo();
            if maximizing {
                best_value = best_value.max(value);
                alpha = alpha.max(best_value);
            } else {
                best_value = best_value.min(value);
                beta = beta.min(best_value);
            }
            if alpha >= beta {
                #[cfg(feature = "search-summary")]
                {
                    self.cutoffs += 1;
                }
                break;
            }
        }
        best_value
    }
}
