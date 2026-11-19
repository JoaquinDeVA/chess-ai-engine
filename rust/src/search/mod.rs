//! Alpha-beta search with iterative deepening.
//!
//! The values are integer scores of the root player: the evaluation of the network in centipawns
//! of the logistic scale, within `±EVAL_LIMIT`, and a mate `n` plies away as `±(MATE - n)`, so
//! that quicker wins and slower losses are preferred; a draw is 0. The search keeps a lock-free
//! transposition table, killers and history between iterations and between the moves of a game, and
//! ends every branch at the horizon with a quiescence search over captures and promotions.

mod alpha_beta;
mod control;
mod deepening;
mod diagnostics;
mod memory;
mod options;
mod ordering;
mod scores;
mod stats;
mod transposition_table;

pub use alpha_beta::{alpha_beta, SearchResult};
pub use control::*;
pub use deepening::iterative_deepening;
pub use diagnostics::*;
pub use memory::*;
pub use options::*;
pub use scores::{INFINITE, MATE, MATE_BOUND};
pub use stats::*;

#[cfg(test)]
mod tests;
