//! Alpha-beta search with iterative deepening.
//!
//! The values are integer scores of the root player: the evaluation of the network in centipawns
//! of the logistic scale, within `±EVAL_LIMIT`, and a mate `n` plies away as `±(MATE - n)`, so
//! that quicker wins and slower losses are preferred; a draw is 0. The search keeps a lock-free
//! transposition table and killer moves between iterations and between the moves of a game, and
//! ends every branch at the horizon with a quiescence search over captures and promotions.

mod deepening;
mod memory;
mod node;
mod options;
mod ordering;
mod quiescence;
mod search_techniques;
mod stats;
mod tt;

pub use deepening::iterative_deepening;
pub use memory::*;
pub use node::{alpha_beta, SearchResult, INFINITE, MATE, MATE_BOUND};
pub use options::*;
pub use stats::*;

#[cfg(test)]
mod tests;
