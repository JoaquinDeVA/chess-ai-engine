//! Alpha-beta search with iterative deepening.
//!
//! The values are expected scores of the root player in `[0, 1]`; terminal values are 1, 0.5 and
//! 0, nudged by `epsilon` per ply so that quicker wins and slower losses are preferred. The
//! search keeps a bounded transposition table and killer moves between iterations and between the
//! moves of a game, and ends every branch at the horizon with a quiescence search over captures
//! and promotions.

mod deepening;
pub mod estimators;
mod node;
mod options;
pub mod shared;
pub mod tt;

pub use deepening::iterative_deepening;
pub use node::{alpha_beta, alpha_beta_window, SearchResult};
pub use options::*;

#[cfg(test)]
mod tests;
