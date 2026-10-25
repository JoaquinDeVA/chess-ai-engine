#![allow(clippy::result_large_err)] // the error types of shakmaty are large
//! Core of the chess engine: rules on top of `shakmaty`, evaluations, alpha-beta search and the
//! knowledge (opening book, endgame tables). The Python package `chess_engine` calls it through
//! the extension module `chess_engine._core` (feature `python`).

pub mod board;
pub mod eval;
pub mod search;
