//! Evaluation: the sparse network, which returns the expected score of a player in `[0.01, 0.99]`.

pub mod sparse_features;
pub mod sparse_net;

use shakmaty::{Chess, Color, Move};

/// Distance of the lowest and highest evaluation from 0 and 1, to stay clear of the terminal values 0 and 1.
pub const MARGIN: f64 = 1e-2;

/// A position evaluation seen from a player.
pub trait Evaluator: Send + Sync {
    /// Expected score of `player` in `pos`, in `[0.01, 0.99]`.
    fn evaluate(&self, pos: &Chess, player: Color) -> f64;

    /// An evaluation state that follows the moves of the search instead of starting from the
    /// board at every leaf, for the evaluations that can (`None` for the others).
    fn new_state<'a>(&'a self, _pos: &Chess) -> Option<Box<dyn EvalState + 'a>> {
        None
    }
}

/// An evaluation that is updated move by move.
///
/// The search calls `apply` for every move it plays and `undo` when it takes it back, and
/// `evaluate` for the position it is in; the result equals the evaluator's own `evaluate` up to
/// rounding.
pub trait EvalState {
    /// The side to move in `pos` (the position before the move) plays `m`.
    fn apply(&mut self, pos: &Chess, m: &Move);
    /// The side to move passes (a null move).
    fn apply_null(&mut self);
    /// Take back the last move played with `apply` or `apply_null`.
    fn undo(&mut self);
    /// Expected score of `player` in `pos`, the position the state is following.
    fn evaluate(&self, pos: &Chess, player: Color) -> f64;
}

/// Expected score of White turned into the value of `player`, kept within the margin.
pub fn for_player(white_score: f64, player: Color) -> f64 {
    let score = if player == Color::White {
        white_score
    } else {
        1.0 - white_score
    };
    (1.0 - MARGIN).min(MARGIN.max(score))
}

/// Logistic function, stable for large magnitudes (as the Python `_sigmoid`).
pub fn sigmoid(z: f64) -> f64 {
    0.5 * (1.0 + (0.5 * z).tanh())
}

/// The evaluations the search can run natively.
pub enum Eval {
    Net(sparse_net::Net),
}

impl Evaluator for Eval {
    fn evaluate(&self, pos: &Chess, player: Color) -> f64 {
        let Eval::Net(e) = self;
        e.evaluate(pos, player)
    }

    fn new_state<'a>(&'a self, pos: &Chess) -> Option<Box<dyn EvalState + 'a>> {
        let Eval::Net(e) = self;
        e.new_state(pos)
    }
}
