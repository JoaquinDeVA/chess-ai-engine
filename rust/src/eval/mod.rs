//! Evaluations: the hand-written heuristic `h1`, the linear (logistic) models and the sparse
//! network. All return the expected score of a player in `[0.01, 0.99]`, like the Python
//! `ModelEvaluator`.

pub mod h1;
pub mod linear;
pub mod net;

use shakmaty::{Chess, Color};

/// Lowest and highest value of a heuristic, to stay clear of the terminal values 0 and 1.
pub const MARGIN: f64 = 1e-2;

/// A position evaluation seen from a player.
pub trait Evaluator: Send + Sync {
    /// Expected score of `player` in `pos`, in `[0.01, 0.99]`.
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
    H1(h1::H1),
    Linear(linear::LinearModel),
    Net(net::Net),
}

impl Evaluator for Eval {
    fn evaluate(&self, pos: &Chess, player: Color) -> f64 {
        match self {
            Eval::H1(e) => e.evaluate(pos, player),
            Eval::Linear(e) => e.evaluate(pos, player),
            Eval::Net(e) => e.evaluate(pos, player),
        }
    }
}
