//! Evaluation: the sparse network, which returns the expected score of a player in `[0.01, 0.99]`.

mod accumulator;
pub mod features;
mod net;

pub use accumulator::NetState;
pub use features::FeatureSet;
pub use net::{Layer, Net};

use shakmaty::Color;

/// Distance of the lowest and highest evaluation from 0 and 1, to stay clear of the terminal
/// values 0 and 1.
pub const MARGIN: f64 = 1e-2;

/// Expected score of White turned into the value of `player`, kept within the margin.
fn for_player(white_score: f64, player: Color) -> f64 {
    let score = if player == Color::White {
        white_score
    } else {
        1.0 - white_score
    };
    (1.0 - MARGIN).min(MARGIN.max(score))
}

/// Logistic function, stable for large magnitudes.
pub fn sigmoid(z: f64) -> f64 {
    0.5 * (1.0 + (0.5 * z).tanh())
}
