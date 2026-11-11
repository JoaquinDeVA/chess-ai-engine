//! Evaluation: the sparse network. The search reads it as an integer score of the side it asks
//! for; Python and the `eval` command read it as an expected score in `[0.01, 0.99]`.

mod accumulator;
pub mod features;
mod net;

pub use accumulator::NetState;
pub use features::FeatureSet;
pub use net::{Layer, Net};

use shakmaty::Color;

/// Distance of the lowest and highest expected score from 0 and 1, to stay clear of the terminal
/// values 0 and 1.
pub const MARGIN: f64 = 1e-2;

/// Units of the score per unit of log-odds: the score is in centipawns of the usual logistic scale
/// (`400 / ln 10`), so a pawn up is about 100 and the margins of the techniques read as such.
pub const SCORE_PER_LOGIT: f32 = 173.717_8;

/// Highest score of an evaluation, so that a mate is always beyond it: the clip of the expected
/// score to `[MARGIN, 1 - MARGIN]` on this scale (`ln 99 · 400 / ln 10`).
pub const EVAL_LIMIT: i32 = 798;

/// Expected score of White turned into the value of `player`, kept within the margin.
fn for_player(white_score: f64, player: Color) -> f64 {
    let score = if player == Color::White {
        white_score
    } else {
        1.0 - white_score
    };
    (1.0 - MARGIN).min(MARGIN.max(score))
}

/// Score of `player` from the logit of the side to move, `mover`, clipped to the evaluation range.
fn score(logit: f32, mover: Color, player: Color) -> i32 {
    let score = ((logit * SCORE_PER_LOGIT).round() as i32).clamp(-EVAL_LIMIT, EVAL_LIMIT);
    if mover == player {
        score
    } else {
        -score
    }
}

/// Logistic function, stable for large magnitudes.
pub fn sigmoid(z: f64) -> f64 {
    0.5 * (1.0 + (0.5 * z).tanh())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_score_is_the_logit_in_centipawns_clipped_like_the_expected_score() {
        assert_eq!(score(1.0, Color::White, Color::White), 174);
        assert_eq!(score(1.0, Color::White, Color::Black), -174);
        // 0.99 of expected score is the highest evaluation in both readings.
        let top = (0.99f64 / 0.01).ln() as f32;
        assert_eq!(score(top, Color::Black, Color::Black), EVAL_LIMIT);
        assert_eq!(score(10.0, Color::Black, Color::White), -EVAL_LIMIT);
    }
}
