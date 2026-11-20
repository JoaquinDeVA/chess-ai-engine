//! What the accumulators say about a position: the logit of the network for the side to move, and
//! the two readings of it, the integer score of the search and the expected score of the `eval`
//! command. The dense inputs of the threats are added to the mover's accumulator and the rest of the
//! network is `Net::finish`.

use shakmaty::{Chess, Color, Position};

use super::{slot, NetState};
use crate::eval::net::add_clipped_lanes;

/// Distance of the lowest and highest expected score from 0 and 1, to stay clear of the terminal
/// values 0 and 1.
const PROBABILITY_CLIP: f64 = 1e-2;

/// Units of the score per unit of log-odds: the score is in centipawns of the usual logistic scale
/// (`400 / ln 10`), so a pawn up is about 100 and the margins of the techniques read as such.
const SCORE_PER_LOGIT: f32 = 173.717_8;

/// Highest score of an evaluation, so that a mate is always beyond it: the clip of the expected
/// score to `[PROBABILITY_CLIP, 1 - PROBABILITY_CLIP]` on this scale (`ln 99 · 400 / ln 10`).
pub const EVAL_LIMIT: i32 = 798;

/// The logit in centipawns, clipped to `±EVAL_LIMIT`.
fn centipawns(logit: f32) -> i32 {
    ((logit * SCORE_PER_LOGIT).round() as i32).clamp(-EVAL_LIMIT, EVAL_LIMIT)
}

/// Logistic function, stable for large magnitudes.
fn sigmoid(z: f64) -> f64 {
    0.5 * (1.0 + (0.5 * z).tanh())
}

impl NetState<'_> {
    /// Score of `player` in `pos`, the position the state is following: the logit of the side to
    /// move in centipawns, clipped to `±EVAL_LIMIT`.
    pub fn score(&mut self, pos: &Chess, player: Color) -> i32 {
        let score = centipawns(self.logit(pos));
        if pos.turn() == player {
            score
        } else {
            -score
        }
    }

    /// Expected score of `player` in `pos`, in `[0.01, 0.99]`.
    pub fn evaluate(&mut self, pos: &Chess, player: Color) -> f64 {
        let mover = sigmoid(f64::from(self.logit(pos)));
        let expected = if pos.turn() == player {
            mover
        } else {
            1.0 - mover
        };
        expected.clamp(PROBABILITY_CLIP, 1.0 - PROBABILITY_CLIP)
    }

    /// Adds the dense inputs of the position (the threats) to the first `width` values of `hidden`,
    /// which hold the mover's accumulator: they have no opponent view.
    fn add_dense(&mut self, pos: &Chess) {
        let net = self.net;
        self.dense_indices.clear();
        self.dense_values.clear();
        net.features()
            .dense_inputs(pos, &mut self.dense_indices, &mut self.dense_values);
        for (&index, &value) in self.dense_indices.iter().zip(&self.dense_values) {
            if value == 0.0 {
                continue;
            }
            for (h, w) in self.hidden[..self.width]
                .iter_mut()
                .zip(net.first_row(index))
            {
                *h += w * value as f32;
            }
        }
    }

    /// Network output before the logistic function, for the side to move in `pos`, the position
    /// the state is following.
    pub(super) fn logit(&mut self, pos: &Chess) -> f32 {
        let net = self.net;
        let width = self.width;
        let own = (2 * self.top + slot(pos.turn())) * width;
        let rival = (2 * self.top + slot(!pos.turn())) * width;
        let has_dense = net.features().has_dense();
        // both views, one shared first layer: the mover's, with the dense inputs, and the rival's
        if has_dense {
            self.hidden[..width].copy_from_slice(&self.stack[own..own + width]);
            self.add_dense(pos);
        }
        if let (Some(out), true) = (net.direct_output(), width.is_multiple_of(8)) {
            // one output layer straight over the accumulators: no copy of the rival's
            let mover = if has_dense {
                &self.hidden[..width]
            } else {
                &self.stack[own..own + width]
            };
            let mut lanes = [0.0f32; 8];
            add_clipped_lanes(&mut lanes, mover, &out.weights[..width]);
            add_clipped_lanes(
                &mut lanes,
                &self.stack[rival..rival + width],
                &out.weights[width..2 * width],
            );
            return out.bias[0] + lanes.iter().sum::<f32>();
        }
        if !has_dense {
            self.hidden[..width].copy_from_slice(&self.stack[own..own + width]);
        }
        self.hidden[width..2 * width].copy_from_slice(&self.stack[rival..rival + width]);
        net.finish(&mut self.hidden, &mut self.spare)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_score_is_the_logit_in_centipawns_clipped_like_the_expected_score() {
        assert_eq!(centipawns(1.0), 174);
        assert_eq!(centipawns(-1.0), -174);
        // 0.99 of expected score is the highest evaluation in both readings
        assert_eq!(centipawns((0.99f64 / 0.01).ln() as f32), EVAL_LIMIT);
        assert_eq!(centipawns(10.0), EVAL_LIMIT);
        assert_eq!(centipawns(-10.0), -EVAL_LIMIT);
    }

    #[test]
    fn the_expected_score_of_both_players_adds_up_to_one() {
        assert!((sigmoid(0.7) + sigmoid(-0.7) - 1.0).abs() < 1e-12);
    }
}
