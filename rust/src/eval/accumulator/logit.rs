//! What the accumulators say about a position: the logit of the network for the side to move, and
//! the two readings of it, the integer score of the search and the expected score of the `eval`
//! command. The dense inputs of the threats are added to the mover's accumulator and the output
//! layer is `Net::output`.

use shakmaty::{Chess, Color, Position};

use super::{slot, NetState};

/// Distance of the lowest and highest expected score from 0 and 1, to stay clear of the terminal
/// values 0 and 1.
const PROBABILITY_CLIP: f64 = 1e-2;

/// Units of the score per unit of log-odds: the score is in centipawns of the usual logistic scale
/// (`400 / ln 10`), so a pawn up is about 100 and the margins of the techniques read as such. It is
/// not the scale of the training targets (the Lichess curve, about 271.6 per logit): the margins of
/// the search were tuned by SPRT on this one, so it only changes with a new SPRT of all of them.
const SCORE_PER_LOGIT: f32 = 173.717_8;

/// Highest score of an evaluation, so that a mate is always beyond it. It is its own bound of the
/// search, not derived from `PROBABILITY_CLIP`: it only has to stay below `MATE_BOUND`.
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

    /// Network output before the logistic function, for the side to move in `pos`, the position
    /// the state is following. Both views read the same first layer: the mover's accumulator, with
    /// the dense inputs added (they have no opponent view), and the rival's.
    pub(super) fn logit(&mut self, pos: &Chess) -> f32 {
        let net = self.net;
        let width = self.width;
        let own = (2 * self.top + slot(pos.turn())) * width;
        let rival = (2 * self.top + slot(!pos.turn())) * width;
        let own = &self.stack[own..own + width];
        let rival = &self.stack[rival..rival + width];
        let pieces = pos.board().occupied().count();
        let Some(dense) = net.features().dense_inputs(pos) else {
            return net.output(own, rival, pieces);
        };
        self.with_dense.copy_from_slice(own);
        for (row, value) in dense {
            if value == 0.0 {
                continue;
            }
            for (h, w) in self.with_dense.iter_mut().zip(net.first_row(row)) {
                *h += w * value;
            }
        }
        net.output(&self.with_dense, rival, pieces)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_score_is_the_logit_in_centipawns_clipped_like_the_expected_score() {
        assert_eq!(centipawns(1.0), 174);
        assert_eq!(centipawns(-1.0), -174);
        assert_eq!(centipawns(10.0), EVAL_LIMIT);
        assert_eq!(centipawns(-10.0), -EVAL_LIMIT);
    }

    #[test]
    fn the_score_per_logit_is_the_elo_scale() {
        let elo_scale = 400.0 / std::f64::consts::LN_10;
        assert!((f64::from(SCORE_PER_LOGIT) - elo_scale).abs() < 1e-4);
    }

    #[test]
    fn the_expected_score_of_both_players_adds_up_to_one() {
        assert!((sigmoid(0.7) + sigmoid(-0.7) - 1.0).abs() < 1e-12);
    }
}
