//! The logistic (linear) model over sparse features.
//!
//! It scores a position as `bias + sum(weights[index] * value)` and the logistic function turns
//! the score into the expected score.

use shakmaty::{Chess, Color, Position};

use super::features::{FeatureSet, Perspective};
use super::{for_player, sigmoid, Evaluator};

/// A logistic model: `sigmoid(bias + sum(weights[index] * value))`.
pub struct LinearModel {
    features: FeatureSet,
    perspective: Perspective,
    weights: Vec<f64>,
    bias: f64,
}

impl LinearModel {
    /// A model, checking that the weights fit the feature set.
    pub fn new(
        features: FeatureSet,
        perspective: Perspective,
        weights: Vec<f64>,
        bias: f64,
    ) -> Result<LinearModel, String> {
        if perspective == Perspective::White && features == FeatureSet::KingBucket {
            return Err(
                "king_bucket features are defined for the side_to_move perspective only".into(),
            );
        }
        let expected = features.size(perspective);
        if weights.len() != expected {
            return Err(format!(
                "expected {expected} weights, found {}",
                weights.len()
            ));
        }
        Ok(LinearModel {
            features,
            perspective,
            weights,
            bias,
        })
    }

    /// Linear score of a position, before the logistic function.
    pub fn logit(&self, pos: &Chess) -> f64 {
        let (mut indices, mut values) = (Vec::with_capacity(70), Vec::with_capacity(70));
        self.features
            .active(pos, self.perspective, &mut indices, &mut values);
        let mut total = self.bias;
        for (index, value) in indices.iter().zip(&values) {
            total += self.weights[*index] * value;
        }
        total
    }

    /// Expected score of White in `[0, 1]`.
    pub fn expected_score(&self, pos: &Chess) -> f64 {
        let score = sigmoid(self.logit(pos));
        if self.perspective == Perspective::SideToMove && pos.turn() == Color::Black {
            1.0 - score
        } else {
            score
        }
    }
}

impl Evaluator for LinearModel {
    fn evaluate(&self, pos: &Chess, player: Color) -> f64 {
        for_player(self.expected_score(pos), player)
    }
}
