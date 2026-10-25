//! The sparse network `net4_pst`: slotted inputs, an accumulator of clipped sums and dense layers.

use shakmaty::{Chess, Color, Position};

use super::linear::{FeatureSet, Perspective};
use super::{for_player, sigmoid, Evaluator};

/// One dense layer: `weights` is row-major `(fan_in, fan_out)`.
pub struct Layer {
    pub fan_in: usize,
    pub fan_out: usize,
    pub weights: Vec<f64>,
    pub bias: Vec<f64>,
}

/// A feed-forward network over sparse inputs; the first layer is the accumulator.
pub struct Net {
    features: FeatureSet,
    perspective: Perspective,
    layers: Vec<Layer>,
}

impl Net {
    /// A network, checking the shapes of its layers.
    pub fn new(
        features: FeatureSet,
        perspective: Perspective,
        layers: Vec<Layer>,
    ) -> Result<Net, String> {
        if layers.len() < 2 {
            return Err("a network needs an accumulator and an output layer".into());
        }
        if layers[0].fan_in != features.size(perspective) {
            return Err(format!(
                "the first layer has {} inputs, the features give {}",
                layers[0].fan_in,
                features.size(perspective)
            ));
        }
        for pair in layers.windows(2) {
            if pair[0].fan_out != pair[1].fan_in {
                return Err("consecutive layers do not fit".into());
            }
        }
        for layer in &layers {
            if layer.weights.len() != layer.fan_in * layer.fan_out
                || layer.bias.len() != layer.fan_out
            {
                return Err("a layer has the wrong number of weights".into());
            }
        }
        if layers.last().map(|l| l.fan_out) != Some(1) {
            return Err("the output layer must have one unit".into());
        }
        Ok(Net {
            features,
            perspective,
            layers,
        })
    }

    /// Network output before the logistic function.
    pub fn logit(&self, pos: &Chess) -> f64 {
        let (mut indices, mut values) = (Vec::with_capacity(40), Vec::with_capacity(40));
        self.features
            .active(pos, self.perspective, &mut indices, &mut values);
        let first = &self.layers[0];
        let mut hidden = first.bias.clone();
        for (index, value) in indices.iter().zip(&values) {
            let row = &first.weights[index * first.fan_out..(index + 1) * first.fan_out];
            for (h, w) in hidden.iter_mut().zip(row) {
                *h += w * value;
            }
        }
        for h in &mut hidden {
            *h = h.clamp(0.0, 1.0);
        }
        let last = self.layers.len() - 1;
        for layer in &self.layers[1..last] {
            hidden = dense(layer, &hidden, true);
        }
        dense(&self.layers[last], &hidden, false)[0]
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

fn dense(layer: &Layer, input: &[f64], relu: bool) -> Vec<f64> {
    let mut out = layer.bias.clone();
    for (i, x) in input.iter().enumerate() {
        let row = &layer.weights[i * layer.fan_out..(i + 1) * layer.fan_out];
        for (o, w) in out.iter_mut().zip(row) {
            *o += x * w;
        }
    }
    if relu {
        for o in &mut out {
            *o = o.max(0.0);
        }
    }
    out
}

impl Evaluator for Net {
    fn evaluate(&self, pos: &Chess, player: Color) -> f64 {
        for_player(self.expected_score(pos), player)
    }
}
