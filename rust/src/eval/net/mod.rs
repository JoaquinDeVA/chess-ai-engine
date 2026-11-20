//! The sparse network: the layers and how to ask them. The first layer is the accumulator that
//! [`NetState`](super::NetState) keeps move by move; this file builds the network and evaluates a
//! position from the board. Reading it from a file is `json.rs`, the end of the network `forward.rs`.

mod forward;
mod json;

use shakmaty::{Chess, Color, Position};

use super::features::FeatureSet;
use super::NetState;
use super::{for_player, sigmoid};

/// One dense layer: `weights` is row-major `(fan_in, fan_out)`.
pub struct Layer {
    pub fan_in: usize,
    pub fan_out: usize,
    pub weights: Vec<f32>,
    pub bias: Vec<f32>,
}

impl Layer {
    /// The row of input `index`: its `fan_out` weights.
    pub(super) fn row(&self, index: usize) -> &[f32] {
        &self.weights[index * self.fan_out..(index + 1) * self.fan_out]
    }
}

/// A feed-forward network over sparse inputs; the first layer is the accumulator.
pub struct Net {
    pub(super) features: FeatureSet,
    pub(super) layers: Vec<Layer>,
    /// Widest layer before the output: the size of the buffers of an evaluation.
    pub(super) max_width: usize,
}

impl Net {
    /// A network whose second layer reads both sides' accumulators, which share the first layer's
    /// weights (`layers[1].fan_in` is twice the accumulator width). The opponent's view of a piece is
    /// the other block of the same square; the dense inputs (the threats) are the mover's alone and
    /// only reach the mover's half. The shapes of the layers are checked.
    pub fn new(features: FeatureSet, layers: Vec<Layer>) -> Result<Net, String> {
        if layers.len() < 2 {
            return Err("a network needs an accumulator and an output layer".into());
        }
        if layers[0].fan_in != features.size() {
            return Err(format!(
                "the first layer has {} inputs, the features give {}",
                layers[0].fan_in,
                features.size()
            ));
        }
        for (position, pair) in layers.windows(2).enumerate() {
            let wanted = if position == 0 {
                2 * pair[0].fan_out
            } else {
                pair[0].fan_out
            };
            if wanted != pair[1].fan_in {
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
        let max_width = layers
            .iter()
            .map(|l| l.fan_out)
            .max()
            .unwrap_or(1)
            .max(2 * layers[0].fan_out);
        Ok(Net {
            features,
            layers,
            max_width,
        })
    }

    /// Network output before the logistic function, for the side to move, from the board: the
    /// accumulators are built from scratch, where the search follows them move by move with
    /// [`NetState`](super::NetState).
    pub fn logit(&self, pos: &Chess) -> f32 {
        NetState::new(self, pos).logit(pos)
    }

    /// Expected score of `player` in `pos`, in `[0.01, 0.99]`, from the board.
    pub fn evaluate(&self, pos: &Chess, player: Color) -> f64 {
        let mover = sigmoid(f64::from(self.logit(pos)));
        let white = if pos.turn() == Color::White {
            mover
        } else {
            1.0 - mover
        };
        for_player(white, player)
    }
}
