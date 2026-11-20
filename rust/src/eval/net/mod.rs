//! The sparse network: its layers. The first layer is the accumulator that
//! [`NetState`](super::NetState) keeps move by move, and the rest is `forward.rs`; reading a network
//! from a file is `json.rs`. The rest of `eval/` reads the layers through the methods below.

mod forward;
mod json;

pub(super) use forward::add_clipped_lanes;

use super::features::FeatureSet;

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
    features: FeatureSet,
    layers: Vec<Layer>,
    /// Widest layer before the output: the size of the buffers of an evaluation.
    max_width: usize,
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

    /// The feature set of the inputs.
    pub(super) fn features(&self) -> FeatureSet {
        self.features
    }

    /// Width of the accumulator: the units of the first layer.
    pub(super) fn width(&self) -> usize {
        self.layers[0].fan_out
    }

    /// Row of the first layer for input `index`: what the input adds to an accumulator.
    pub(super) fn first_row(&self, index: usize) -> &[f32] {
        self.layers[0].row(index)
    }

    /// Bias of the first layer: where an accumulator starts.
    pub(super) fn first_bias(&self) -> &[f32] {
        &self.layers[0].bias
    }

    /// Length of the buffers an evaluation needs.
    pub(super) fn buffer_len(&self) -> usize {
        self.max_width
    }

    /// The output layer when it follows the accumulator directly, with no hidden layer between.
    pub(super) fn direct_output(&self) -> Option<&Layer> {
        (self.layers.len() == 2).then(|| &self.layers[1])
    }
}
