//! The sparse network: the accumulator, which [`NetState`](super::NetState) keeps move by move,
//! and the output layer over both sides' accumulators, `forward.rs`; reading a network from a
//! file is `json.rs`. The rest of `eval/` reads the layers through the methods below.

mod forward;
mod json;

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
    fn row(&self, index: usize) -> &[f32] {
        &self.weights[index * self.fan_out..(index + 1) * self.fan_out]
    }
}

/// A network over sparse inputs: the accumulator and one output unit over both sides' accumulators.
pub struct Net {
    features: FeatureSet,
    accumulator: Layer,
    output: Layer,
}

impl Net {
    /// A network of two layers: the accumulator, whose width is a multiple of 8, and one output
    /// unit that reads both sides' accumulators, which share the accumulator's weights (its
    /// `fan_in` is twice the accumulator width). The opponent's view of a piece is the other block
    /// of the same square; the dense inputs (the threats) are the mover's alone and only reach the
    /// mover's half. The shapes of the layers are checked.
    pub fn new(features: FeatureSet, layers: Vec<Layer>) -> Result<Net, String> {
        let [accumulator, output]: [Layer; 2] = layers
            .try_into()
            .map_err(|_| "a network has an accumulator and an output layer, nothing else")?;
        if accumulator.fan_in != features.size() {
            return Err(format!(
                "the first layer has {} inputs, the features give {}",
                accumulator.fan_in,
                features.size()
            ));
        }
        if !accumulator.fan_out.is_multiple_of(8) {
            return Err("the accumulator width must be a multiple of 8".into());
        }
        if output.fan_in != 2 * accumulator.fan_out {
            return Err("consecutive layers do not fit".into());
        }
        for layer in [&accumulator, &output] {
            if layer.weights.len() != layer.fan_in * layer.fan_out
                || layer.bias.len() != layer.fan_out
            {
                return Err("a layer has the wrong number of weights".into());
            }
        }
        if output.fan_out != 1 {
            return Err("the output layer must have one unit".into());
        }
        Ok(Net {
            features,
            accumulator,
            output,
        })
    }

    /// The feature set of the inputs.
    pub(super) fn features(&self) -> FeatureSet {
        self.features
    }

    /// Width of the accumulator: the units of the first layer.
    pub(super) fn width(&self) -> usize {
        self.accumulator.fan_out
    }

    /// Row of the first layer for input `index`: what the input adds to an accumulator.
    pub(super) fn first_row(&self, index: usize) -> &[f32] {
        self.accumulator.row(index)
    }

    /// Bias of the first layer: where an accumulator starts.
    pub(super) fn first_bias(&self) -> &[f32] {
        &self.accumulator.bias
    }
}

/// The network of the tests of `eval/`: random weights of a fixed seed, so that every input row
/// and both halves of the output layer count.
#[cfg(test)]
pub(in crate::eval) mod dummy {
    use super::{Layer, Net};
    use crate::eval::FeatureSet;

    /// A layer of weights in `[-0.5, 0.5)` and biases in `[-0.25, 0.25)` drawn from `seed`.
    pub(in crate::eval) fn layer(fan_in: usize, fan_out: usize, seed: u64) -> Layer {
        let mut state = seed;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (((state >> 33) as f64 / (1u64 << 31) as f64) - 0.5) as f32
        };
        Layer {
            fan_in,
            fan_out,
            weights: (0..fan_in * fan_out).map(|_| next()).collect(),
            bias: (0..fan_out).map(|_| next() * 0.5).collect(),
        }
    }

    /// A network of `features` with an accumulator of 16 units.
    pub(in crate::eval) fn net(features: FeatureSet) -> Net {
        Net::new(
            features,
            vec![layer(features.size(), 16, 1), layer(32, 1, 2)],
        )
        .unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::dummy::layer;
    use super::*;
    use crate::eval::Families;

    const PIECES: FeatureSet = FeatureSet::PstCtx(Families {
        material: false,
        pawn_files: false,
        king_buckets: false,
    });

    #[test]
    fn the_accumulator_width_is_a_multiple_of_8() {
        assert!(Net::new(PIECES, vec![layer(768, 8, 1), layer(16, 1, 2)]).is_ok());
        assert!(Net::new(PIECES, vec![layer(768, 12, 1), layer(24, 1, 2)]).is_err());
    }

    #[test]
    fn a_network_has_an_accumulator_and_an_output_layer() {
        let three = vec![layer(768, 8, 1), layer(16, 8, 2), layer(8, 1, 3)];
        assert!(Net::new(PIECES, three).is_err());
        assert!(Net::new(PIECES, vec![layer(768, 1, 1)]).is_err());
    }
}
