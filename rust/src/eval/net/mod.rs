//! The sparse network: the accumulator, which [`NetState`](super::NetState) keeps move by move,
//! and the output layer over both sides' accumulators, `forward.rs`; reading a network from a
//! file is `json.rs`. The rest of `eval/` reads the layers through the methods below.

mod forward;
mod json;

use super::features::FeatureSet;

/// Most units of the output layer: one per piece count.
const MAX_BUCKETS: usize = 32;

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

/// A network over sparse inputs: the accumulator and an output layer over both sides'
/// accumulators, with one unit per material bucket.
pub struct Net {
    features: FeatureSet,
    accumulator: Layer,
    /// The output weights by bucket: bucket `b` is `columns[b * fan_in..(b + 1) * fan_in]`, the
    /// transpose of the row-major layer, so that one evaluation reads contiguous weights.
    columns: Vec<f32>,
    /// One bias per bucket.
    output_bias: Vec<f32>,
}

impl Net {
    /// A network of two layers: the accumulator, whose width is a multiple of 8, and an output
    /// layer that reads both sides' accumulators, which share the accumulator's weights (its
    /// `fan_in` is twice the accumulator width). The output has one unit per material bucket,
    /// from 1 to 32 (see [`Net::bucket`]). The opponent's view of a piece is the other block
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
        if !(1..=MAX_BUCKETS).contains(&output.fan_out) {
            return Err(format!(
                "the output layer must have 1 to {MAX_BUCKETS} units"
            ));
        }
        let columns = (0..output.fan_out)
            .flat_map(|b| (0..output.fan_in).map(move |i| (b, i)))
            .map(|(b, i)| output.weights[i * output.fan_out + b])
            .collect();
        Ok(Net {
            features,
            accumulator,
            columns,
            output_bias: output.bias,
        })
    }

    /// The output unit for a position with `pieces` pieces on the board (kings included, 2 to
    /// 32): `(pieces - 1) * buckets / 32`, so that eight buckets take four piece counts each, the
    /// first from 2 to 4 pieces and the last from 29 to 32; one bucket is always unit 0.
    pub(in crate::eval) fn bucket(&self, pieces: usize) -> usize {
        (pieces.saturating_sub(1) * self.output_bias.len() / MAX_BUCKETS)
            .min(self.output_bias.len() - 1)
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

    /// A network of `features` with an accumulator of 16 units and eight material buckets.
    pub(in crate::eval) fn net(features: FeatureSet) -> Net {
        Net::new(
            features,
            vec![layer(features.size(), 16, 1), layer(32, 8, 2)],
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

    #[test]
    fn the_output_has_one_to_32_buckets_of_four_piece_counts_each_with_8() {
        assert!(Net::new(PIECES, vec![layer(768, 8, 1), layer(16, 33, 2)]).is_err());
        let net = Net::new(PIECES, vec![layer(768, 8, 1), layer(16, 8, 2)]).unwrap();
        let buckets: Vec<usize> = [2, 4, 5, 8, 9, 28, 29, 32].map(|p| net.bucket(p)).to_vec();
        assert_eq!(buckets, [0, 0, 1, 1, 2, 6, 7, 7]);
        let one = Net::new(PIECES, vec![layer(768, 8, 1), layer(16, 1, 2)]).unwrap();
        assert_eq!(one.bucket(32), 0);
    }

    #[test]
    fn each_bucket_reads_its_column_of_the_row_major_output_layer() {
        let output = layer(16, 8, 2);
        let (weights, bias) = (output.weights.clone(), output.bias.clone());
        let net = Net::new(PIECES, vec![layer(768, 8, 1), output]).unwrap();
        let mover: Vec<f32> = (0..8).map(|i| i as f32 / 7.0).collect();
        let rival: Vec<f32> = (0..8).map(|i| 1.5 - i as f32 / 4.0).collect();
        for pieces in [3, 7, 12, 18, 21, 26, 30, 32] {
            let b = net.bucket(pieces);
            let clipped = mover.iter().chain(&rival).map(|x| x.clamp(0.0, 1.0));
            let expected: f32 = bias[b]
                + clipped
                    .enumerate()
                    .map(|(i, x)| x * weights[i * 8 + b])
                    .sum::<f32>();
            assert!((net.output(&mover, &rival, pieces) - expected).abs() < 1e-5);
        }
    }
}
