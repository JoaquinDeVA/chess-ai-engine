//! The sparse network: the accumulator, which [`NetState`](super::NetState) keeps move by move,
//! and the output over both sides' accumulators (a linear unit or a hidden head per material
//! bucket), `forward.rs`; reading a network from a file is `json.rs`. The rest of `eval/` reads
//! the layers through the methods below.

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

/// A network over sparse inputs: the accumulator and an output over both sides' accumulators,
/// with one linear unit or one hidden head per material bucket.
pub struct Net {
    features: FeatureSet,
    accumulator: Layer,
    /// The weights of the second layer by unit: unit `u` is `columns[u * fan_in..(u + 1) *
    /// fan_in]`, the transpose of the row-major layer, so that one evaluation reads contiguous
    /// weights. Without a head unit `b` is the output of bucket `b`; with one, units `b * width`
    /// to `(b + 1) * width - 1` are the hidden units of bucket `b`.
    columns: Vec<f32>,
    /// One bias per unit of the second layer.
    output_bias: Vec<f32>,
    /// Units of the output: one per material bucket.
    buckets: usize,
    head: Option<Head>,
}

/// The last layer of a network with a hidden head per bucket: one linear unit per bucket over
/// that bucket's `width` hidden units.
struct Head {
    width: usize,
    /// Bucket `b` is `weights[b * width..(b + 1) * width]`, the transpose of the row-major layer.
    weights: Vec<f32>,
    /// One bias per bucket.
    bias: Vec<f32>,
}

impl Net {
    /// A network of two or three layers: the accumulator, whose width is a multiple of 8, and a
    /// layer that reads both sides' accumulators, which share the accumulator's weights (its
    /// `fan_in` is twice the accumulator width). The opponent's view of a piece is the other block
    /// of the same square; the dense inputs (the threats) are the mover's alone and only reach the
    /// mover's half. With two layers the second is the output, one unit per material bucket,
    /// from 1 to 32 (see [`Net::bucket`]). With three, the third is the output, `(width,
    /// buckets)`, and the second holds the `width` hidden units of every bucket, bucket after
    /// bucket, which are clipped to `[0, 1]` like the accumulator. The shapes are checked.
    pub fn new(features: FeatureSet, layers: Vec<Layer>) -> Result<Net, String> {
        let mut layers = layers.into_iter();
        let (Some(accumulator), Some(output), last, None) =
            (layers.next(), layers.next(), layers.next(), layers.next())
        else {
            return Err("a network has an accumulator, an output layer and at most a head".into());
        };
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
        for layer in [Some(&accumulator), Some(&output), last.as_ref()]
            .into_iter()
            .flatten()
        {
            if layer.weights.len() != layer.fan_in * layer.fan_out
                || layer.bias.len() != layer.fan_out
            {
                return Err("a layer has the wrong number of weights".into());
            }
        }
        let buckets = last.as_ref().map_or(output.fan_out, |head| head.fan_out);
        if !(1..=MAX_BUCKETS).contains(&buckets) {
            return Err(format!(
                "the output layer must have 1 to {MAX_BUCKETS} units"
            ));
        }
        if last
            .as_ref()
            .is_some_and(|head| head.fan_in * buckets != output.fan_out)
        {
            return Err("the hidden layer must have the head's units for every bucket".into());
        }
        let head = last.map(|layer| Head {
            width: layer.fan_in,
            weights: transpose(&layer),
            bias: layer.bias,
        });
        Ok(Net {
            features,
            columns: transpose(&output),
            output_bias: output.bias,
            accumulator,
            buckets,
            head,
        })
    }

    /// The output unit for a position with `pieces` pieces on the board (kings included, 2 to
    /// 32): `(pieces - 1) * buckets / 32`, so that eight buckets take four piece counts each, the
    /// first from 2 to 4 pieces and the last from 29 to 32; one bucket is always unit 0.
    pub(in crate::eval) fn bucket(&self, pieces: usize) -> usize {
        (pieces.saturating_sub(1) * self.buckets / MAX_BUCKETS).min(self.buckets - 1)
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

/// The weights of `layer` by unit: unit `u` is `[u * fan_in..(u + 1) * fan_in]`.
fn transpose(layer: &Layer) -> Vec<f32> {
    (0..layer.fan_out)
        .flat_map(|u| (0..layer.fan_in).map(move |i| layer.weights[i * layer.fan_out + u]))
        .collect()
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

    /// As [`net`], with a head of 8 hidden units in every bucket.
    pub(in crate::eval) fn heads_net(features: FeatureSet) -> Net {
        Net::new(
            features,
            vec![
                layer(features.size(), 16, 1),
                layer(32, 64, 2),
                layer(8, 8, 3),
            ],
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
    fn a_network_has_an_accumulator_an_output_layer_and_at_most_a_head() {
        let head = |hidden: usize, width: usize| {
            vec![layer(768, 8, 1), layer(16, hidden, 2), layer(width, 8, 3)]
        };
        assert!(Net::new(PIECES, head(64, 8)).is_ok());
        assert!(Net::new(PIECES, head(60, 8)).is_err());
        let mut four = head(64, 8);
        four.push(layer(8, 1, 4));
        assert!(Net::new(PIECES, four).is_err());
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

    #[test]
    fn each_bucket_reads_its_block_of_hidden_units_and_its_column_of_the_head() {
        let (hidden, last) = (layer(16, 32, 2), layer(4, 8, 3));
        let (w1, b1) = (hidden.weights.clone(), hidden.bias.clone());
        let (w2, b2) = (last.weights.clone(), last.bias.clone());
        let net = Net::new(PIECES, vec![layer(768, 8, 1), hidden, last]).unwrap();
        let mover: Vec<f32> = (0..8).map(|i| i as f32 / 7.0).collect();
        let rival: Vec<f32> = (0..8).map(|i| 1.5 - i as f32 / 4.0).collect();
        let x: Vec<f32> = mover
            .iter()
            .chain(&rival)
            .map(|x| x.clamp(0.0, 1.0))
            .collect();
        for pieces in [3, 7, 12, 18, 21, 26, 30, 32] {
            let b = net.bucket(pieces);
            let expected: f32 = b2[b]
                + (0..4)
                    .map(|k| {
                        let unit = 4 * b + k;
                        let sum: f32 = (0..16).map(|i| x[i] * w1[i * 32 + unit]).sum();
                        (b1[unit] + sum).clamp(0.0, 1.0) * w2[k * 8 + b]
                    })
                    .sum::<f32>();
            assert!((net.output(&mover, &rival, pieces) - expected).abs() < 1e-5);
        }
    }
}
