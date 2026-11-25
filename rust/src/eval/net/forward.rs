//! The end of the network: from the sums of both accumulators to the logit. `NetState` keeps the
//! sums move by move and calls [`Net::output`].

use super::Net;

impl Net {
    /// The logit from the accumulators of the mover (with its dense inputs already added) and of
    /// its rival, for the material bucket of `pieces` (the pieces on the board, kings included):
    /// the bucket's output unit, or with a head its hidden units, each clipped to `[0, 1]`,
    /// weighted by the bucket's column of the last layer.
    pub(in crate::eval) fn output(&self, mover: &[f32], rival: &[f32], pieces: usize) -> f32 {
        let bucket = self.bucket(pieces);
        let Some(head) = &self.head else {
            return self.unit(mover, rival, bucket);
        };
        let weights = &head.weights[head.width * bucket..head.width * (bucket + 1)];
        let first = head.width * bucket;
        head.bias[bucket]
            + (first..first + head.width)
                .zip(weights)
                .map(|(unit, w)| self.unit(mover, rival, unit).clamp(0.0, 1.0) * w)
                .sum::<f32>()
    }

    /// Unit `unit` of the second layer: both accumulators, each clipped to `[0, 1]`, weighted by
    /// its half of the unit's weights.
    fn unit(&self, mover: &[f32], rival: &[f32], unit: usize) -> f32 {
        let width = self.width();
        let weights = &self.columns[2 * width * unit..2 * width * (unit + 1)];
        let mut lanes = [0.0f32; 8];
        add_clipped_lanes(&mut lanes, mover, &weights[..width]);
        add_clipped_lanes(&mut lanes, rival, &weights[width..]);
        self.output_bias[unit] + lanes.iter().sum::<f32>()
    }
}

/// Adds to `lanes` the products of `acc`, each value clipped to `[0, 1]` as the accumulator is
/// read, with `weights`, eight lanes at a time so that the compiler can use vector instructions (a
/// single running sum would have to add the terms one after another). The length of both is a
/// multiple of 8 (`Net::new` checks the width).
#[inline(always)]
fn add_clipped_lanes(lanes: &mut [f32; 8], acc: &[f32], weights: &[f32]) {
    for (x, w) in acc
        .as_chunks::<8>()
        .0
        .iter()
        .zip(weights.as_chunks::<8>().0)
    {
        for lane in 0..8 {
            lanes[lane] += x[lane].clamp(0.0, 1.0) * w[lane];
        }
    }
}
