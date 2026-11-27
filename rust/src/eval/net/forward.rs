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
        let mut sum = 0.0;
        let (blocks, rest) = weights.as_chunks::<BLOCK>();
        for (i, block) in blocks.iter().enumerate() {
            let hidden = self.units::<BLOCK>(mover, rival, first + BLOCK * i);
            for (h, w) in hidden.iter().zip(block) {
                sum += h.clamp(0.0, 1.0) * w;
            }
        }
        for (i, w) in rest.iter().enumerate() {
            let unit = first + BLOCK * blocks.len() + i;
            sum += self.unit(mover, rival, unit).clamp(0.0, 1.0) * w;
        }
        head.bias[bucket] + sum
    }

    /// Unit `unit` of the second layer.
    fn unit(&self, mover: &[f32], rival: &[f32], unit: usize) -> f32 {
        let [value] = self.units::<1>(mover, rival, unit);
        value
    }

    /// The `N` units of the second layer from `first`: both accumulators, each clipped to
    /// `[0, 1]`, weighted by the unit's two halves of weights. The units are summed side by side,
    /// so that each value is read once and their additions do not wait for one another.
    #[inline(always)]
    fn units<const N: usize>(&self, mover: &[f32], rival: &[f32], first: usize) -> [f32; N] {
        let width = self.width();
        let weights: [&[f32]; N] = std::array::from_fn(|u| {
            &self.columns[2 * width * (first + u)..2 * width * (first + u + 1)]
        });
        let mut lanes = [[0.0f32; 8]; N];
        add_clipped_lanes(&mut lanes, mover, weights.map(|w| &w[..width]));
        add_clipped_lanes(&mut lanes, rival, weights.map(|w| &w[width..]));
        std::array::from_fn(|u| self.output_bias[first + u] + lanes[u].iter().sum::<f32>())
    }
}

/// Hidden units of a head computed side by side: more would not fit the vector registers.
const BLOCK: usize = 8;

/// Adds to `lanes[u]` the products of `acc`, each value clipped to `[0, 1]` as the accumulator is
/// read, with `weights[u]`, eight lanes at a time so that the compiler can use vector instructions
/// (a single running sum would have to add the terms one after another). The length of all is a
/// multiple of 8 (`Net::new` checks the width).
#[inline(always)]
fn add_clipped_lanes<const N: usize>(lanes: &mut [[f32; 8]; N], acc: &[f32], weights: [&[f32]; N]) {
    let weights = weights.map(|w| &w.as_chunks::<8>().0[..acc.len() / 8]);
    for (i, x) in acc.as_chunks::<8>().0.iter().enumerate() {
        let x = x.map(|v| v.clamp(0.0, 1.0));
        for u in 0..N {
            for lane in 0..8 {
                lanes[u][lane] += x[lane] * weights[u][i][lane];
            }
        }
    }
}
