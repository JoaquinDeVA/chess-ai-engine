//! The output of the network from the accumulators of the position: the dense inputs of the
//! threats are added to the mover's, and the rest of the network is `Net::finish`.

use shakmaty::{Chess, Position};

use super::{slot, NetState};

/// Adds to `lanes` the dot product of `acc` clipped to `[0, 1]` with `weights`, eight lanes at a
/// time (the length is a multiple of 8).
#[inline(always)]
fn clipped_dot(acc: &[f32], weights: &[f32], lanes: &mut [f32; 8]) {
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

impl NetState<'_> {
    /// Network output before the logistic function, for the side to move in `pos`, the position
    /// the state is following.
    pub fn logit(&mut self, pos: &Chess) -> f32 {
        let net = self.net;
        let width = self.width;
        let start = (2 * self.top + slot(pos.turn())) * width;
        if net.layers.len() == 2 && width.is_multiple_of(8) {
            // one output layer straight over the accumulators: no intermediate buffer, except the
            // mover's own with the dense inputs added (they have no opponent view)
            let out = &net.layers[1];
            let mut lanes = [0.0f32; 8];
            if net.features.has_dense() {
                self.dense_indices.clear();
                self.dense_values.clear();
                net.features
                    .dense_inputs(pos, &mut self.dense_indices, &mut self.dense_values);
                let own = &mut self.hidden[..width];
                own.copy_from_slice(&self.stack[start..start + width]);
                for (&index, &value) in self.dense_indices.iter().zip(&self.dense_values) {
                    if value == 0.0 {
                        continue;
                    }
                    for (h, w) in own.iter_mut().zip(net.layers[0].row(index)) {
                        *h += w * value as f32;
                    }
                }
                clipped_dot(own, &out.weights[..width], &mut lanes);
            } else {
                clipped_dot(
                    &self.stack[start..start + width],
                    &out.weights[..width],
                    &mut lanes,
                );
            }
            let opponent = (2 * self.top + slot(!pos.turn())) * width;
            clipped_dot(
                &self.stack[opponent..opponent + width],
                &out.weights[width..2 * width],
                &mut lanes,
            );
            return out.bias[0] + lanes.iter().sum::<f32>();
        }
        self.hidden[..width].copy_from_slice(&self.stack[start..start + width]);
        // the opponent's accumulator follows the mover's: both views, one shared first layer
        let opponent = (2 * self.top + slot(!pos.turn())) * width;
        self.hidden[width..2 * width].copy_from_slice(&self.stack[opponent..opponent + width]);
        if net.features.has_dense() {
            self.dense_indices.clear();
            self.dense_values.clear();
            net.features
                .dense_inputs(pos, &mut self.dense_indices, &mut self.dense_values);
            for (&index, &value) in self.dense_indices.iter().zip(&self.dense_values) {
                for (h, w) in self.hidden.iter_mut().zip(net.layers[0].row(index)) {
                    *h += w * value as f32;
                }
            }
        }
        net.finish(&mut self.hidden, &mut self.spare)
    }
}
