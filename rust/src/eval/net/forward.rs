//! The end of the network: from the sums of the accumulator to the output. `NetState` keeps the
//! sums move by move and calls [`Net::finish`] to get the logit.

use super::Net;

impl Net {
    /// The rest of the network from the sums of the accumulator, the first values of `hidden`: the
    /// clip to `[0, 1]`, the hidden layers with ReLU and the output. `hidden` and `spare` hold
    /// `max_width` values each, so that an evaluation allocates nothing.
    pub(in crate::eval) fn finish(&self, hidden: &mut [f32], spare: &mut [f32]) -> f32 {
        let mut width = 2 * self.layers[0].fan_out;
        for h in &mut hidden[..width] {
            *h = h.clamp(0.0, 1.0);
        }
        let last = self.layers.len() - 1;
        for layer in &self.layers[1..last] {
            let out = &mut spare[..layer.fan_out];
            out.copy_from_slice(&layer.bias);
            for (index, &x) in hidden[..width].iter().enumerate() {
                for (o, w) in out.iter_mut().zip(layer.row(index)) {
                    *o += x * w;
                }
            }
            for (h, o) in hidden.iter_mut().zip(out.iter()) {
                *h = o.max(0.0);
            }
            width = layer.fan_out;
        }
        let output = &self.layers[last];
        output.bias[0] + dot(&hidden[..width], &output.weights)
    }
}

/// Adds to `lanes` the products of `a`, each value read through `read`, with `b`, eight lanes at a
/// time so that the compiler can use vector instructions (a single running sum would have to add
/// the terms one after another). Only whole groups of eight are added.
#[inline(always)]
fn add_lanes(lanes: &mut [f32; 8], a: &[f32], b: &[f32], read: impl Fn(f32) -> f32) {
    for (x, w) in a.as_chunks::<8>().0.iter().zip(b.as_chunks::<8>().0) {
        for lane in 0..8 {
            lanes[lane] += read(x[lane]) * w[lane];
        }
    }
}

/// [`add_lanes`] of the values clipped to `[0, 1]`, as the accumulator is read; the length of
/// both is a multiple of 8.
#[inline(always)]
pub(in crate::eval) fn add_clipped_lanes(lanes: &mut [f32; 8], acc: &[f32], weights: &[f32]) {
    add_lanes(lanes, acc, weights, |x| x.clamp(0.0, 1.0));
}

/// Dot product: the lanes of [`add_lanes`] and the terms left over.
fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut lanes = [0.0f32; 8];
    add_lanes(&mut lanes, a, b, |x| x);
    let tail: f32 = a
        .as_chunks::<8>()
        .1
        .iter()
        .zip(b.as_chunks::<8>().1)
        .map(|(x, y)| x * y)
        .sum();
    lanes.iter().sum::<f32>() + tail
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_dot_product_adds_every_term() {
        let a: Vec<f32> = (0..19).map(|i| i as f32).collect();
        let b = vec![2.0; 19];
        assert_eq!(dot(&a, &b), 2.0 * (0..19).sum::<i32>() as f32);
    }
}
