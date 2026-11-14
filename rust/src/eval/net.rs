//! The sparse network: slotted inputs, an accumulator of clipped sums and dense layers.

use shakmaty::{Chess, Color, Position};

use super::features::{FeatureSet, OPPONENT_PIECE_BLOCK, PIECE_INPUTS};
use super::{for_player, score, sigmoid};

/// One dense layer as the lab writes it: `weights` is row-major `(fan_in, fan_out)`.
pub struct Layer {
    pub fan_in: usize,
    pub fan_out: usize,
    pub weights: Vec<f64>,
    pub bias: Vec<f64>,
}

/// A layer as the engine keeps it: the same layout in `f32`.
pub(super) struct Dense {
    pub(super) fan_out: usize,
    pub(super) weights: Vec<f32>,
    pub(super) bias: Vec<f32>,
}

impl Dense {
    /// The row of input `index`: its `fan_out` weights.
    pub(super) fn row(&self, index: usize) -> &[f32] {
        &self.weights[index * self.fan_out..(index + 1) * self.fan_out]
    }
}

/// A feed-forward network over sparse inputs; the first layer is the accumulator.
pub struct Net {
    pub(super) features: FeatureSet,
    pub(super) layers: Vec<Dense>,
    /// Widest layer before the output: the size of the buffers of an evaluation.
    pub(super) max_width: usize,
    /// Whether the second layer reads the accumulators of both sides (the mover's, then the
    /// rival's, 2 * width values) instead of the mover's alone.
    pub(super) perspective: bool,
}

impl Net {
    /// A network of the side to move alone, checking the shapes of its layers.
    pub fn new(features: FeatureSet, layers: Vec<Layer>) -> Result<Net, String> {
        Net::build(features, layers, false)
    }

    /// A network whose second layer reads both sides' accumulators, which share the first layer's
    /// weights (`layers[1].fan_in` is twice the accumulator width). Only the slotted inputs, where
    /// the rival's view of a piece is the other block of the same square, can be seen from both
    /// sides.
    pub fn with_perspective(features: FeatureSet, layers: Vec<Layer>) -> Result<Net, String> {
        Net::build(features, layers, true)
    }

    fn build(features: FeatureSet, layers: Vec<Layer>, perspective: bool) -> Result<Net, String> {
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
        if perspective && features != FeatureSet::Slotted {
            return Err("both perspectives need the slotted inputs".into());
        }
        for (position, pair) in layers.windows(2).enumerate() {
            let wanted = if perspective && position == 0 {
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
        let mut max_width = layers.iter().map(|l| l.fan_out).max().unwrap_or(1);
        if perspective {
            max_width = max_width.max(2 * layers[0].fan_out);
        }
        let single = |values: &[f64]| values.iter().map(|&v| v as f32).collect();
        let layers = layers
            .iter()
            .map(|layer| Dense {
                fan_out: layer.fan_out,
                weights: single(&layer.weights),
                bias: single(&layer.bias),
            })
            .collect();
        Ok(Net {
            features,
            layers,
            max_width,
            perspective,
        })
    }
    /// A network from the JSON of `chess_lab.models.export_uci`: its `features` and its `layers`,
    /// each with `fan_in`, `fan_out`, row-major `weights` and `bias`. Files written before the
    /// networks were the only model carry `"perspective": "side_to_move"`; any other perspective
    /// is refused; `"both"` makes the second layer read both sides' accumulators.
    pub fn from_json(text: &str) -> Result<Net, String> {
        let value: serde_json::Value = serde_json::from_str(text).map_err(|e| e.to_string())?;
        let features = value["features"]
            .as_str()
            .and_then(FeatureSet::from_name)
            .ok_or("model: unknown features")?;
        let perspective = match value.get("perspective").map(|p| p.as_str()) {
            None | Some(Some("side_to_move")) => false,
            Some(Some("both")) => true,
            _ => return Err("model: the perspective must be side_to_move or both".into()),
        };
        let numbers = |value: &serde_json::Value| -> Result<Vec<f64>, String> {
            value
                .as_array()
                .ok_or("model: no weights")?
                .iter()
                .map(|w| {
                    w.as_f64()
                        .ok_or("model: weight is not a number".to_string())
                })
                .collect()
        };
        let layers = value["layers"]
            .as_array()
            .ok_or("model: layers is not a list")?
            .iter()
            .map(|layer| {
                let size = |key: &str| {
                    layer[key]
                        .as_u64()
                        .map(|n| n as usize)
                        .ok_or(format!("model: layer has no {key}"))
                };
                Ok(Layer {
                    fan_in: size("fan_in")?,
                    fan_out: size("fan_out")?,
                    weights: numbers(&layer["weights"])?,
                    bias: numbers(&layer["bias"])?,
                })
            })
            .collect::<Result<Vec<Layer>, String>>()?;
        Net::build(features, layers, perspective)
    }

    /// The feature set of the inputs.
    pub fn features(&self) -> FeatureSet {
        self.features
    }

    /// Network output before the logistic function, for the side to move, from the board (the
    /// search follows it move by move with [`NetState`](super::NetState)).
    pub fn logit(&self, pos: &Chess) -> f32 {
        let (mut indices, mut values) = (Vec::with_capacity(40), Vec::with_capacity(40));
        self.features.active(pos, &mut indices, &mut values);
        let first = &self.layers[0];
        let width = first.fan_out;
        let mut hidden = first.bias.clone();
        for (&index, &value) in indices.iter().zip(&values) {
            for (h, w) in hidden.iter_mut().zip(first.row(index)) {
                *h += w * value as f32;
            }
        }
        if self.perspective {
            // the rival's view: every piece in the other block of its square
            hidden.extend_from_slice(&first.bias);
            for (&index, &value) in indices.iter().zip(&values) {
                for (h, w) in hidden[width..]
                    .iter_mut()
                    .zip(first.row((index + OPPONENT_PIECE_BLOCK) % PIECE_INPUTS))
                {
                    *h += w * value as f32;
                }
            }
        }
        hidden.resize(self.max_width, 0.0);
        self.finish(&mut hidden, &mut vec![0.0; self.max_width])
    }

    /// The rest of the network from the sums of the accumulator, the first values of `hidden`: the
    /// clip to `[0, 1]`, the hidden layers with ReLU and the output. `hidden` and `spare` hold
    /// `max_width` values each, so that an evaluation allocates nothing.
    pub(super) fn finish(&self, hidden: &mut [f32], spare: &mut [f32]) -> f32 {
        let mut width = self.layers[0].fan_out;
        if self.perspective {
            width *= 2;
        }
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

    /// Score of `player` in `pos`, from the board (see [`Net::logit`]).
    pub fn score(&self, pos: &Chess, player: Color) -> i32 {
        score(self.logit(pos), pos.turn(), player)
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

/// Dot product in eight lanes, so that the compiler can use vector instructions (a single running
/// sum would have to add the terms one after another).
fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut lanes = [0.0f32; 8];
    let ((chunks_a, rest_a), (chunks_b, rest_b)) = (a.as_chunks::<8>(), b.as_chunks::<8>());
    let tail: f32 = rest_a.iter().zip(rest_b).map(|(x, y)| x * y).sum();
    for (x, y) in chunks_a.iter().zip(chunks_b) {
        for lane in 0..8 {
            lanes[lane] += x[lane] * y[lane];
        }
    }
    lanes.iter().sum::<f32>() + tail
}

#[cfg(test)]
mod tests {
    use super::*;

    fn net_json(fan_in: usize, extra: &str) -> String {
        let first = format!(
            r#"{{"fan_in":{fan_in},"fan_out":2,"weights":{:?},"bias":[0.5,0.5]}}"#,
            vec![0.0; fan_in * 2]
        );
        let last = r#"{"fan_in":2,"fan_out":1,"weights":[1.0,1.0],"bias":[0.0]}"#;
        format!(r#"{{"features":"pst_slotted",{extra}"layers":[{first},{last}]}}"#)
    }

    #[test]
    fn a_network_file_is_loaded_with_or_without_the_old_perspective_field() {
        assert!(Net::from_json(&net_json(768, "")).is_ok());
        let old = r#""perspective":"side_to_move","#;
        assert!(Net::from_json(&net_json(768, old)).is_ok());
        let white = r#""perspective":"white","#;
        assert!(Net::from_json(&net_json(768, white)).is_err());
    }

    #[test]
    fn both_perspectives_need_a_second_layer_twice_as_wide() {
        let both = r#""perspective":"both","#;
        assert!(Net::from_json(&net_json(768, both)).is_err());
        let first = format!(
            r#"{{"fan_in":768,"fan_out":2,"weights":{:?},"bias":[0.0,0.0]}}"#,
            vec![0.0; 768 * 2]
        );
        let last = r#"{"fan_in":4,"fan_out":1,"weights":[1.0,1.0,1.0,1.0],"bias":[0.0]}"#;
        let text = format!(r#"{{"features":"pst_slotted",{both}"layers":[{first},{last}]}}"#);
        assert!(Net::from_json(&text).is_ok());
        assert!(Net::from_json(&text.replace("pst_slotted", "pst_hybrid")).is_err());
    }

    #[test]
    fn a_network_with_the_wrong_input_size_is_rejected() {
        assert!(Net::from_json(&net_json(10, "")).is_err());
    }

    #[test]
    fn the_dot_product_adds_every_term() {
        let a: Vec<f32> = (0..19).map(|i| i as f32).collect();
        let b = vec![2.0; 19];
        assert_eq!(dot(&a, &b), 2.0 * (0..19).sum::<i32>() as f32);
    }
}
