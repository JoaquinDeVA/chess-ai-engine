//! The sparse network: slotted inputs, an accumulator of clipped sums and dense layers.

use shakmaty::{Chess, Color, Position};

use super::features::FeatureSet;
use super::{for_player, sigmoid};

/// One dense layer: `weights` is row-major `(fan_in, fan_out)`.
pub struct Layer {
    pub fan_in: usize,
    pub fan_out: usize,
    pub weights: Vec<f64>,
    pub bias: Vec<f64>,
}

/// A feed-forward network over sparse inputs; the first layer is the accumulator.
pub struct Net {
    pub(super) features: FeatureSet,
    pub(super) layers: Vec<Layer>,
}

impl Net {
    /// A network, checking the shapes of its layers.
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
        for pair in layers.windows(2) {
            if pair[0].fan_out != pair[1].fan_in {
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
        Ok(Net { features, layers })
    }

    /// A network from the JSON of `chess_lab.models.export_uci`: its `features` and its `layers`,
    /// each with `fan_in`, `fan_out`, row-major `weights` and `bias`. Files written before the
    /// networks were the only model carry `"perspective": "side_to_move"`; any other perspective
    /// is refused.
    pub fn from_json(text: &str) -> Result<Net, String> {
        let value: serde_json::Value = serde_json::from_str(text).map_err(|e| e.to_string())?;
        let features = value["features"]
            .as_str()
            .and_then(FeatureSet::from_name)
            .ok_or("model: unknown features")?;
        if value
            .get("perspective")
            .is_some_and(|p| *p != "side_to_move")
        {
            return Err("model: only the side-to-move perspective is supported".into());
        }
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
        Net::new(features, layers)
    }

    /// The feature set of the inputs.
    pub fn features(&self) -> FeatureSet {
        self.features
    }

    /// Network output before the logistic function, for the side to move.
    pub fn logit(&self, pos: &Chess) -> f64 {
        let (mut indices, mut values) = (Vec::with_capacity(40), Vec::with_capacity(40));
        self.features.active(pos, &mut indices, &mut values);
        let first = &self.layers[0];
        let mut hidden = first.bias.clone();
        for (index, value) in indices.iter().zip(&values) {
            let row = &first.weights[index * first.fan_out..(index + 1) * first.fan_out];
            for (h, w) in hidden.iter_mut().zip(row) {
                *h += w * value;
            }
        }
        for h in &mut hidden {
            *h = h.clamp(0.0, 1.0);
        }
        self.finish(hidden)
    }

    /// The layers after the accumulator, applied to its clipped output.
    pub(super) fn finish(&self, mut hidden: Vec<f64>) -> f64 {
        let last = self.layers.len() - 1;
        for layer in &self.layers[1..last] {
            hidden = dense(layer, &hidden, true);
        }
        dense(&self.layers[last], &hidden, false)[0]
    }

    /// Expected score of White in `[0, 1]`.
    pub fn expected_score(&self, pos: &Chess) -> f64 {
        let score = sigmoid(self.logit(pos));
        if pos.turn() == Color::Black {
            1.0 - score
        } else {
            score
        }
    }

    /// Expected score of `player` in `pos`, in `[0.01, 0.99]`, from the board (the search uses
    /// [`NetState`], which gives the same value move by move).
    pub fn evaluate(&self, pos: &Chess, player: Color) -> f64 {
        for_player(self.expected_score(pos), player)
    }
}

fn dense(layer: &Layer, input: &[f64], relu: bool) -> Vec<f64> {
    let mut out = layer.bias.clone();
    for (i, x) in input.iter().enumerate() {
        let row = &layer.weights[i * layer.fan_out..(i + 1) * layer.fan_out];
        for (o, w) in out.iter_mut().zip(row) {
            *o += x * w;
        }
    }
    if relu {
        for o in &mut out {
            *o = o.max(0.0);
        }
    }
    out
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
    fn a_network_with_the_wrong_input_size_is_rejected() {
        assert!(Net::from_json(&net_json(10, "")).is_err());
    }
}
