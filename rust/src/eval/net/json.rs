//! Reading a network from the JSON of `chess_lab.models.export_uci`: its `features`, its
//! `families` and its `layers`, with the weights rounded to `f32` as they are read.

use super::{Layer, Net};
use crate::eval::features::{Families, FeatureSet};

impl Net {
    /// A network from the JSON of `chess_lab.models.export_uci`: its `features` and its `layers`,
    /// each with `fan_in`, `fan_out`, row-major `weights` and `bias`. The `"perspective"` field may
    /// be absent or `"both"`: the networks of the side to move alone are no longer read.
    pub fn from_json(text: &str) -> Result<Net, String> {
        let value: serde_json::Value = serde_json::from_str(text).map_err(|e| e.to_string())?;
        let features = value["features"]
            .as_str()
            .and_then(FeatureSet::from_name)
            .ok_or("model: unknown features")?;
        let families: Vec<&str> = match value.get("families") {
            None => Vec::new(),
            Some(list) => list
                .as_array()
                .ok_or("model: families is not a list")?
                .iter()
                .map(|name| name.as_str().ok_or("model: a family is not a name"))
                .collect::<Result<_, _>>()?,
        };
        let features = features.with_families(Families::from_names(&families)?)?;
        if !matches!(
            value.get("perspective").map(|p| p.as_str()),
            None | Some(Some("both"))
        ) {
            return Err("model: the perspective must be both".into());
        }
        let numbers = |value: &serde_json::Value| -> Result<Vec<f32>, String> {
            value
                .as_array()
                .ok_or("model: no weights")?
                .iter()
                .map(|w| {
                    w.as_f64()
                        .map(|w| w as f32)
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
}

#[cfg(test)]
mod tests {
    use super::*;

    fn net_json(fan_in: usize, extra: &str) -> String {
        let first = format!(
            r#"{{"fan_in":{fan_in},"fan_out":2,"weights":{:?},"bias":[0.5,0.5]}}"#,
            vec![0.0; fan_in * 2]
        );
        let last = r#"{"fan_in":4,"fan_out":1,"weights":[1.0,1.0,1.0,1.0],"bias":[0.0]}"#;
        format!(r#"{{"features":"pst_slotted",{extra}"layers":[{first},{last}]}}"#)
    }

    #[test]
    fn the_perspective_field_is_absent_or_both() {
        assert!(Net::from_json(&net_json(768, "")).is_ok());
        assert!(Net::from_json(&net_json(768, r#""perspective":"both","#)).is_ok());
        assert!(Net::from_json(&net_json(768, r#""perspective":"side_to_move","#)).is_err());
        assert!(Net::from_json(&net_json(768, r#""perspective":"white","#)).is_err());
    }

    #[test]
    fn the_second_layer_must_be_twice_as_wide_as_the_accumulator() {
        let first = format!(
            r#"{{"fan_in":768,"fan_out":2,"weights":{:?},"bias":[0.0,0.0]}}"#,
            vec![0.0; 768 * 2]
        );
        let last = r#"{"fan_in":2,"fan_out":1,"weights":[1.0,1.0],"bias":[0.0]}"#;
        let text = format!(r#"{{"features":"pst_slotted","layers":[{first},{last}]}}"#);
        assert!(Net::from_json(&text).is_err());
        // the dense inputs make the first layer wider than the 768 pieces of this file
        assert!(Net::from_json(&net_json(768, "").replace("pst_slotted", "pst_threats")).is_err());
    }

    #[test]
    fn a_network_with_the_wrong_input_size_is_rejected() {
        assert!(Net::from_json(&net_json(10, "")).is_err());
    }

    #[test]
    fn a_ctx_network_reads_its_families_from_the_file() {
        let rows = 768 * 4 + 30;
        let first = format!(
            r#"{{"fan_in":{rows},"fan_out":2,"weights":{:?},"bias":[0.5,0.5]}}"#,
            vec![0.0; rows * 2]
        );
        let last = r#"{"fan_in":4,"fan_out":1,"weights":[1.0,1.0,1.0,1.0],"bias":[0.0]}"#;
        let text = |families: &str| {
            format!(r#"{{"features":"pst_ctx",{families}"layers":[{first},{last}]}}"#)
        };
        assert!(Net::from_json(&text(r#""families":["material","king_buckets"],"#)).is_ok());
        assert!(Net::from_json(&text("")).is_err());
        assert!(Net::from_json(&text(r#""families":["nothing"],"#)).is_err());
    }
}
