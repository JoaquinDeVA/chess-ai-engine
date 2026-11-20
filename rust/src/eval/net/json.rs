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

    /// The file of a network of two units in the accumulator, with `rows` inputs and an output
    /// layer of `last_in` inputs, the `extra` fields before the layers.
    fn net_json(features: &str, extra: &str, rows: usize, last_in: usize) -> String {
        let first = format!(
            r#"{{"fan_in":{rows},"fan_out":2,"weights":{:?},"bias":[0.5,0.5]}}"#,
            vec![0.0; rows * 2]
        );
        let last = format!(
            r#"{{"fan_in":{last_in},"fan_out":1,"weights":{:?},"bias":[0.0]}}"#,
            vec![1.0; last_in]
        );
        format!(r#"{{"features":"{features}",{extra}"layers":[{first},{last}]}}"#)
    }

    fn loads(features: &str, extra: &str, rows: usize, last_in: usize) -> bool {
        Net::from_json(&net_json(features, extra, rows, last_in)).is_ok()
    }

    #[test]
    fn the_perspective_field_is_absent_or_both() {
        assert!(loads("pst_ctx", "", 768, 4));
        assert!(loads("pst_ctx", r#""perspective":"both","#, 768, 4));
        assert!(!loads(
            "pst_ctx",
            r#""perspective":"side_to_move","#,
            768,
            4
        ));
        assert!(!loads("pst_ctx", r#""perspective":"white","#, 768, 4));
    }

    #[test]
    fn the_second_layer_must_be_twice_as_wide_as_the_accumulator() {
        assert!(!loads("pst_ctx", "", 768, 2));
        // the dense inputs make the first layer wider than the 768 pieces of this file
        assert!(!loads("pst_threats", "", 768, 4));
        assert!(loads("pst_threats", "", 773, 4));
    }

    #[test]
    fn a_network_with_the_wrong_input_size_is_rejected() {
        assert!(!loads("pst_ctx", "", 10, 4));
    }

    #[test]
    fn a_ctx_network_reads_its_families_from_the_file() {
        let rows = 768 * 4 + 30;
        assert!(loads(
            "pst_ctx",
            r#""families":["material","king_buckets"],"#,
            rows,
            4
        ));
        assert!(!loads("pst_ctx", "", rows, 4));
        assert!(!loads("pst_ctx", r#""families":["nothing"],"#, rows, 4));
    }

    #[test]
    fn the_old_name_of_the_pieces_alone_still_loads() {
        assert!(loads("pst_slotted", "", 768, 4));
    }
}
