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

    /// The file of a network of eight units in the accumulator, with `rows` inputs and an output
    /// layer of `last_in` inputs, the `extra` fields before the layers.
    fn net_json(features: &str, extra: &str, rows: usize, last_in: usize) -> String {
        let first = format!(
            r#"{{"fan_in":{rows},"fan_out":8,"weights":{:?},"bias":{:?}}}"#,
            vec![0.0; rows * 8],
            [0.5; 8]
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
        assert!(loads("pst_ctx", "", 768, 16));
        assert!(loads("pst_ctx", r#""perspective":"both","#, 768, 16));
        assert!(!loads(
            "pst_ctx",
            r#""perspective":"side_to_move","#,
            768,
            16
        ));
        assert!(!loads("pst_ctx", r#""perspective":"white","#, 768, 16));
    }

    #[test]
    fn the_second_layer_must_be_twice_as_wide_as_the_accumulator() {
        assert!(!loads("pst_ctx", "", 768, 8));
        // the dense inputs make the first layer wider than the 768 pieces of this file
        assert!(!loads("pst_threats", "", 768, 16));
        assert!(loads("pst_threats", "", 773, 16));
    }

    #[test]
    fn the_output_layer_may_have_one_unit_per_material_bucket() {
        let file = |units: usize| {
            net_json("pst_ctx", "", 768, 16).replace(
                &format!(r#""fan_out":1,"weights":{:?},"bias":[0.0]"#, vec![1.0; 16]),
                &format!(
                    r#""fan_out":{units},"weights":{:?},"bias":{:?}"#,
                    vec![1.0; 16 * units],
                    vec![0.0; units]
                ),
            )
        };
        assert!(Net::from_json(&file(8)).is_ok());
        assert!(Net::from_json(&file(33)).is_err());
    }

    #[test]
    fn a_third_layer_is_a_head_per_bucket() {
        let head = format!(
            r#",{{"fan_in":2,"fan_out":8,"weights":{:?},"bias":{:?}}}]"#,
            vec![1.0; 16],
            vec![0.0; 8]
        );
        let file = |hidden: usize| {
            net_json("pst_ctx", "", 768, 16)
                .replace(
                    &format!(r#""fan_out":1,"weights":{:?},"bias":[0.0]"#, vec![1.0; 16]),
                    &format!(
                        r#""fan_out":{hidden},"weights":{:?},"bias":{:?}"#,
                        vec![1.0; 16 * hidden],
                        vec![0.0; hidden]
                    ),
                )
                .replacen("}]}", &format!("}}{head}}}"), 1)
        };
        assert!(Net::from_json(&file(16)).is_ok());
        assert!(Net::from_json(&file(15)).is_err());
    }

    #[test]
    fn a_network_with_the_wrong_input_size_is_rejected() {
        assert!(!loads("pst_ctx", "", 10, 16));
    }

    #[test]
    fn a_ctx_network_reads_its_families_from_the_file() {
        let rows = 768 * 4 + 30;
        assert!(loads(
            "pst_ctx",
            r#""families":["material","king_buckets"],"#,
            rows,
            16
        ));
        assert!(!loads("pst_ctx", "", rows, 16));
        assert!(!loads("pst_ctx", r#""families":["nothing"],"#, rows, 16));
    }

    #[test]
    fn the_old_name_of_the_pieces_alone_still_loads() {
        assert!(loads("pst_slotted", "", 768, 16));
    }
}
