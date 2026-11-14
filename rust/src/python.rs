//! Python bindings for the Rust core, exposed as `chess_engine._core`.
//!
//! This module exposes the evaluators and the configuration metadata needed by the laboratory.
//! The search, opening book and endgame tablebases run only in the UCI binary
//! (`src/bin/uci/`); Python reads the corresponding defaults from this module.

use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use shakmaty::Color;

use crate::board::position_from_fen;
use crate::eval::FeatureSet;
use crate::eval::{Layer, Net};

fn value_error<E: std::fmt::Display>(error: E) -> PyErr {
    PyValueError::new_err(error.to_string())
}

/// Resolve a public feature-set name to the Rust implementation used by the evaluator.
fn parse_features(name: &str) -> PyResult<FeatureSet> {
    FeatureSet::from_name(name).ok_or_else(|| value_error(format!("unknown feature set {name:?}")))
}

/// A Python-owned evaluator backed by a Rust neural network.
///
#[pyclass(name = "Evaluator", frozen)]
struct PyEvaluator {
    /// The network.
    net: Net,
}

#[pymethods]
impl PyEvaluator {
    /// Build a neural network over a named sparse feature set.
    ///
    /// Each layer is a Python `(weights, bias)` pair. Weights are copied into Rust-owned storage;
    /// the first layer consumes the feature accumulator produced by the selected feature set.
    ///
    /// With `perspective` the second layer reads the accumulators of both sides, which share the
    /// first layer (see `Net::with_perspective`).
    #[staticmethod]
    #[pyo3(signature = (features, layers, perspective=false))]
    fn network(
        features: &str,
        layers: Vec<(PyReadonlyArray2<f64>, PyReadonlyArray1<f64>)>,
        perspective: bool,
    ) -> PyResult<PyEvaluator> {
        let set = parse_features(features)?;
        let layers = layers
            .iter()
            .map(|(weights, bias)| {
                let (fan_in, fan_out) = weights.as_array().dim();
                Layer {
                    fan_in,
                    fan_out,
                    weights: weights.as_array().iter().copied().collect(),
                    bias: bias.as_array().to_vec(),
                }
            })
            .collect();
        let net = if perspective {
            Net::with_perspective(set, layers)
        } else {
            Net::new(set, layers)
        }
        .map_err(value_error)?;
        Ok(PyEvaluator { net })
    }

    /// Evaluate a FEN position from White's or Black's perspective.
    ///
    /// `white=True` evaluates for White; `white=False` evaluates for Black. The returned score is
    /// in the evaluator's `[0.01, 0.99]` range.
    fn evaluate(&self, fen: &str, white: bool) -> PyResult<f64> {
        let pos = position_from_fen(fen).map_err(value_error)?;
        Ok(self
            .net
            .evaluate(&pos, if white { Color::White } else { Color::Black }))
    }

    /// Return the active sparse feature indices and values for a FEN position.
    fn features(&self, fen: &str) -> PyResult<(Vec<usize>, Vec<f64>)> {
        let set = self.net.features();
        let pos = position_from_fen(fen).map_err(value_error)?;
        let (mut indices, mut values) = (Vec::new(), Vec::new());
        set.active(&pos, &mut indices, &mut values);
        Ok((indices, values))
    }
}

/// Register the Rust evaluator and search metadata in `chess_engine._core`.
#[pymodule]
mod _core {
    use pyo3::prelude::*;

    #[pymodule_export]
    use super::PyEvaluator;

    #[pymodule_init]
    fn init(module: &Bound<'_, PyModule>) -> PyResult<()> {
        // Keep these values sourced from Rust so Python tooling uses the same defaults as the
        // engine instead of maintaining a second copy.
        module.add("MAX_DEPTH", crate::search::MAX_DEPTH)?;
        module.add("FEATURE_SETS", crate::eval::FeatureSet::NAMES.to_vec())?;
        Ok(())
    }
}
