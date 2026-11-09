//! The Python module `chess_engine._core`.
//!
//! The evaluations, for the laboratory. The search, the opening book and the endgame tables run
//! only in the UCI binary (`src/bin/uci.rs`); Python reads its defaults from here.

use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use shakmaty::Color;

use crate::board::position_from_fen;
use crate::eval::classic_heuristic::H1;
use crate::eval::sparse_features::FeatureSet;
use crate::eval::sparse_net::{Layer, Net};
use crate::eval::{Eval, Evaluator};

fn value_error<E: std::fmt::Display>(error: E) -> PyErr {
    PyValueError::new_err(error.to_string())
}

fn parse_features(name: &str) -> PyResult<FeatureSet> {
    FeatureSet::from_name(name).ok_or_else(|| value_error(format!("unknown feature set {name:?}")))
}

/// An evaluation of positions.
#[pyclass(name = "Evaluator", frozen)]
struct PyEvaluator {
    eval: Eval,
    features: Option<FeatureSet>,
}

#[pymethods]
impl PyEvaluator {
    /// The heuristic `h1` with its six weights.
    #[staticmethod]
    fn h1(weights: Vec<f64>) -> PyResult<PyEvaluator> {
        let weights: [f64; 6] = weights
            .try_into()
            .map_err(|_| value_error("h1 needs exactly six weights"))?;
        Ok(PyEvaluator {
            eval: Eval::H1(H1::new(weights)),
            features: None,
        })
    }

    /// A network over sparse features: `layers` are `(weights, bias)` pairs, accumulator first.
    #[staticmethod]
    fn network(
        features: &str,
        layers: Vec<(PyReadonlyArray2<f64>, PyReadonlyArray1<f64>)>,
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
        let net = Net::new(set, layers).map_err(value_error)?;
        Ok(PyEvaluator {
            eval: Eval::Net(net),
            features: Some(set),
        })
    }

    /// Value of the position for White (`white=True`) or Black, in `[0.01, 0.99]`.
    fn evaluate(&self, fen: &str, white: bool) -> PyResult<f64> {
        let pos = position_from_fen(fen).map_err(value_error)?;
        Ok(self
            .eval
            .evaluate(&pos, if white { Color::White } else { Color::Black }))
    }

    /// Active feature indices and values of a position (networks only).
    fn features(&self, fen: &str) -> PyResult<(Vec<usize>, Vec<f64>)> {
        let set = self
            .features
            .ok_or_else(|| value_error("this evaluator has no sparse features"))?;
        let pos = position_from_fen(fen).map_err(value_error)?;
        let (mut indices, mut values) = (Vec::new(), Vec::new());
        set.active(&pos, &mut indices, &mut values);
        Ok((indices, values))
    }
}

/// Evaluation of the chess engine, in Rust, and the defaults of its search.
#[pymodule]
mod _core {
    use pyo3::prelude::*;

    #[pymodule_export]
    use super::PyEvaluator;

    #[pymodule_init]
    fn init(module: &Bound<'_, PyModule>) -> PyResult<()> {
        module.add("MAX_DEPTH", crate::search::MAX_DEPTH)?;
        module.add(
            "DEFAULT_TECHNIQUES",
            crate::search::DEFAULT_TECHNIQUES.to_vec(),
        )?;
        module.add("TECHNIQUES", crate::search::Techniques::NAMES.to_vec())?;
        module.add(
            "FEATURE_SETS",
            crate::eval::sparse_features::FeatureSet::NAMES.to_vec(),
        )?;
        Ok(())
    }
}
