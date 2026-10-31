//! The Python module `chess_engine._core`.
//!
//! One call per move: Python hands over a position (root FEN and the moves played from it, so that
//! repetitions count), an evaluator and the limits, and gets the move and the statistics back.

use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use shakmaty::Color;

use crate::board::{position_from_fen, uci_string, GameState};
use crate::eval::features::{FeatureSet, Perspective};
use crate::eval::h1::H1;
use crate::eval::linear::LinearModel;
use crate::eval::net::{Layer, Net};
use crate::eval::{Eval, Evaluator};
use crate::knowledge::book::OpeningBook;
use crate::knowledge::tablebase::EndgameTablebase;
use crate::search::{self, Control, Memory, Options};

fn value_error<E: std::fmt::Display>(error: E) -> PyErr {
    PyValueError::new_err(error.to_string())
}

fn parse_perspective(name: &str) -> PyResult<Perspective> {
    match name {
        "white" => Ok(Perspective::White),
        "side_to_move" => Ok(Perspective::SideToMove),
        other => Err(value_error(format!("unknown perspective {other:?}"))),
    }
}

fn parse_features(name: &str) -> PyResult<FeatureSet> {
    FeatureSet::from_name(name).ok_or_else(|| value_error(format!("unknown feature set {name:?}")))
}

/// An evaluation of positions.
#[pyclass(name = "Evaluator", frozen)]
struct PyEvaluator {
    eval: Arc<Eval>,
    features: Option<(FeatureSet, Perspective)>,
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
            eval: Arc::new(Eval::H1(H1::new(weights))),
            features: None,
        })
    }

    /// A logistic model over sparse features.
    #[staticmethod]
    fn linear(
        features: &str,
        perspective: &str,
        weights: PyReadonlyArray1<f64>,
        bias: f64,
    ) -> PyResult<PyEvaluator> {
        let (set, view) = (parse_features(features)?, parse_perspective(perspective)?);
        let model =
            LinearModel::new(set, view, weights.as_array().to_vec(), bias).map_err(value_error)?;
        Ok(PyEvaluator {
            eval: Arc::new(Eval::Linear(model)),
            features: Some((set, view)),
        })
    }

    /// A network over sparse features: `layers` are `(weights, bias)` pairs, accumulator first.
    #[staticmethod]
    fn network(
        features: &str,
        perspective: &str,
        layers: Vec<(PyReadonlyArray2<f64>, PyReadonlyArray1<f64>)>,
    ) -> PyResult<PyEvaluator> {
        let (set, view) = (parse_features(features)?, parse_perspective(perspective)?);
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
        let net = Net::new(set, view, layers).map_err(value_error)?;
        Ok(PyEvaluator {
            eval: Arc::new(Eval::Net(net)),
            features: Some((set, view)),
        })
    }

    /// Value of the position for White (`white=True`) or Black, in `[0.01, 0.99]`.
    fn evaluate(&self, fen: &str, white: bool) -> PyResult<f64> {
        let pos = position_from_fen(fen).map_err(value_error)?;
        Ok(self
            .eval
            .evaluate(&pos, if white { Color::White } else { Color::Black }))
    }

    /// Active feature indices and values of a position (models over sparse features only).
    fn features(&self, fen: &str) -> PyResult<(Vec<usize>, Vec<f64>)> {
        let (set, view) = self
            .features
            .ok_or_else(|| value_error("this evaluator has no sparse features"))?;
        let pos = position_from_fen(fen).map_err(value_error)?;
        let (mut indices, mut values) = (Vec::new(), Vec::new());
        set.active(&pos, view, &mut indices, &mut values);
        Ok((indices, values))
    }
}

/// An opening book.
#[pyclass(name = "Book", frozen)]
struct PyBook {
    book: OpeningBook,
}

#[pymethods]
impl PyBook {
    /// Load the book from a JSON file `{fen: uci}`.
    #[staticmethod]
    fn load(path: &str) -> PyResult<PyBook> {
        Ok(PyBook {
            book: OpeningBook::load(Path::new(path)).map_err(value_error)?,
        })
    }

    /// A book from a dictionary `{fen: uci}`.
    #[staticmethod]
    fn from_entries(entries: Vec<(String, String)>) -> PyBook {
        PyBook {
            book: OpeningBook::from_entries(entries),
        }
    }

    fn __len__(&self) -> usize {
        self.book.len()
    }

    /// The book move (UCI) for the position, or `None`.
    fn lookup(&self, fen: &str) -> PyResult<Option<String>> {
        let pos = position_from_fen(fen).map_err(value_error)?;
        Ok(self.book.lookup(&pos).map(|m| uci_string(&m)))
    }
}

/// Syzygy endgame tables.
#[pyclass(name = "Tablebase", frozen)]
struct PyTablebase {
    tables: EndgameTablebase,
}

#[pymethods]
impl PyTablebase {
    /// Open the tables of a folder.
    #[staticmethod]
    fn open(directory: &str) -> PyResult<PyTablebase> {
        Ok(PyTablebase {
            tables: EndgameTablebase::open(Path::new(directory)).map_err(value_error)?,
        })
    }

    /// Whether the position can be probed (false for a position that is not legal).
    fn covers(&self, fen: &str) -> bool {
        position_from_fen(fen).is_ok_and(|pos| self.tables.covers(&pos))
    }

    /// Result for the side to move (2 win ... -2 loss), or `None` if a table is missing.
    fn wdl(&self, fen: &str) -> Option<i32> {
        position_from_fen(fen)
            .ok()
            .and_then(|pos| self.tables.wdl(&pos))
    }

    /// The best move (UCI) according to the tables, or `None`.
    fn best_move(&self, fen: &str) -> Option<String> {
        let pos = position_from_fen(fen).ok()?;
        self.tables.best_move(&pos).map(|m| uci_string(&m))
    }
}

/// What a game keeps between moves, and the searches.
#[pyclass(name = "Searcher")]
struct PySearcher {
    memory: Memory,
}

fn options(
    quiescence_plies: u32,
    min_depth: usize,
    estimator: &str,
    disabled: &[String],
    techniques: &[String],
) -> PyResult<Options> {
    let mut options = Options {
        quiescence_plies,
        min_depth,
        estimator: estimator.to_string(),
        techniques: search::Techniques::from_names(techniques).map_err(value_error)?,
        ..Options::default()
    };
    for name in disabled {
        match name.as_str() {
            "table" => options.use_table = false,
            "killers" => options.use_killers = false,
            "ordering" => options.use_capture_order = false,
            other => {
                return Err(value_error(format!(
                    "unknown technique {other:?}; available: table, killers, ordering"
                )))
            }
        }
    }
    Ok(options)
}

#[pymethods]
impl PySearcher {
    /// A searcher. With `threads > 1` the threads share a lock-free table of `hash_mb` MB and
    /// `max_tt_entries` is ignored.
    #[new]
    #[pyo3(signature = (max_tt_entries=search::tt::DEFAULT_MAX_TT_ENTRIES, threads=1, hash_mb=64))]
    fn new(max_tt_entries: usize, threads: usize, hash_mb: usize) -> PySearcher {
        PySearcher {
            memory: Memory::with_threads(max_tt_entries, threads.max(1), hash_mb),
        }
    }

    /// Forget the table and the killer moves (a new game).
    fn clear(&mut self) {
        self.memory.clear();
    }

    /// Number of threads that search.
    #[getter]
    fn threads(&self) -> usize {
        self.memory.threads
    }

    /// Number of entries of the transposition table.
    fn table_size(&self) -> usize {
        self.memory.table.len()
    }

    /// Iterative deepening. Returns `(move, statistics)`; `move` is a UCI string or `None`.
    #[pyo3(signature = (root_fen, moves, evaluator, max_depth, time_limit, hard_limit, quiescence_plies, min_depth, estimator, disabled=Vec::new(), techniques=Vec::new()))]
    #[allow(clippy::too_many_arguments)]
    fn search<'py>(
        &mut self,
        py: Python<'py>,
        root_fen: &str,
        moves: Vec<String>,
        evaluator: &PyEvaluator,
        max_depth: usize,
        time_limit: f64,
        hard_limit: f64,
        quiescence_plies: u32,
        min_depth: usize,
        estimator: &str,
        disabled: Vec<String>,
        techniques: Vec<String>,
    ) -> PyResult<(Option<String>, Bound<'py, PyDict>)> {
        let mut state = GameState::from_moves(root_fen, &moves).map_err(value_error)?;
        let options = options(
            quiescence_plies,
            min_depth,
            estimator,
            &disabled,
            &techniques,
        )?;
        let eval = evaluator.eval.clone();
        let memory = &mut self.memory;
        let run = move || {
            let origin = Instant::now();
            let clock = || origin.elapsed().as_secs_f64();
            let control = Control {
                start: clock(),
                hard_limit,
                clock: &clock,
                stop: None,
                progress: None,
                rotation: 0,
            };
            search::iterative_deepening(
                &mut state, &*eval, memory, &options, max_depth, time_limit, &control,
            )
        };
        let result = py.detach(run);
        let (best, stats) = result.map_err(value_error)?;
        Ok((best.as_ref().map(uci_string), stats_dict(py, &stats)?))
    }
}

fn stats_dict<'py>(py: Python<'py>, stats: &search::Stats) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("nodes_visited", stats.nodes_visited)?;
    dict.set_item("pruning_count", stats.pruning_count)?;
    dict.set_item("max_depth_reached", stats.max_depth_reached)?;
    dict.set_item("truncated", stats.truncated)?;
    dict.set_item("completed_depth", stats.completed_depth)?;
    let iterations = PyList::empty(py);
    for it in &stats.iterations {
        iterations.append((it.depth, it.seconds, it.nodes, it.truncated))?;
    }
    dict.set_item("iterations", iterations)?;
    dict.set_item("stop_reason", stats.stop_reason.as_str())?;
    dict.set_item("stop_predicted", stats.stop_predicted)?;
    Ok(dict)
}

/// Predicted seconds of the next iteration, by the named estimator, from `(seconds, nodes)` pairs.
#[pyfunction]
fn estimate(name: &str, history: Vec<(f64, u64)>) -> PyResult<f64> {
    let estimator = search::estimators::get(name).ok_or_else(|| {
        value_error(format!(
            "unknown estimator {name:?}; available: {:?}",
            search::estimators::NAMES
        ))
    })?;
    if history.is_empty() {
        return Err(value_error("the history of iterations is empty"));
    }
    Ok(estimator(&history))
}

/// Version of the core.
#[pyfunction]
fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

/// Search, evaluation and knowledge of the chess engine, in Rust.
#[pymodule]
mod _core {
    use pyo3::prelude::*;

    #[pymodule_export]
    use super::{estimate, version, PyBook, PyEvaluator, PySearcher, PyTablebase};

    #[pymodule_init]
    fn init(module: &Bound<'_, PyModule>) -> PyResult<()> {
        module.add("DEFAULT_ESTIMATOR", crate::search::DEFAULT_ESTIMATOR)?;
        module.add("DEFAULT_MIN_DEPTH", crate::search::DEFAULT_MIN_DEPTH)?;
        module.add("MAX_DEPTH", crate::search::MAX_DEPTH)?;
        module.add(
            "DEFAULT_QUIESCENCE_PLIES",
            crate::search::DEFAULT_QUIESCENCE_PLIES,
        )?;
        module.add(
            "DEFAULT_MAX_TT_ENTRIES",
            crate::search::tt::DEFAULT_MAX_TT_ENTRIES,
        )?;
        module.add("TECHNIQUES", crate::search::Techniques::NAMES.to_vec())?;
        module.add(
            "FEATURE_SETS",
            crate::eval::features::FeatureSet::NAMES.to_vec(),
        )?;
        module.add("ESTIMATOR_NAMES", crate::search::estimators::NAMES.to_vec())?;
        Ok(())
    }
}
