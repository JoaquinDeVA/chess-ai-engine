//! Options, counters and memory of a search: what the caller gives and gets back.

use std::sync::atomic::{AtomicBool, AtomicUsize};
use std::sync::Arc;

use super::shared::SharedTable;
use super::tt::{History, Killers, Table, Tt, DEFAULT_MAX_TT_ENTRIES};

/// Iterations that are always completed before the time estimate may stop the deepening: the first,
/// so that there is always an answer; after it the clock decides.
pub const DEFAULT_MIN_DEPTH: usize = 1;
/// Technical cap of the iterations; the clock stops the deepening long before in a real game.
pub const MAX_DEPTH: usize = 30;
/// Rule that predicts the duration of the next iteration.
pub const DEFAULT_ESTIMATOR: &str = "geometric_growth";
/// Weight of the "win faster / lose slower" preference, in utility units per ply.
pub const DEPTH_PREFERENCE_EPSILON: f64 = 0.003;
/// Plies of quiescence search of the bots.
pub const DEFAULT_QUIESCENCE_PLIES: u32 = 4;

/// Why the deepening ended.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum StopReason {
    MaxDepth,
    Estimate,
    Budget,
    Truncated,
}

impl StopReason {
    pub fn as_str(self) -> &'static str {
        match self {
            StopReason::MaxDepth => "max_depth",
            StopReason::Estimate => "estimate",
            StopReason::Budget => "budget",
            StopReason::Truncated => "truncated",
        }
    }
}

/// One iteration of the deepening: depth, seconds, nodes and whether the clock cut it.
#[derive(Clone, Copy, Debug)]
pub struct IterationInfo {
    pub depth: usize,
    pub seconds: f64,
    pub nodes: u64,
    pub truncated: bool,
}

/// What the search measured.
#[derive(Clone, Debug)]
pub struct Stats {
    pub nodes_visited: u64,
    pub pruning_count: u64,
    pub max_depth_reached: i32,
    pub truncated: bool,
    pub completed_depth: usize,
    pub iterations: Vec<IterationInfo>,
    pub stop_reason: StopReason,
    pub stop_predicted: f64,
}

/// Settings of a search.
#[derive(Clone, Debug)]
pub struct Options {
    pub quiescence_plies: u32,
    pub min_depth: usize,
    pub estimator: String,
    pub epsilon: f64,
    /// Techniques that can be switched off to measure what each one saves (all on by default).
    pub use_table: bool,
    pub use_killers: bool,
    pub use_capture_order: bool,
    /// Techniques of selective search; all off reproduces the reference engine.
    pub techniques: Techniques,
}

/// The techniques added to the reference search, each with its own switch (all off by default).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Techniques {
    /// Update the evaluation move by move instead of from the board at every leaf (same values up
    /// to rounding; only the evaluations that support it, `net4_pst`).
    pub incremental: bool,
    /// Try the best move stored in the table first.
    pub tt_move: bool,
    /// Order quiet moves by how often they caused cut-offs.
    pub history: bool,
    /// Principal variation search: zero-width windows after the first move.
    pub pvs: bool,
    /// Search the root with a narrow window around the previous value.
    pub aspiration: bool,
    /// Prune a node if passing the turn already cuts.
    pub null_move: bool,
    /// Search late quiet moves with less depth.
    pub lmr: bool,
}

impl Techniques {
    /// Names, as the switches are called from Python.
    pub const NAMES: [&'static str; 7] = [
        "incremental",
        "tt_move",
        "history",
        "pvs",
        "aspiration",
        "null_move",
        "lmr",
    ];

    /// The techniques with the given names.
    pub fn from_names<S: AsRef<str>>(names: &[S]) -> Result<Techniques, String> {
        let mut techniques = Techniques::default();
        for name in names {
            match name.as_ref() {
                "incremental" => techniques.incremental = true,
                "tt_move" => techniques.tt_move = true,
                "history" => techniques.history = true,
                "pvs" => techniques.pvs = true,
                "aspiration" => techniques.aspiration = true,
                "null_move" => techniques.null_move = true,
                "lmr" => techniques.lmr = true,
                other => {
                    return Err(format!(
                        "unknown technique {other:?}; available: {:?}",
                        Techniques::NAMES
                    ))
                }
            }
        }
        Ok(techniques)
    }

    /// Every technique on.
    pub fn all() -> Techniques {
        Techniques {
            incremental: true,
            tt_move: true,
            history: true,
            pvs: true,
            aspiration: true,
            null_move: true,
            lmr: true,
        }
    }
}

impl Default for Options {
    fn default() -> Options {
        Options {
            quiescence_plies: 0,
            min_depth: DEFAULT_MIN_DEPTH,
            estimator: DEFAULT_ESTIMATOR.to_string(),
            epsilon: DEPTH_PREFERENCE_EPSILON,
            use_table: true,
            use_killers: true,
            use_capture_order: true,
            techniques: Techniques::default(),
        }
    }
}

/// What a game keeps between two moves: the transposition table and the killer moves, and how
/// many threads search.
pub struct Memory {
    pub table: Tt,
    pub killers: Killers,
    pub history: History,
    pub threads: usize,
}

impl Memory {
    /// A single-thread memory with the exact, bounded table of the Python engine.
    pub fn new(max_tt_entries: usize) -> Memory {
        Memory {
            table: Tt::Local(Table::new(max_tt_entries)),
            killers: Killers::default(),
            history: History::default(),
            threads: 1,
        }
    }

    /// A memory for `threads` threads (at least 1): with one, the exact table; with more, a
    /// lock-free table of `hash_megabytes` shared by all of them.
    pub fn with_threads(max_tt_entries: usize, threads: usize, hash_megabytes: usize) -> Memory {
        if threads <= 1 {
            return Memory::new(max_tt_entries);
        }
        Memory {
            table: Tt::Shared(Arc::new(SharedTable::with_megabytes(hash_megabytes))),
            killers: Killers::default(),
            history: History::default(),
            threads,
        }
    }

    pub fn clear(&mut self) {
        self.table.clear();
        self.killers.clear();
        self.history.clear();
    }
}

impl Default for Memory {
    fn default() -> Memory {
        Memory::new(DEFAULT_MAX_TT_ENTRIES)
    }
}

/// A clock: seconds since some fixed instant.
pub type Clock<'a> = &'a (dyn Fn() -> f64 + Sync);

/// How a search is controlled from outside: the clock, the hard limit and, for the helper
/// threads, the flag that tells them to stop.
pub struct Control<'a> {
    /// Time (in the units of `clock`) the elapsed time is measured from.
    pub start: f64,
    /// Elapsed time after which the search is cut.
    pub hard_limit: f64,
    pub clock: Clock<'a>,
    pub stop: Option<&'a AtomicBool>,
    /// Depth the main thread is searching, written by it and read by the helpers.
    pub progress: Option<&'a AtomicUsize>,
    /// Helper threads rotate the moves of the root by this much, so that they do not all start with
    /// the same one.
    pub rotation: usize,
}
