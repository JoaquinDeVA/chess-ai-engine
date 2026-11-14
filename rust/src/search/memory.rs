//! What a game keeps between searches, and the clock and stop switches of one search.

use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;

use super::options::DEFAULT_HASH_MB;
use super::ordering::{ContinuationHistory, Countermoves, History, Killers};
use super::stats::SearchDiagnostics;
use super::tt::Table;

/// What a game keeps between two moves: the transposition table and the killer moves, and how
/// many threads search.
pub struct Memory {
    /// Shared by every thread of a search.
    pub table: Arc<Table>,
    pub killers: Killers,
    pub history: History,
    pub continuation: ContinuationHistory,
    pub countermoves: Countermoves,
    pub threads: usize,
    pub diagnostics: SearchDiagnostics,
}

impl Memory {
    /// A memory for `threads` threads (at least 1) with a table of about `hash_megabytes` MB.
    pub fn new(threads: usize, hash_megabytes: usize) -> Memory {
        Memory {
            table: Arc::new(Table::with_megabytes(hash_megabytes)),
            killers: Killers::default(),
            history: History::default(),
            continuation: ContinuationHistory::default(),
            countermoves: Countermoves::default(),
            threads: threads.max(1),
            diagnostics: SearchDiagnostics::default(),
        }
    }

    pub fn clear(&mut self) {
        self.table.clear();
        self.killers.clear();
        self.history.clear();
        self.continuation.clear();
        self.countermoves.clear();
    }
}

impl Default for Memory {
    fn default() -> Memory {
        Memory::new(1, DEFAULT_HASH_MB)
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
    /// Optional shared hard limit, in clock seconds encoded as an atomic `f64`.
    pub dynamic_hard_limit: Option<&'a AtomicU64>,
    /// Optional shared soft limit, in clock seconds encoded as an atomic `f64`.
    pub dynamic_soft_limit: Option<&'a AtomicU64>,
    /// Optional clock second at which the real budget began (the `ponderhit`); 0 when it began with
    /// the search.
    pub dynamic_budget_start: Option<&'a AtomicU64>,
    pub clock: Clock<'a>,
    pub stop: Option<&'a AtomicBool>,
    /// Depth the main thread is searching, written by it and read by the helpers.
    pub progress: Option<&'a AtomicUsize>,
    /// Helper threads rotate the moves of the root by this much, so that they do not all start with
    /// the same one.
    pub rotation: usize,
}

impl Control<'_> {
    pub fn current_hard_limit(&self) -> f64 {
        self.dynamic_hard_limit.map_or(self.hard_limit, |limit| {
            f64::from_bits(limit.load(Ordering::Acquire))
        })
    }

    pub fn current_soft_limit(&self, fallback: f64) -> f64 {
        self.dynamic_soft_limit.map_or(fallback, |limit| {
            f64::from_bits(limit.load(Ordering::Acquire))
        })
    }

    /// The soft limit with `scale` applied to the budget only: the time spent before the budget
    /// began (pondering) is not scaled.
    pub fn scaled_soft_limit(&self, fallback: f64, scale: f64) -> f64 {
        let begin = self
            .dynamic_budget_start
            .map_or(0.0, |start| f64::from_bits(start.load(Ordering::Acquire)));
        begin + (self.current_soft_limit(fallback) - begin) * scale
    }
}
