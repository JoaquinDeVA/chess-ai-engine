//! What a game keeps between searches, and the clock and stop switches of one search.

use std::sync::atomic::{AtomicBool, AtomicUsize};
use std::sync::Arc;

use super::ordering::{History, Killers};
use super::shared::SharedTable;
use super::tt::{Table, Tt, DEFAULT_MAX_TT_ENTRIES};

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
