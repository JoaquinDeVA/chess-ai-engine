//! What a game keeps between searches, and the clock and stop switches of one search.

use std::sync::Arc;

use super::diagnostics::SearchDiagnostics;
use super::options::DEFAULT_HASH_MB;
use super::ordering::{ContinuationHistory, Countermoves, History, Killers};
use super::transposition_table::Table;

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
