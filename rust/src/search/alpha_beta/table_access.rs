//! The transposition table at a node: what an entry can answer, and what the node stores.

use shakmaty::Position;

use super::{NodeResult, Search};
use crate::search::scores::{at_ply, for_table};
use crate::search::transposition_table::Bound;

/// What the table said about a node.
pub(super) struct TableProbe {
    /// The best move the entry remembers, to be tried first.
    pub(super) table_move: Option<u32>,
    /// Whether a value searched this deep cannot depend on the fifty-move clock.
    pub(super) fifty_move_safe: bool,
    /// The value that ends the node, when the entry is deep enough and exact for the window.
    pub(super) cut: Option<NodeResult>,
}

impl Search<'_> {
    /// Ask the table about the node, narrowing `alpha` and `beta` with a bound it gives.
    pub(super) fn probe_table(
        &mut self,
        key: u64,
        depth_remaining: i32,
        ply: i32,
        alpha: &mut i32,
        beta: &mut i32,
    ) -> TableProbe {
        // A value searched `depth_remaining` plies deep can depend on the fifty-move clock only if
        // the clock can reach 100 inside that search; then the entry gives its move and nothing
        // else. A position with a repetition behind it is a possible draw: the table is not asked.
        let fifty_move_safe = self.state.pos().halfmoves() as i32 + depth_remaining < 100;
        let entry = if self.state.has_repeated() {
            None
        } else {
            self.memory.table.probe(&key)
        };
        let table_move = entry.and_then(|e| e.best);
        self.memory.diagnostics.table_probe();
        let mut cut = None;
        if let Some(entry) = entry {
            self.memory.diagnostics.table_hit();
            // A restricted root is never cut by the table: its move may be one the list excludes.
            let restricted_root = ply == 0 && self.options.root_moves.is_some();
            if fifty_move_safe && entry.depth >= depth_remaining && !restricted_root {
                let value = at_ply(entry.value, ply);
                match entry.flag {
                    Bound::Exact => {
                        self.memory.diagnostics.table_cut();
                        cut = Some((value, entry.best));
                    }
                    Bound::Lower => *alpha = (*alpha).max(value),
                    Bound::Upper => *beta = (*beta).min(value),
                }
                if cut.is_none() && *alpha >= *beta {
                    self.memory.diagnostics.table_cut();
                    cut = Some((value, entry.best));
                }
            }
        }
        TableProbe {
            table_move,
            fifty_move_safe,
            cut,
        }
    }

    /// Store the result of a node searched in the window `(alpha, beta)` it came in with.
    pub(super) fn store_result(
        &mut self,
        key: u64,
        (value, best): NodeResult,
        depth_remaining: i32,
        original_window: (i32, i32),
        ply: i32,
        repetitions_before: u64,
    ) {
        // Mates are stored by their distance from this node, not from the root.
        let evicted = self.memory.table.store(
            key,
            for_table(value, ply),
            best,
            depth_remaining,
            (
                for_table(original_window.0, ply),
                for_table(original_window.1, ply),
            ),
        );
        let over_repetition = self.memory.diagnostics.repetition_draws() > repetitions_before;
        self.memory
            .diagnostics
            .store(over_repetition, evicted, depth_remaining);
    }
}
