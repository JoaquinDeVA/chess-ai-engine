//! What a search measured and why the deepening ended.

/// Why the deepening ended.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum StopReason {
    MaxDepth,
    Budget,
    Truncated,
}

impl StopReason {
    pub fn as_str(self) -> &'static str {
        match self {
            StopReason::MaxDepth => "max_depth",
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
    #[cfg(feature = "search-summary")]
    pub pruning_count: u64,
    #[cfg(feature = "search-summary")]
    pub max_depth_reached: i32,
    pub truncated: bool,
    pub completed_depth: usize,
    pub iterations: Vec<IterationInfo>,
    pub stop_reason: StopReason,
}

/// Optional per-iteration measurements that do not affect the selected move.
#[cfg(feature = "search-summary")]
#[derive(Clone, Copy, Debug, Default)]
pub struct SearchReport {
    pub prunings: u64,
    pub max_ply: i32,
}

/// Detailed counters collected over every search made with one [`Memory`](super::Memory).
///
/// The counters are enabled by `search-diagnostics`. Without that feature the type is empty and
/// its recording methods are inlined away, while the search keeps the state it needs to stop and
/// choose a move.
#[derive(Clone, Copy, Debug, Default)]
pub struct SearchDiagnostics {
    /// Nodes of the quiescence search (the horizon node itself is not one of them).
    #[cfg(feature = "search-diagnostics")]
    pub quiescence_nodes: u64,
    /// Table lookups, those that found the position, and those that ended the node.
    #[cfg(feature = "search-diagnostics")]
    pub table_probes: u64,
    #[cfg(feature = "search-diagnostics")]
    pub table_hits: u64,
    #[cfg(feature = "search-diagnostics")]
    pub table_cuts: u64,
    /// Beta cutoffs of the main search, and those of the first move tried.
    #[cfg(feature = "search-diagnostics")]
    pub cuts: u64,
    #[cfg(feature = "search-diagnostics")]
    pub first_move_cuts: u64,
    /// Cutoffs by a quiet killer that is not the table move, by its slot (0 is the oldest).
    #[cfg(feature = "search-diagnostics")]
    pub killer_cuts: [u64; 3],
    /// Nodes that are a draw by repetition.
    #[cfg(feature = "search-diagnostics")]
    pub repetition_draws: u64,
    /// Entries stored, and those with a draw by repetition somewhere in their subtree.
    #[cfg(feature = "search-diagnostics")]
    pub stores: u64,
    #[cfg(feature = "search-diagnostics")]
    pub stores_over_repetition: u64,
    /// Stores that dropped an entry of another position, and those where it was deeper.
    #[cfg(feature = "search-diagnostics")]
    pub evictions: u64,
    #[cfg(feature = "search-diagnostics")]
    pub deeper_evictions: u64,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct DiagnosticsSnapshot {
    pub quiescence_nodes: u64,
    pub table_probes: u64,
    pub table_hits: u64,
    pub table_cuts: u64,
    pub cuts: u64,
    pub first_move_cuts: u64,
    pub killer_cuts: [u64; 3],
    pub repetition_draws: u64,
    pub stores: u64,
    pub stores_over_repetition: u64,
    pub evictions: u64,
    pub deeper_evictions: u64,
}

impl SearchDiagnostics {
    #[inline]
    pub fn table_probe(&mut self) {
        #[cfg(feature = "search-diagnostics")]
        {
            self.table_probes += 1;
        }
    }

    #[inline]
    pub fn table_hit(&mut self) {
        #[cfg(feature = "search-diagnostics")]
        {
            self.table_hits += 1;
        }
    }

    #[inline]
    pub fn table_cut(&mut self) {
        #[cfg(feature = "search-diagnostics")]
        {
            self.table_cuts += 1;
        }
    }

    #[inline]
    pub fn repetition_draw(&mut self) {
        #[cfg(feature = "search-diagnostics")]
        {
            self.repetition_draws += 1;
        }
    }

    #[inline]
    pub fn repetition_draws(&self) -> u64 {
        #[cfg(feature = "search-diagnostics")]
        {
            return self.repetition_draws;
        }
        #[cfg(not(feature = "search-diagnostics"))]
        0
    }

    #[inline]
    pub fn cutoff(&mut self, first_move: bool, killer_slot: Option<usize>) {
        #[cfg(feature = "search-diagnostics")]
        {
            self.cuts += 1;
            self.first_move_cuts += u64::from(first_move);
            if let Some(slot) = killer_slot {
                self.killer_cuts[slot] += 1;
            }
        }
        #[cfg(not(feature = "search-diagnostics"))]
        let _ = (first_move, killer_slot);
    }

    #[inline]
    pub fn quiescence_node(&mut self) {
        #[cfg(feature = "search-diagnostics")]
        {
            self.quiescence_nodes += 1;
        }
    }

    #[inline]
    pub fn store(&mut self, over_repetition: bool, evicted_depth: Option<i32>, depth: i32) {
        #[cfg(feature = "search-diagnostics")]
        {
            self.stores += 1;
            self.stores_over_repetition += u64::from(over_repetition);
            if let Some(evicted_depth) = evicted_depth {
                self.evictions += 1;
                self.deeper_evictions += u64::from(evicted_depth > depth);
            }
        }
        #[cfg(not(feature = "search-diagnostics"))]
        let _ = (over_repetition, evicted_depth, depth);
    }

    #[inline]
    pub fn snapshot(&self) -> DiagnosticsSnapshot {
        #[cfg(feature = "search-diagnostics")]
        {
            return DiagnosticsSnapshot {
                quiescence_nodes: self.quiescence_nodes,
                table_probes: self.table_probes,
                table_hits: self.table_hits,
                table_cuts: self.table_cuts,
                cuts: self.cuts,
                first_move_cuts: self.first_move_cuts,
                killer_cuts: self.killer_cuts,
                repetition_draws: self.repetition_draws,
                stores: self.stores,
                stores_over_repetition: self.stores_over_repetition,
                evictions: self.evictions,
                deeper_evictions: self.deeper_evictions,
            };
        }
        #[cfg(not(feature = "search-diagnostics"))]
        DiagnosticsSnapshot::default()
    }
}
