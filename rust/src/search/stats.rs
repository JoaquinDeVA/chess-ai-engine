//! What a search measured and why the deepening ended.

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

/// What the search did, counted over every search made with one [`Memory`](super::Memory) since
/// it was created (`uci ttd` prints them; nothing in the search reads them).
#[derive(Clone, Copy, Debug, Default)]
pub struct Counters {
    /// Nodes of the quiescence search (the horizon node itself is not one of them).
    pub quiescence_nodes: u64,
    /// Table lookups, those that found the position, and those that ended the node.
    pub table_probes: u64,
    pub table_hits: u64,
    pub table_cuts: u64,
    /// Beta cutoffs of the main search, and those of the first move tried.
    pub cuts: u64,
    pub first_move_cuts: u64,
    /// Cutoffs by a quiet killer that is not the table move, by its slot (0 is the oldest).
    pub killer_cuts: [u64; 3],
    /// Nodes that are a draw by repetition.
    pub repetition_draws: u64,
    /// Entries stored, and those with a draw by repetition somewhere in their subtree.
    pub stores: u64,
    pub stores_over_repetition: u64,
    /// Stores that dropped an entry of another position, and those where it was deeper.
    pub evictions: u64,
    pub deeper_evictions: u64,
}
