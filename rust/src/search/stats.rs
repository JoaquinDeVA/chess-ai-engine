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
