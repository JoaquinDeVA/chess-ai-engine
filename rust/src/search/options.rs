//! Settings of a search: limits and defaults.

/// Iterations that are always completed before the soft time limit may stop the deepening: the first,
/// so that there is always an answer; after it the clock decides.
pub const DEFAULT_MIN_DEPTH: usize = 1;
/// Technical cap of the iterations; the clock stops the deepening long before in a real game.
pub const MAX_DEPTH: usize = 30;
/// Megabytes of the transposition table by default (the `Hash` option of the protocol).
pub const DEFAULT_HASH_MB: usize = 64;
/// Plies of quiescence search of the bots.
pub const DEFAULT_QUIESCENCE_PLIES: u32 = 4;

/// Settings of a search.
#[derive(Clone, Debug)]
pub struct Options {
    pub quiescence_plies: u32,
    pub min_depth: usize,
    /// Nodes an iteration may visit before it is cut (`None`: no limit). Each iteration counts
    /// its own nodes, so a fixed-node search is deterministic whatever the machine.
    pub node_limit: Option<u64>,
    /// Ids (`board::move_id`) of the only moves the root may play (`None`: all of them). The endgame
    /// tables use it to leave out the moves that throw the result away.
    pub root_moves: Option<Vec<u32>>,
}

impl Default for Options {
    fn default() -> Options {
        Options {
            quiescence_plies: DEFAULT_QUIESCENCE_PLIES,
            min_depth: DEFAULT_MIN_DEPTH,
            node_limit: None,
            root_moves: None,
        }
    }
}
