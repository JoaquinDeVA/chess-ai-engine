//! When a search must stop: the stop flag, the node limit and the hard time limit.

use std::sync::atomic::Ordering::Relaxed;

use super::Search;

/// The clock is read once every this many nodes: reading it costs about as much as a node.
const CLOCK_EVERY_NODES: u64 = 2048;

impl Search<'_> {
    /// Whether the search is out of budget (and stays so: a cut search is flagged `truncated`).
    ///
    /// All three limits are looked at on every call, so the clock schedule does not depend on
    /// which of them cuts.
    pub(super) fn out_of_budget(&mut self) -> bool {
        let stopped = self.control.stop.is_some_and(|flag| flag.load(Relaxed));
        let out_of_nodes = self
            .options
            .node_limit
            .is_some_and(|limit| self.nodes >= limit);
        let out_of_time = self.nodes >= self.next_clock_check && {
            self.next_clock_check = self.nodes + CLOCK_EVERY_NODES;
            (self.control.clock)() - self.control.start >= self.control.current_hard_limit()
        };
        stopped || out_of_nodes || out_of_time || self.truncated
    }
}
