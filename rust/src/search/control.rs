//! How a search is controlled from outside: the clock and the switches that stop it.

use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};

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
