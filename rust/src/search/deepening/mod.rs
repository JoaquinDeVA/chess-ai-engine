//! Iterative deepening: helper threads and the clock policy.

mod clock_policy;
mod iteration;
mod lazy_smp;
mod voting;

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering::Relaxed};
use std::sync::Mutex;

use shakmaty::Move;

use super::control::Control;
use super::memory::Memory;
use super::options::*;
use super::stats::*;
use crate::board::GameState;
use crate::eval::Net;
use iteration::deepen;
use lazy_smp::{helper, Report};
use voting::vote;

/// Iterative deepening over [`alpha_beta`]: depth 1, 2, ... `max_depth`, returning the move of the
/// deepest **completed** iteration.
///
/// After each completed iteration the loop stops if the elapsed time passes `time_limit` (the soft
/// limit) times [`soft_scale`]: more time when the best move has just changed or shares the nodes
/// with others, less when it holds and takes most of them. The soft limit is ignored until
/// `min_depth` iterations are complete; the hard limit (`control.hard_limit`) cuts the tree at any
/// time. An iteration cut by the hard limit is discarded unless it is the very first.
///
/// With `memory.threads > 1` the other threads search the same position and share the table (Lazy
/// SMP); the move is always the one of the main thread. `control.stop` and `control.rotation` are
/// those of the main thread and are not used by the helpers' copies.
pub fn iterative_deepening(
    state: &mut GameState,
    eval: &Net,
    memory: &mut Memory,
    options: &Options,
    max_depth: usize,
    time_limit: f64,
    control: &Control,
) -> Result<(Option<Move>, Stats), String> {
    memory.table.new_search();
    if memory.threads <= 1 {
        return deepen(state, eval, memory, options, max_depth, time_limit, control)
            .map(|(best, stats, _)| (best, stats));
    }
    let (table, extra) = (memory.table.clone(), memory.threads - 1);
    // Helpers only fill the shared table; the main thread alone decides the move, so the result
    // does not depend on how the threads interleave. They stop when the main thread is done.
    let stop = AtomicBool::new(false);
    let helper_nodes = std::sync::atomic::AtomicU64::new(0);
    let progress = AtomicUsize::new(0);
    let copies: Vec<GameState> = (0..extra).map(|_| state.clone()).collect();
    let reports: Vec<Report> = (0..extra).map(|_| Mutex::new(None)).collect();
    let history = memory.history.clone();
    let continuation = memory.continuation.clone();
    let result = std::thread::scope(|scope| {
        for (index, copy) in copies.into_iter().enumerate() {
            let (table, stop, nodes, progress) = (table.clone(), &stop, &helper_nodes, &progress);
            let (history, continuation, report) =
                (history.clone(), continuation.clone(), &reports[index]);
            scope.spawn(move || {
                let helper_control = Control {
                    start: control.start,
                    hard_limit: control.hard_limit,
                    dynamic_hard_limit: control.dynamic_hard_limit,
                    dynamic_soft_limit: control.dynamic_soft_limit,
                    dynamic_budget_start: control.dynamic_budget_start,
                    clock: control.clock,
                    stop: Some(stop),
                    progress: Some(progress),
                    rotation: index + 1,
                };
                nodes.fetch_add(
                    helper(
                        index,
                        copy,
                        eval,
                        table,
                        history,
                        continuation,
                        options,
                        max_depth,
                        &helper_control,
                        report,
                    ),
                    Relaxed,
                );
            });
        }
        let main_control = Control {
            start: control.start,
            hard_limit: control.hard_limit,
            dynamic_hard_limit: control.dynamic_hard_limit,
            dynamic_soft_limit: control.dynamic_soft_limit,
            dynamic_budget_start: control.dynamic_budget_start,
            clock: control.clock,
            stop: control.stop,
            progress: Some(&progress),
            rotation: control.rotation,
        };
        let result = deepen(
            state,
            eval,
            memory,
            options,
            max_depth,
            time_limit,
            &main_control,
        );
        stop.store(true, Relaxed);
        result
    });
    result.map(|(best, mut stats, value)| {
        stats.nodes_visited += helper_nodes.load(Relaxed);
        let best = match value {
            Some(value) => vote((stats.completed_depth, value, best), &reports),
            None => best,
        };
        (best, stats)
    })
}
