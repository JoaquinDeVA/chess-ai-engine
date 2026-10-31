//! Iterative deepening: aspiration windows, helper threads and the clock policy.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering::Relaxed};
use std::sync::Arc;

use shakmaty::Move;

use super::estimators;
use super::node::{alpha_beta, alpha_beta_window, SearchResult};
use super::options::*;
use super::shared::SharedTable;
use super::tt::{History, Killers, Tt};
use crate::board::GameState;
use crate::eval::Evaluator;

/// Tolerance for finishing a depth that was started within the budget.
const OVERSHOOT_FACTOR: f64 = 1.15;

/// First depth searched with an aspiration window (the first iterations are too unstable).
const ASPIRATION_FROM_DEPTH: i32 = 4;
/// Half-width of the first aspiration window, in expected score.
const ASPIRATION_DELTA: f64 = 0.02;
/// Half-width beyond which the window is the whole interval.
const ASPIRATION_LIMIT: f64 = 0.4;

/// One iteration of the deepening: [`alpha_beta`], or, with the aspiration technique, a search
/// in a narrow window around the previous value that is repeated wider when the value falls
/// outside. The counters are those of all the searches of the iteration. A search cut by the
/// clock ends the iteration at once.
fn iteration(
    state: &mut GameState,
    eval: &dyn Evaluator,
    memory: &mut Memory,
    options: &Options,
    depth: i32,
    control: &Control,
    previous: Option<f64>,
) -> SearchResult {
    let Some(value) =
        previous.filter(|_| options.techniques.aspiration && depth >= ASPIRATION_FROM_DEPTH)
    else {
        return alpha_beta(state, eval, memory, options, depth, control);
    };
    let mut delta = ASPIRATION_DELTA;
    let (mut low, mut high) = (value - delta, value + delta);
    let (mut nodes, mut prunings, mut max_ply) = (0, 0, 0);
    loop {
        let mut result =
            alpha_beta_window(state, eval, memory, options, depth, control, (low, high));
        nodes += result.nodes;
        prunings += result.prunings;
        max_ply = max_ply.max(result.max_ply);
        let inside = result.value > low && result.value < high;
        if result.truncated || inside || (low == f64::NEG_INFINITY && high == f64::INFINITY) {
            result.nodes = nodes;
            result.prunings = prunings;
            result.max_ply = max_ply;
            return result;
        }
        delta *= 4.0;
        if delta > ASPIRATION_LIMIT {
            (low, high) = (f64::NEG_INFINITY, f64::INFINITY);
        } else if result.value <= low {
            low = value - delta;
        } else {
            high = value + delta;
        }
    }
}

/// The work of a helper thread: iterative deepening on its own copy of the position that only
/// fills the shared table. It searches one or two plies deeper than the main thread is, rotates the moves
/// of the root and stops when the flag is raised. Returns the nodes it visited.
fn helper(
    index: usize,
    mut state: GameState,
    eval: &dyn Evaluator,
    table: Arc<SharedTable>,
    options: &Options,
    max_depth: usize,
    control: &Control,
) -> u64 {
    let mut memory = Memory {
        table: Tt::Shared(table),
        killers: Killers::default(),
        history: History::default(),
        threads: 1,
    };
    let mut nodes = 0;
    while !control.stop.is_some_and(|flag| flag.load(Relaxed)) {
        // Stay ahead of the main thread: the table is useful with what it will need next.
        let main_depth = control.progress.map_or(1, |p| p.load(Relaxed));
        let depth = (main_depth + 1 + index % 2).min(max_depth).max(1);
        let result = alpha_beta(
            &mut state,
            eval,
            &mut memory,
            options,
            depth as i32,
            control,
        );
        nodes += result.nodes;
        if result.truncated {
            break;
        }
    }
    nodes
}

/// Iterative deepening over [`alpha_beta`]: depth 1, 2, ... `max_depth`, returning the move of the
/// deepest **completed** iteration.
///
/// Before starting an iteration the time it would take is estimated from the finished ones and the
/// loop stops if it does not fit in what is left of `time_limit` (the soft budget). The estimate and
/// the soft budget are ignored until `min_depth` iterations are complete; only the hard limit can
/// stop earlier. An iteration cut by the hard limit is discarded unless it is the very first.
///
/// With `memory.threads > 1` the other threads search the same position and share the table (Lazy
/// SMP); the move is always the one of the main thread. `control.stop` and `control.rotation` are
/// those of the main thread and are not used by the helpers' copies.
pub fn iterative_deepening(
    state: &mut GameState,
    eval: &dyn Evaluator,
    memory: &mut Memory,
    options: &Options,
    max_depth: usize,
    time_limit: f64,
    control: &Control,
) -> Result<(Option<Move>, Stats), String> {
    let helpers = match &memory.table {
        Tt::Shared(table) if memory.threads > 1 => Some((table.clone(), memory.threads - 1)),
        _ => None,
    };
    let Some((table, extra)) = helpers else {
        return deepen(state, eval, memory, options, max_depth, time_limit, control);
    };
    table.new_search();
    // Helpers only fill the shared table; the main thread alone decides the move, so the result
    // does not depend on how the threads interleave. They stop when the main thread is done.
    let stop = AtomicBool::new(false);
    let helper_nodes = std::sync::atomic::AtomicU64::new(0);
    let progress = AtomicUsize::new(0);
    let copies: Vec<GameState> = (0..extra).map(|_| state.clone()).collect();
    let result = std::thread::scope(|scope| {
        for (index, copy) in copies.into_iter().enumerate() {
            let (table, stop, nodes, progress) = (table.clone(), &stop, &helper_nodes, &progress);
            scope.spawn(move || {
                let helper_control = Control {
                    start: control.start,
                    hard_limit: control.hard_limit,
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
                        options,
                        max_depth,
                        &helper_control,
                    ),
                    Relaxed,
                );
            });
        }
        let main_control = Control {
            start: control.start,
            hard_limit: control.hard_limit,
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
    result.map(|(best, mut stats)| {
        stats.nodes_visited += helper_nodes.load(Relaxed);
        (best, stats)
    })
}

fn deepen(
    state: &mut GameState,
    eval: &dyn Evaluator,
    memory: &mut Memory,
    options: &Options,
    max_depth: usize,
    time_limit: f64,
    control: &Control,
) -> Result<(Option<Move>, Stats), String> {
    let estimate = estimators::get(&options.estimator).ok_or_else(|| {
        format!(
            "unknown estimator {:?}; available: {:?}",
            options.estimator,
            estimators::NAMES
        )
    })?;
    let clock = control.clock;
    let start = clock();
    let mut best_move: Option<Move> = None;
    let (mut total_nodes, mut total_prunings, mut deepest_ply) = (0u64, 0u64, 0i32);
    let mut completed_depth = 0usize;
    let mut last_truncated = false;
    let mut history: Vec<estimators::Iteration> = Vec::new();
    let mut iterations: Vec<IterationInfo> = Vec::new();
    let mut stop_reason = StopReason::MaxDepth;
    let mut stop_predicted = 0.0;
    let mut last_value: Option<f64> = None;

    for depth in 1..=max_depth {
        // Clock policy: before an iteration, skip it if the estimate says it will not finish in
        // what is left (the overshoot factor forgives a near miss); never skip before `min_depth`,
        // so the move always comes from a minimum of real search.
        let remaining = time_limit - (clock() - start);
        if depth > options.min_depth && !history.is_empty() {
            let predicted = estimate(&history);
            if predicted > remaining * OVERSHOOT_FACTOR {
                stop_reason = StopReason::Estimate;
                stop_predicted = predicted;
                break;
            }
        }
        if let Some(progress) = control.progress {
            progress.store(depth, Relaxed);
        }
        let iteration_start = clock();
        let result = iteration(
            state,
            eval,
            memory,
            options,
            depth as i32,
            control,
            last_value,
        );
        let seconds = clock() - iteration_start;
        iterations.push(IterationInfo {
            depth,
            seconds,
            nodes: result.nodes,
            truncated: result.truncated,
        });
        total_nodes += result.nodes;
        total_prunings += result.prunings;
        last_truncated = result.truncated;
        deepest_ply = deepest_ply.max(result.max_ply);

        // A cut iteration is discarded: its value comes from a search that stopped half way.
        if result.truncated {
            if best_move.is_none() && result.best.is_some() {
                best_move = result.best; // better a partial answer than none at all
            }
            stop_reason = StopReason::Truncated;
            break;
        }
        history.push((seconds, result.nodes));
        last_value = Some(result.value);
        best_move = result.best;
        completed_depth = depth;
        // The soft budget is also checked after the iteration, since the estimate can be low.
        if depth >= options.min_depth && clock() - start >= time_limit {
            stop_reason = StopReason::Budget;
            break;
        }
    }

    let stats = Stats {
        nodes_visited: total_nodes,
        pruning_count: total_prunings,
        max_depth_reached: deepest_ply,
        truncated: last_truncated && best_move.is_none(),
        completed_depth,
        iterations,
        stop_reason,
        stop_predicted,
    };
    Ok((best_move, stats))
}
