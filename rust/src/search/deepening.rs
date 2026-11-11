//! Iterative deepening: helper threads and the clock policy.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering::Relaxed};
use std::sync::{Arc, Mutex};

use shakmaty::Move;

use super::iteration_time;
use super::memory::*;
use super::node::alpha_beta;
use super::options::*;
use super::ordering::{History, Killers};
use super::shared_tt::SharedTable;
use super::stats::*;
use super::tt::Tt;
use crate::board::GameState;
use crate::eval::Net;

/// Tolerance for finishing a depth that was started within the budget.
const OVERSHOOT_FACTOR: f64 = 1.15;

/// How many consecutive depths a helper skips and searches in turn, and where its pattern starts
/// (the first six rows of Stockfish's tables), so that the helpers do not all search the same
/// depths.
const SKIP_SIZE: [usize; 6] = [1, 1, 2, 2, 2, 2];
const SKIP_PHASE: [usize; 6] = [0, 1, 0, 1, 2, 3];

/// The last finished iteration of a thread: depth, value and best move of the root.
type Report = Mutex<Option<(usize, f64, Move)>>;

/// The work of a helper thread: its own iterative deepening on its own copy of the position,
/// sharing the table and the history. It never searches shallower than the main thread is, skips
/// some depths by its index, rotates the moves of the root and stops when the flag is raised. It
/// leaves the result of each finished iteration in `report` for the vote. Returns the nodes it
/// visited.
#[allow(clippy::too_many_arguments)]
fn helper(
    index: usize,
    mut state: GameState,
    eval: &Net,
    table: Arc<SharedTable>,
    history: History,
    options: &Options,
    max_depth: usize,
    control: &Control,
    report: &Report,
) -> u64 {
    let mut memory = Memory {
        table: Tt::Shared(table),
        killers: Killers::default(),
        history,
        threads: 1,
        counters: Counters::default(),
    };
    let (size, phase) = (
        SKIP_SIZE[index % SKIP_SIZE.len()],
        SKIP_PHASE[index % SKIP_PHASE.len()],
    );
    let (mut nodes, mut depth) = (0, 1);
    while !control.stop.is_some_and(|flag| flag.load(Relaxed)) {
        depth = depth.max(control.progress.map_or(1, |p| p.load(Relaxed)));
        if depth > max_depth {
            break;
        }
        if (depth + phase) / size % 2 == 1 {
            depth += 1;
            continue;
        }
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
        if let Some(best) = result.best {
            *report.lock().unwrap() = Some((depth, result.value, best));
        }
        depth += 1;
    }
    nodes
}

/// The move the threads vote for. Each thread with a finished iteration at least as deep as the
/// main thread's gives its move a weight of its depth times how far its value is above the lowest
/// of the voters plus a small margin, so that deeper and better-valued threads count more; the
/// move with most weight wins and the main thread's move wins a tie.
fn vote(main: (usize, f64, Option<Move>), reports: &[Report]) -> Option<Move> {
    let (main_depth, main_value, main_move) = main;
    let mut voters: Vec<(usize, f64, Move)> = reports
        .iter()
        .filter_map(|report| *report.lock().unwrap())
        .filter(|(depth, _, _)| *depth >= main_depth)
        .collect();
    let main_move = main_move?;
    voters.insert(0, (main_depth, main_value, main_move));
    let lowest = voters.iter().map(|v| v.1).fold(f64::INFINITY, f64::min);
    let mut weights: Vec<(Move, f64)> = Vec::new();
    for (depth, value, mv) in voters {
        let weight = depth as f64 * (value - lowest + VOTE_MARGIN);
        match weights.iter_mut().find(|(m, _)| *m == mv) {
            Some((_, total)) => *total += weight,
            None => weights.push((mv, weight)),
        }
    }
    let mut best = 0;
    for (index, (_, weight)) in weights.iter().enumerate() {
        if *weight > weights[best].1 {
            best = index;
        }
    }
    Some(weights.swap_remove(best).0)
}

/// Margin of the vote, in expected score: keeps the thread with the lowest value from weighing zero.
const VOTE_MARGIN: f64 = 0.01;

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
    eval: &Net,
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
        return deepen(state, eval, memory, options, max_depth, time_limit, control)
            .map(|(best, stats, _)| (best, stats));
    };
    table.new_search();
    // Helpers only fill the shared table; the main thread alone decides the move, so the result
    // does not depend on how the threads interleave. They stop when the main thread is done.
    let stop = AtomicBool::new(false);
    let helper_nodes = std::sync::atomic::AtomicU64::new(0);
    let progress = AtomicUsize::new(0);
    let copies: Vec<GameState> = (0..extra).map(|_| state.clone()).collect();
    let reports: Vec<Report> = (0..extra).map(|_| Mutex::new(None)).collect();
    let history = memory.history.clone();
    let result = std::thread::scope(|scope| {
        for (index, copy) in copies.into_iter().enumerate() {
            let (table, stop, nodes, progress) = (table.clone(), &stop, &helper_nodes, &progress);
            let (history, report) = (history.clone(), &reports[index]);
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
                        history,
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

fn deepen(
    state: &mut GameState,
    eval: &Net,
    memory: &mut Memory,
    options: &Options,
    max_depth: usize,
    time_limit: f64,
    control: &Control,
) -> Result<(Option<Move>, Stats, Option<f64>), String> {
    let estimate = options.estimator;
    let clock = control.clock;
    let start = clock();
    let mut best_move: Option<Move> = None;
    let (mut total_nodes, mut total_prunings, mut deepest_ply) = (0u64, 0u64, 0i32);
    let mut completed_depth = 0usize;
    let mut last_truncated = false;
    let mut history: Vec<iteration_time::Iteration> = Vec::new();
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
        let result = alpha_beta(state, eval, memory, options, depth as i32, control);
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
    Ok((best_move, stats, last_value))
}

#[cfg(test)]
mod vote_tests {
    use shakmaty::Position;

    use super::*;
    use crate::board::position_from_fen;

    fn two_moves() -> (Move, Move) {
        let start =
            position_from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1").unwrap();
        let moves = start.legal_moves();
        (moves[0], moves[1])
    }

    fn report(depth: usize, value: f64, best: Move) -> Report {
        Mutex::new(Some((depth, value, best)))
    }

    #[test]
    fn the_main_move_wins_when_nobody_else_is_deep_enough() {
        let (a, b) = two_moves();
        let reports = [report(5, 0.9, b), report(5, 0.9, b)];
        assert_eq!(vote((6, 0.5, Some(a)), &reports), Some(a));
    }

    #[test]
    fn deeper_threads_that_agree_outvote_the_main_move() {
        let (a, b) = two_moves();
        let reports = [report(7, 0.6, b), report(7, 0.6, b)];
        assert_eq!(vote((6, 0.6, Some(a)), &reports), Some(b));
    }

    #[test]
    fn a_thread_with_no_iteration_does_not_vote() {
        let (a, _) = two_moves();
        let reports = [Mutex::new(None)];
        assert_eq!(vote((6, 0.6, Some(a)), &reports), Some(a));
        assert_eq!(vote((0, 0.6, None), &reports), None);
    }
}
