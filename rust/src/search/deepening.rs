//! Iterative deepening: helper threads and the clock policy.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering::Relaxed};
use std::sync::{Arc, Mutex};

use shakmaty::Move;

use super::memory::*;
use super::node::alpha_beta;
use super::options::*;
use super::ordering::{ContinuationHistory, Countermoves, History, Killers};
use super::stats::*;
use super::tt::Table;
use crate::board::GameState;
use crate::eval::Net;

/// Time of an iteration over the time of the one before it, rounded down (measured: median 2.4,
/// 46 % above 2.5, 23 % above 4).
const NEXT_ITERATION_COST: f64 = 3.5;
/// An iteration is not started either when it would end more than this many times past the scaled
/// soft limit: it would spend far more than the move was given (the cut it risks is the hard one).
const SOFT_OVERRUN: f64 = 4.0;
/// First completed depth whose best move scales the soft limit (the first ones are noise).
const SCALE_FROM_DEPTH: usize = 6;
/// Factor of the soft limit by the completed iterations in a row with the same best move: one that
/// has just changed gets more time, one that has held for four or more gets less.
const STABILITY_SCALE: [f64; 5] = [2.5, 1.2, 0.9, 0.8, 0.75];
/// Factor of the soft limit by the share `f` of the nodes spent under the best move:
/// `(NODE_SHARE_PIVOT - f) * NODE_SHARE_SCALE`, from 2.0 (f = 0) to 0.68 (f = 1).
const NODE_SHARE_PIVOT: f64 = 1.5;
const NODE_SHARE_SCALE: f64 = 1.35;

/// Factor of the soft limit after an iteration of `depth` whose best move has held `stable`
/// iterations in a row and took `share` of the nodes.
fn soft_scale(depth: usize, stable: usize, share: f64) -> f64 {
    if depth < SCALE_FROM_DEPTH {
        return 1.0;
    }
    STABILITY_SCALE[stable.min(STABILITY_SCALE.len() - 1)]
        * (NODE_SHARE_PIVOT - share)
        * NODE_SHARE_SCALE
}

/// How many consecutive depths a helper skips and searches in turn, and where its pattern starts
/// (the first six rows of Stockfish's tables), so that the helpers do not all search the same
/// depths.
const SKIP_SIZE: [usize; 6] = [1, 1, 2, 2, 2, 2];
const SKIP_PHASE: [usize; 6] = [0, 1, 0, 1, 2, 3];

/// The last finished iteration of a thread: depth, value and best move of the root.
type Report = Mutex<Option<(usize, i32, Move)>>;

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
    table: Arc<Table>,
    history: History,
    continuation: ContinuationHistory,
    options: &Options,
    max_depth: usize,
    control: &Control,
    report: &Report,
) -> u64 {
    let mut memory = Memory {
        table,
        killers: Killers::default(),
        history,
        continuation,
        countermoves: Countermoves::default(),
        threads: 1,
        diagnostics: Default::default(),
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
fn vote(main: (usize, i32, Option<Move>), reports: &[Report]) -> Option<Move> {
    let (main_depth, main_value, main_move) = main;
    let mut voters: Vec<(usize, i32, Move)> = reports
        .iter()
        .filter_map(|report| *report.lock().unwrap())
        .filter(|(depth, _, _)| *depth >= main_depth)
        .collect();
    let main_move = main_move?;
    voters.insert(0, (main_depth, main_value, main_move));
    let lowest = voters
        .iter()
        .map(|v| v.1)
        .min()
        .expect("the main thread votes");
    let mut weights: Vec<(Move, i64)> = Vec::new();
    for (depth, value, mv) in voters {
        let weight = depth as i64 * i64::from(value - lowest + VOTE_MARGIN);
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

/// Margin of the vote, in units of the score: keeps the thread with the lowest value from weighing
/// zero (about 0.01 of expected score near an even position).
const VOTE_MARGIN: i32 = 7;

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
) -> Result<(Option<Move>, Stats, Option<i32>), String> {
    let clock = control.clock;
    let start = clock();
    let mut best_move: Option<Move> = None;
    let mut total_nodes = 0u64;
    #[cfg(feature = "search-summary")]
    let (mut total_prunings, mut deepest_ply) = (0u64, 0i32);
    let mut completed_depth = 0usize;
    let mut last_truncated = false;
    let mut iterations: Vec<IterationInfo> = Vec::new();
    let mut stop_reason = StopReason::MaxDepth;
    let mut stable = 0usize;
    let mut last_value: Option<i32> = None;

    for depth in 1..=max_depth {
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
        #[cfg(feature = "search-summary")]
        {
            total_prunings += result.report.prunings;
            deepest_ply = deepest_ply.max(result.report.max_ply);
        }
        last_truncated = result.truncated;

        // A cut iteration is discarded: its value comes from a search that stopped half way.
        if result.truncated {
            if best_move.is_none() && result.best.is_some() {
                best_move = result.best; // better a partial answer than none at all
            }
            stop_reason = StopReason::Truncated;
            break;
        }
        stable = if result.best.is_some() && result.best == best_move {
            stable + 1
        } else {
            0
        };
        last_value = Some(result.value);
        best_move = result.best;
        completed_depth = depth;
        // Clock policy: never stop before `min_depth`, so the move always comes from a minimum of
        // real search; after it, the soft limit scaled by how settled the best move is.
        let soft = time_limit * soft_scale(depth, stable, result.best_share);
        if depth >= options.min_depth && clock() - start >= soft {
            stop_reason = StopReason::Budget;
            break;
        }
        // An iteration costs about `NEXT_ITERATION_COST` times the last one. One that the hard limit
        // would cut is not started (its work is thrown away), nor one that would end `SOFT_OVERRUN`
        // times past the soft limit.
        let predicted_end = clock() - control.start + seconds * NEXT_ITERATION_COST;
        if depth >= options.min_depth
            && (predicted_end >= control.hard_limit || predicted_end > soft * SOFT_OVERRUN)
        {
            stop_reason = StopReason::Budget;
            break;
        }
    }

    let stats = Stats {
        nodes_visited: total_nodes,
        #[cfg(feature = "search-summary")]
        pruning_count: total_prunings,
        #[cfg(feature = "search-summary")]
        max_depth_reached: deepest_ply,
        truncated: last_truncated && best_move.is_none(),
        completed_depth,
        iterations,
        stop_reason,
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

    fn report(depth: usize, value: i32, best: Move) -> Report {
        Mutex::new(Some((depth, value, best)))
    }

    #[test]
    fn the_soft_limit_grows_for_a_new_move_and_shrinks_for_a_settled_one() {
        assert_eq!(soft_scale(SCALE_FROM_DEPTH - 1, 0, 0.0), 1.0);
        let fresh = soft_scale(SCALE_FROM_DEPTH, 0, 0.3);
        let settled = soft_scale(SCALE_FROM_DEPTH, 9, 0.9);
        assert!((fresh - 2.5 * 1.2 * 1.35).abs() < 1e-9, "{fresh}");
        assert!(settled < 0.65 && settled < fresh, "{settled}");
    }

    #[test]
    fn the_main_move_wins_when_nobody_else_is_deep_enough() {
        let (a, b) = two_moves();
        let reports = [report(5, 380, b), report(5, 380, b)];
        assert_eq!(vote((6, 0, Some(a)), &reports), Some(a));
    }

    #[test]
    fn deeper_threads_that_agree_outvote_the_main_move() {
        let (a, b) = two_moves();
        let reports = [report(7, 70, b), report(7, 70, b)];
        assert_eq!(vote((6, 70, Some(a)), &reports), Some(b));
    }

    #[test]
    fn a_thread_with_no_iteration_does_not_vote() {
        let (a, _) = two_moves();
        let reports = [Mutex::new(None)];
        assert_eq!(vote((6, 70, Some(a)), &reports), Some(a));
        assert_eq!(vote((0, 70, None), &reports), None);
    }
}
