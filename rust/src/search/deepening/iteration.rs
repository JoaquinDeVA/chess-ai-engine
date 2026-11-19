//! The iterative deepening of one thread: depth 1, 2, ... and when to stop.

use std::sync::atomic::Ordering::Relaxed;

use shakmaty::Move;

use super::clock_policy::{next_iteration_fits, soft_scale};
use crate::board::GameState;
use crate::eval::Net;
use crate::search::alpha_beta::alpha_beta;
use crate::search::control::Control;
use crate::search::memory::Memory;
use crate::search::options::*;
use crate::search::stats::*;

pub(super) fn deepen(
    state: &mut GameState,
    net: &Net,
    memory: &mut Memory,
    options: &Options,
    max_depth: usize,
    soft_limit: f64,
    control: &Control,
) -> Result<(Option<Move>, Stats, Option<i32>), String> {
    let clock = control.clock;
    let start = clock();
    let mut best_move: Option<Move> = None;
    let mut total_nodes = 0u64;
    #[cfg(feature = "search-summary")]
    let (mut total_cutoffs, mut seldepth) = (0u64, 0i32);
    let mut completed_depth = 0usize;
    let mut last_truncated = false;
    let mut iterations: Vec<IterationInfo> = Vec::new();
    let mut stop_reason = StopReason::MaxDepth;
    let mut stable_iterations = 0usize;
    let mut last_value: Option<i32> = None;

    for depth in 1..=max_depth {
        if let Some(main_depth) = control.main_depth {
            main_depth.store(depth, Relaxed);
        }
        let iteration_start = clock();
        let result = alpha_beta(state, net, memory, options, depth as i32, control);
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
            total_cutoffs += result.report.cutoffs;
            seldepth = seldepth.max(result.report.seldepth);
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
        stable_iterations = if result.best.is_some() && result.best == best_move {
            stable_iterations + 1
        } else {
            0
        };
        last_value = Some(result.value);
        best_move = result.best;
        completed_depth = depth;
        // Clock policy: never stop before `min_depth`, so the move always comes from a minimum of
        // real search; after it, the soft limit scaled by how settled the best move is.
        let soft = control.scaled_soft_limit(
            soft_limit,
            soft_scale(depth, stable_iterations, result.best_share),
        );
        if depth >= options.min_depth && clock() - start >= soft {
            stop_reason = StopReason::Budget;
            break;
        }
        // An iteration the hard limit would cut, or that would end far past the soft limit, is not
        // started.
        if depth >= options.min_depth
            && !next_iteration_fits(
                clock() - control.start,
                seconds,
                control.current_hard_limit(),
                soft,
            )
        {
            stop_reason = StopReason::Budget;
            break;
        }
    }

    let stats = Stats {
        nodes_visited: total_nodes,
        #[cfg(feature = "search-summary")]
        cutoffs: total_cutoffs,
        #[cfg(feature = "search-summary")]
        seldepth,
        truncated: last_truncated && best_move.is_none(),
        completed_depth,
        iterations,
        stop_reason,
    };
    Ok((best_move, stats, last_value))
}
