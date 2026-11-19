//! Lazy SMP: the helper threads that fill the shared table while the main thread decides the move.

use std::sync::atomic::Ordering::Relaxed;
use std::sync::{Arc, Mutex};

use shakmaty::Move;

use crate::board::GameState;
use crate::eval::Net;
use crate::search::alpha_beta::alpha_beta;
use crate::search::control::Control;
use crate::search::memory::Memory;
use crate::search::options::*;
use crate::search::ordering::{ContinuationHistory, History, Killers};
use crate::search::transposition_table::Table;

/// How many consecutive depths a helper skips and searches in turn, and where its pattern starts
/// (the first six rows of Stockfish's tables), so that the helpers do not all search the same
/// depths.
const SKIP_SIZE: [usize; 6] = [1, 1, 2, 2, 2, 2];
const SKIP_PHASE: [usize; 6] = [0, 1, 0, 1, 2, 3];

/// The last finished iteration of a thread: depth, value and best move of the root.
pub(super) type Report = Mutex<Option<(usize, i32, Move)>>;

/// The work of a helper thread: its own iterative deepening on its own copy of the position,
/// sharing the table and the history. It never searches shallower than the main thread is, skips
/// some depths by its index, rotates the moves of the root and stops when the flag is raised. It
/// leaves the result of each finished iteration in `report` for the vote. Returns the nodes it
/// visited.
#[allow(clippy::too_many_arguments)]
pub(super) fn helper(
    index: usize,
    mut state: GameState,
    eval: &Net,
    table: Arc<Table>,
    history: History,
    continuation_history: ContinuationHistory,
    options: &Options,
    max_depth: usize,
    control: &Control,
    report: &Report,
) -> u64 {
    let mut memory = Memory {
        table,
        killers: Killers::default(),
        history,
        continuation_history,
        threads: 1,
        diagnostics: Default::default(),
    };
    let (size, phase) = (
        SKIP_SIZE[index % SKIP_SIZE.len()],
        SKIP_PHASE[index % SKIP_PHASE.len()],
    );
    let (mut nodes, mut depth) = (0, 1);
    while !control.stop.is_some_and(|flag| flag.load(Relaxed)) {
        depth = depth.max(control.main_depth.map_or(1, |p| p.load(Relaxed)));
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
