//! One search running on its own thread: the signals the UCI thread sends it (`stop`, `ponderhit`,
//! the time limits) and the `bestmove` it prints when it ends.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering::Relaxed, Ordering::Release};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use _core::board::{move_id, ordered_moves, uci_string};
use _core::eval::Net;
use _core::search::{self, Control, Memory, Options};
use shakmaty::Move;

use crate::limits::SearchLimits;
use crate::position::Position;

/// What the UCI thread and the search thread share. Limits are seconds since `origin`, stored as
/// the bits of an `f64`; infinity until a pondering search gets its `ponderhit`.
struct Signals {
    stop: AtomicBool,
    /// Set by the first `ponderhit`; a pondering search holds its `bestmove` until it or `stop`.
    ponder_hit: AtomicBool,
    soft_limit: AtomicU64,
    hard_limit: AtomicU64,
    /// Second (since `origin`) at which the real budget began: 0, or the first `ponderhit`.
    budget_start: AtomicU64,
    origin: Instant,
}

struct SearchJob {
    position: Position,
    limits: SearchLimits,
    root_moves: Option<Vec<u32>>,
    signals: Arc<Signals>,
}

struct SearchResult {
    best: String,
    ponder: Option<String>,
}

/// A search in progress: the thread and what is needed to steer it from the UCI thread.
pub struct RunningSearch {
    handle: JoinHandle<Memory>,
    signals: Arc<Signals>,
    ponder: bool,
    resume_soft_time: f64,
    resume_hard_time: f64,
}

impl RunningSearch {
    pub fn start(
        memory: Memory,
        eval: Arc<Net>,
        position: Position,
        limits: SearchLimits,
        root_moves: Option<Vec<u32>>,
    ) -> Self {
        let signals = Arc::new(Signals {
            stop: AtomicBool::new(false),
            ponder_hit: AtomicBool::new(false),
            soft_limit: AtomicU64::new(limits.soft_time.to_bits()),
            hard_limit: AtomicU64::new(limits.hard_time.to_bits()),
            budget_start: AtomicU64::new(0.0_f64.to_bits()),
            origin: Instant::now(),
        });
        let (ponder, resume_soft_time, resume_hard_time) = (
            limits.ponder,
            limits.resume_soft_time,
            limits.resume_hard_time,
        );
        let job = SearchJob {
            position,
            limits,
            root_moves,
            signals: signals.clone(),
        };
        Self {
            handle: std::thread::spawn(move || search_job(memory, eval, job)),
            signals,
            ponder,
            resume_soft_time,
            resume_hard_time,
        }
    }

    pub fn stop(&self) {
        self.signals.stop.store(true, Relaxed);
    }

    /// The pondered move was played: the real time budget starts counting now, in the same search.
    pub fn ponder_hit(&self) {
        // Only the first `ponderhit` counts: a repeated one must not extend the budget.
        if self.ponder && !self.signals.ponder_hit.swap(true, Relaxed) {
            let elapsed = self.signals.origin.elapsed().as_secs_f64();
            self.signals.budget_start.store(elapsed.to_bits(), Release);
            self.signals
                .soft_limit
                .store((elapsed + self.resume_soft_time).to_bits(), Release);
            self.signals
                .hard_limit
                .store((elapsed + self.resume_hard_time).to_bits(), Release);
        }
    }

    /// Waits for the thread and returns the memory of the game.
    pub fn finish(self) -> Memory {
        self.handle.join().expect("search thread panicked")
    }
}

fn search_job(mut memory: Memory, eval: Arc<Net>, job: SearchJob) -> Memory {
    let (ponder, signals) = (job.limits.ponder, job.signals.clone());
    let outcome = run_search(&mut memory, &eval, job);
    // UCI: a search started with `go ponder` must not print `bestmove` before `ponderhit` or `stop`,
    // even if it ended by itself (maximum depth, node limit, forced mate).
    while ponder && !signals.stop.load(Relaxed) && !signals.ponder_hit.load(Relaxed) {
        std::thread::sleep(Duration::from_millis(1));
    }
    match outcome {
        Ok(result) => match result.ponder {
            Some(ponder) => println!("bestmove {} ponder {ponder}", result.best),
            None => println!("bestmove {}", result.best),
        },
        Err(error) => println!("info string {error}\nbestmove 0000"),
    }
    memory
}

fn run_search(memory: &mut Memory, eval: &Net, job: SearchJob) -> Result<SearchResult, String> {
    let mut state = job.position.state()?;
    let signals = &job.signals;
    let options = Options {
        node_limit: job.limits.node_limit,
        root_moves: job.root_moves,
        ..Options::default()
    };
    let clock = || signals.origin.elapsed().as_secs_f64();
    let control = Control {
        start: 0.0,
        hard_limit: job.limits.hard_time,
        dynamic_hard_limit: Some(&signals.hard_limit),
        dynamic_soft_limit: Some(&signals.soft_limit),
        dynamic_budget_start: Some(&signals.budget_start),
        clock: &clock,
        stop: Some(&signals.stop),
        progress: None,
        rotation: 0,
    };
    let (best, stats) = search::iterative_deepening(
        &mut state,
        eval,
        memory,
        &options,
        job.limits.max_depth,
        job.limits.soft_time,
        &control,
    )?;
    println!(
        "info depth {} nodes {} time {}",
        stats.completed_depth,
        stats.nodes_visited,
        (clock() * 1000.0) as u64
    );
    let best_name = best.as_ref().map(uci_string).unwrap_or_else(|| {
        ordered_moves(state.pos())
            .first()
            .map_or("0000".to_string(), uci_string)
    });
    let ponder = match &best {
        Some(mv) if !job.limits.ponder => ponder_reply(&mut state, memory, mv),
        _ => None,
    };
    Ok(SearchResult {
        best: best_name,
        ponder,
    })
}

/// The reply the transposition table keeps for the position after `best`, if it is legal there.
fn ponder_reply(
    state: &mut _core::board::GameState,
    memory: &Memory,
    best: &Move,
) -> Option<String> {
    state.apply(*best);
    let reply = memory
        .table
        .get(&state.key())
        .and_then(|entry| entry.best)
        .and_then(|id| {
            ordered_moves(state.pos())
                .into_iter()
                .find(|mv| move_id(mv) == id)
        })
        .map(|mv| uci_string(&mv));
    state.undo();
    reply
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{load_model, START_FEN};
    use crate::limits::parse_go;
    use crate::position;

    fn start(go: &[&str]) -> RunningSearch {
        RunningSearch::start(
            Memory::new(1, 1),
            Arc::new(load_model().unwrap()),
            position::parse(START_FEN.into(), &[]),
            parse_go(go, true, 1),
            None,
        )
    }

    /// A pondering search that ends by itself holds its `bestmove` until `ponderhit` or `stop`.
    #[test]
    fn a_finished_ponder_search_waits_for_its_release() {
        for release in ["ponderhit", "stop"] {
            let search = start(&["ponder", "depth", "1"]);
            std::thread::sleep(Duration::from_millis(300));
            assert!(
                !search.handle.is_finished(),
                "{release}: bestmove came out early"
            );
            match release {
                "ponderhit" => search.ponder_hit(),
                _ => search.stop(),
            }
            search.finish();
        }
    }

    /// Only the first `ponderhit` sets the budget; a second one leaves it as it was.
    #[test]
    fn a_second_ponderhit_does_not_move_the_limits() {
        let search = start(&["ponder", "movetime", "500"]);
        let limits = || {
            (
                search.signals.soft_limit.load(Relaxed),
                search.signals.hard_limit.load(Relaxed),
                search.signals.budget_start.load(Relaxed),
            )
        };
        assert!(f64::from_bits(limits().0).is_infinite());
        search.ponder_hit();
        let first = limits();
        assert!(f64::from_bits(first.0).is_finite());
        std::thread::sleep(Duration::from_millis(20));
        search.ponder_hit();
        assert_eq!(first, limits());
        search.stop();
        search.finish();
    }
}
