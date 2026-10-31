//! The engine as a UCI program: `uci`, `isready`, `position`, `go`, `stop`, `setoption`, `quit`.
//!
//! The search is the one the Python package calls through `_core`. The clock is the port of the
//! Python `TimeManager`, which does not use the increment (`winc`/`binc` are read and ignored: a
//! known defect, see `docs/`).

use std::io::{self, BufRead};
use std::sync::atomic::{AtomicBool, Ordering::Relaxed};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::Instant;

use _core::board::{ordered_moves, uci_string, GameState};
use _core::eval::features::{FeatureSet, Perspective};
use _core::eval::linear::LinearModel;
use _core::eval::Eval;
use _core::search::tt::DEFAULT_MAX_TT_ENTRIES;
use _core::search::{self, Control, Memory, Options, Techniques};
use shakmaty::Color;

/// The model compiled in (`rust/models/`, written by `chess_lab.scripts.export_uci_model`).
const MODEL_JSON: &str = include_str!("../../models/linear_pst_terms.json");
const MODEL_NAME: &str = "linear_pst_terms";
const START_FEN: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
/// The techniques the Python search uses by default.
const DEFAULT_TECHNIQUES: [&str; 6] = [
    "incremental",
    "tt_move",
    "history",
    "pvs",
    "aspiration",
    "null_move",
];
const DEFAULT_HASH_MB: usize = 64;

/// Settings changed with `setoption`.
struct Settings {
    threads: usize,
    hash_mb: usize,
    techniques: Techniques,
}

/// The position as the GUI sent it: the root FEN and the moves played from it.
#[derive(Clone)]
struct Position {
    fen: String,
    moves: Vec<String>,
}

/// What one `go` asks for. Times are in seconds.
struct Limits {
    depth: usize,
    nodes: Option<u64>,
    time_limit: f64,
    hard_limit: f64,
}

/// Everything a search thread needs, moved into it.
struct Job {
    position: Position,
    limits: Limits,
    techniques: Techniques,
    stop: Arc<AtomicBool>,
}

/// A search running in its own thread. It gives the memory back when it ends.
type Running = (JoinHandle<Memory>, Arc<AtomicBool>);

fn load_model() -> Result<Eval, String> {
    let value: serde_json::Value = serde_json::from_str(MODEL_JSON).map_err(|e| e.to_string())?;
    let features = value["features"]
        .as_str()
        .and_then(FeatureSet::from_name)
        .ok_or("model: unknown features")?;
    let perspective = match value["perspective"].as_str() {
        Some("white") => Perspective::White,
        Some("side_to_move") => Perspective::SideToMove,
        _ => return Err("model: unknown perspective".into()),
    };
    let weights = value["weights"]
        .as_array()
        .ok_or("model: no weights")?
        .iter()
        .map(|w| w.as_f64().ok_or("model: weight is not a number"))
        .collect::<Result<Vec<f64>, _>>()?;
    let bias = value["bias"].as_f64().ok_or("model: no bias")?;
    Ok(Eval::Linear(LinearModel::new(
        features,
        perspective,
        weights,
        bias,
    )?))
}

/// Soft budget and hard limit of one move in seconds, as the Python `TimeManager` computes them.
fn clock_budget(time_left: f64) -> (f64, f64) {
    let budget = (time_left * 0.15).clamp(1.5, 40.0).min(time_left * 0.5);
    (budget, budget * 1.5 - 0.3)
}

/// The limits of a `go` command. `movetime` wins over the clock; with neither the search runs to
/// the maximum depth.
fn go_limits(args: &[&str], white_to_move: bool) -> Limits {
    let (mut depth, mut nodes) = (None, None);
    let (mut movetime, mut wtime, mut btime): (Option<f64>, Option<f64>, Option<f64>) =
        (None, None, None);
    let mut words = args.iter().copied();
    while let Some(key) = words.next() {
        if key == "infinite" {
            continue;
        }
        let Some(value) = words.next() else { break };
        match key {
            "depth" => depth = value.parse().ok(),
            "nodes" => nodes = value.parse().ok(),
            "movetime" => movetime = value.parse::<f64>().ok().map(|ms| ms / 1000.0),
            "wtime" => wtime = value.parse::<f64>().ok().map(|ms| ms / 1000.0),
            "btime" => btime = value.parse::<f64>().ok().map(|ms| ms / 1000.0),
            _ => {} // winc and binc: ignored, as the Python TimeManager does
        }
    }
    let (time_limit, hard_limit) = match (movetime, if white_to_move { wtime } else { btime }) {
        (Some(seconds), _) => (seconds, seconds),
        (None, Some(time_left)) => clock_budget(time_left),
        (None, None) => (f64::INFINITY, f64::INFINITY),
    };
    Limits {
        depth: depth.unwrap_or(search::MAX_DEPTH).max(1),
        nodes,
        time_limit,
        hard_limit,
    }
}

/// Searches the job and prints `info` and `bestmove`. A broken position prints an error and
/// `bestmove 0000`.
fn think(mut memory: Memory, eval: Arc<Eval>, job: Job) -> Memory {
    match run_search(&mut memory, &eval, job) {
        Ok(best) => println!("bestmove {best}"),
        Err(error) => println!("info string {error}\nbestmove 0000"),
    }
    memory
}

/// The options every search of the binary uses (the defaults of the Python search).
fn search_options(techniques: Techniques, node_limit: Option<u64>) -> Options {
    Options {
        quiescence_plies: search::DEFAULT_QUIESCENCE_PLIES,
        min_depth: search::DEFAULT_MIN_DEPTH,
        estimator: search::DEFAULT_ESTIMATOR.to_string(),
        techniques,
        node_limit,
        ..Options::default()
    }
}

fn run_search(memory: &mut Memory, eval: &Eval, job: Job) -> Result<String, String> {
    let mut state = GameState::from_moves(&job.position.fen, &job.position.moves)?;
    let options = search_options(job.techniques, job.limits.nodes);
    let origin = Instant::now();
    let clock = || origin.elapsed().as_secs_f64();
    let control = Control {
        start: 0.0,
        hard_limit: job.limits.hard_limit,
        clock: &clock,
        stop: Some(&job.stop),
        progress: None,
        rotation: 0,
    };
    let (best, stats) = search::iterative_deepening(
        &mut state,
        eval,
        memory,
        &options,
        job.limits.depth,
        job.limits.time_limit,
        &control,
    )?;
    println!(
        "info depth {} nodes {} time {}",
        stats.completed_depth,
        stats.nodes_visited,
        (clock() * 1000.0) as u64
    );
    Ok(match best {
        Some(m) => uci_string(&m),
        // A search cut before its first move still owes a legal move.
        None => ordered_moves(state.pos())
            .first()
            .map_or("0000".to_string(), uci_string),
    })
}

/// Waits for a running search and takes its memory back.
fn finish(running: &mut Option<Running>, memory: &mut Option<Memory>) {
    if let Some((handle, _)) = running.take() {
        *memory = Some(handle.join().expect("search thread panicked"));
    }
}

fn new_memory(settings: &Settings) -> Memory {
    Memory::with_threads(DEFAULT_MAX_TT_ENTRIES, settings.threads, settings.hash_mb)
}

fn set_option(settings: &mut Settings, args: &[&str]) -> Result<bool, String> {
    let text = args.join(" ");
    let (name, value) = match text.split_once(" value ") {
        Some((name, value)) => (name, value.trim()),
        None => (text.as_str(), ""),
    };
    let name = name.strip_prefix("name ").unwrap_or(name).trim();
    match name {
        "Hash" => settings.hash_mb = value.parse().map_err(|_| "Hash needs a number")?,
        "Threads" => {
            settings.threads = value
                .parse::<usize>()
                .map_err(|_| "Threads needs a number")?
                .max(1)
        }
        "Model" if value != MODEL_NAME => return Err(format!("only {MODEL_NAME} is compiled in")),
        "Model" => {}
        "Techniques" => {
            settings.techniques = Techniques::from_names(
                &value
                    .split(',')
                    .filter(|s| !s.is_empty())
                    .collect::<Vec<_>>(),
            )?
        }
        other => return Err(format!("unknown option {other:?}")),
    }
    Ok(name == "Hash" || name == "Threads")
}

/// Positions of `bench`: a fixed set, so that the node counts are a fingerprint of the search.
const BENCH_FENS: [&str; 6] = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
    "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
];
const BENCH_DEPTH: usize = 6;

/// Searches the fixed positions one by one with fresh memory and a fixed depth. The node counts go
/// to stdout (the same on every run); the time to stderr.
fn bench(eval: &Eval) {
    let techniques = Techniques::from_names(&DEFAULT_TECHNIQUES).expect("default techniques");
    let started = Instant::now();
    let mut total_nodes = 0;
    let mut nodes_by_depth = [0u64; BENCH_DEPTH + 1];
    for fen in BENCH_FENS {
        let mut memory = Memory::new(DEFAULT_MAX_TT_ENTRIES);
        let mut state = GameState::from_moves(fen, &[]).expect("bench position is valid");
        let options = search_options(techniques, None);
        let clock = || 0.0;
        let control = Control {
            start: 0.0,
            hard_limit: f64::INFINITY,
            clock: &clock,
            stop: None,
            progress: None,
            rotation: 0,
        };
        let (best, stats) = search::iterative_deepening(
            &mut state,
            eval,
            &mut memory,
            &options,
            BENCH_DEPTH,
            f64::INFINITY,
            &control,
        )
        .expect("bench search");
        let best = best.as_ref().map_or("0000".to_string(), uci_string);
        println!("{fen} nodes {} bestmove {best}", stats.nodes_visited);
        total_nodes += stats.nodes_visited;
        for iteration in &stats.iterations {
            nodes_by_depth[iteration.depth] += iteration.nodes;
        }
    }
    println!("total nodes {total_nodes}");
    for (depth, nodes) in nodes_by_depth.iter().enumerate().skip(1) {
        println!("depth {depth} nodes {nodes}");
    }
    eprintln!("time {:.3} s", started.elapsed().as_secs_f64());
}

fn main() {
    let eval = Arc::new(load_model().expect("the compiled-in model is valid"));
    if std::env::args().nth(1).as_deref() == Some("bench") {
        bench(&eval);
        return;
    }
    let mut settings = Settings {
        threads: 1,
        hash_mb: DEFAULT_HASH_MB,
        techniques: Techniques::from_names(&DEFAULT_TECHNIQUES).expect("default techniques"),
    };
    let mut memory = Some(new_memory(&settings));
    let mut position = Position {
        fen: START_FEN.to_string(),
        moves: Vec::new(),
    };
    let mut running: Option<Running> = None;

    for line in io::stdin().lock().lines() {
        let line = line.expect("stdin is readable");
        let words: Vec<&str> = line.split_whitespace().collect();
        let Some((&command, args)) = words.split_first() else {
            continue;
        };
        match command {
            "uci" => {
                println!("id name chess_core");
                println!("option name Hash type spin default {DEFAULT_HASH_MB} min 1 max 65536");
                println!("option name Threads type spin default 1 min 1 max 64");
                println!("option name Model type string default {MODEL_NAME}");
                println!(
                    "option name Techniques type string default {}",
                    DEFAULT_TECHNIQUES.join(",")
                );
                println!("uciok");
            }
            "isready" => println!("readyok"),
            "ucinewgame" => {
                finish(&mut running, &mut memory);
                if let Some(memory) = memory.as_mut() {
                    memory.clear();
                }
            }
            "setoption" => {
                finish(&mut running, &mut memory);
                match set_option(&mut settings, args) {
                    Ok(true) => memory = Some(new_memory(&settings)),
                    Ok(false) => {}
                    Err(error) => println!("info string {error}"),
                }
            }
            "position" => match args.first() {
                Some(&"startpos") => position = parse_position(START_FEN.into(), &args[1..]),
                Some(&"fen") if args.len() >= 7 => {
                    position = parse_position(args[1..7].join(" "), &args[7..])
                }
                _ => println!("info string position needs startpos or fen"),
            },
            "go" => {
                finish(&mut running, &mut memory);
                let white = GameState::from_moves(&position.fen, &position.moves)
                    .map_or(true, |state| state.turn() == Color::White);
                let job = Job {
                    position: position.clone(),
                    limits: go_limits(args, white),
                    techniques: settings.techniques,
                    stop: Arc::new(AtomicBool::new(false)),
                };
                let stop = job.stop.clone();
                let memory_for_thread = memory.take().expect("memory is back after finish");
                let eval = eval.clone();
                let handle = std::thread::spawn(move || think(memory_for_thread, eval, job));
                running = Some((handle, stop));
            }
            "stop" => {
                if let Some((_, stop)) = &running {
                    stop.store(true, Relaxed);
                }
            }
            "quit" => {
                if let Some((_, stop)) = &running {
                    stop.store(true, Relaxed);
                }
                finish(&mut running, &mut memory);
                return;
            }
            _ => {}
        }
    }
    // End of input: the pending search still answers, so piped scripts get their `bestmove`.
    finish(&mut running, &mut memory);
}

/// The position with the moves that follow the optional `moves` keyword.
fn parse_position(fen: String, rest: &[&str]) -> Position {
    let moves = rest
        .iter()
        .skip_while(|word| **word != "moves")
        .skip(1)
        .map(|word| word.to_string())
        .collect();
    Position { fen, moves }
}
