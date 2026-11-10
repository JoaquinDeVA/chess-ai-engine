//! The engine as a UCI program: `uci`, `isready`, `position`, `go`, `stop`, `setoption`, `quit`.
//!
//! The window and the tournaments play through this binary. The clock spends a share of the
//! time left and does not use the increment (`winc`/`binc` are read and ignored: a known defect,
//! see `docs/`).

use std::io::{self, BufRead};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering::Relaxed};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::Instant;

use _core::board::{ordered_moves, uci_string, GameState};
use _core::eval::sparse_features::FeatureSet;
use _core::eval::sparse_net::{Layer, Net};
use _core::eval::{Eval, Evaluator};
use _core::knowledge::book::OpeningBook;
use _core::knowledge::tablebase::EndgameTablebase;
use _core::search::tt::DEFAULT_MAX_TT_ENTRIES;
use _core::search::{self, Control, Counters, Memory, Options, Techniques, DEFAULT_TECHNIQUES};
use shakmaty::Color;

/// The model compiled in (`rust/models/`, written by `chess_lab.scripts.export_uci_model`).
const MODEL_JSON: &str = include_str!("../../models/net_pst_w32.json");
const MODEL_NAME: &str = "net_pst_w32";
const START_FEN: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
const DEFAULT_HASH_MB: usize = 64;

/// Settings changed with `setoption`.
struct Settings {
    threads: usize,
    hash_mb: usize,
    techniques: Techniques,
    eval: Arc<Eval>,
    /// Opening book (`Book`); off until the option names a file.
    book: Option<OpeningBook>,
    /// Whether the game still follows the book: the first position it lacks ends it.
    in_book: bool,
    /// Endgame tablebases (`Tablebase`); off until the option names a directory.
    tablebase: Option<EndgameTablebase>,
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
    model_from_json(MODEL_JSON)
}

/// A network from the JSON of `export_uci_model`: its `features` and its `layers`, each with
/// `fan_in`, `fan_out`, row-major `weights` and `bias`. Files written before the networks were the
/// only model carry `"perspective": "side_to_move"`; any other perspective is refused.
fn model_from_json(text: &str) -> Result<Eval, String> {
    let value: serde_json::Value = serde_json::from_str(text).map_err(|e| e.to_string())?;
    let features = value["features"]
        .as_str()
        .and_then(FeatureSet::from_name)
        .ok_or("model: unknown features")?;
    if value
        .get("perspective")
        .is_some_and(|p| *p != "side_to_move")
    {
        return Err("model: only the side-to-move perspective is supported".into());
    }
    let layers = value["layers"]
        .as_array()
        .ok_or("model: layers is not a list")?
        .iter()
        .map(|layer| {
            let size = |key: &str| {
                layer[key]
                    .as_u64()
                    .map(|n| n as usize)
                    .ok_or(format!("model: layer has no {key}"))
            };
            Ok(Layer {
                fan_in: size("fan_in")?,
                fan_out: size("fan_out")?,
                weights: number_list(&layer["weights"])?,
                bias: number_list(&layer["bias"])?,
            })
        })
        .collect::<Result<Vec<Layer>, String>>()?;
    Ok(Eval::Net(Net::new(features, layers)?))
}

fn number_list(value: &serde_json::Value) -> Result<Vec<f64>, String> {
    value
        .as_array()
        .ok_or("model: no weights")?
        .iter()
        .map(|w| {
            w.as_f64()
                .ok_or("model: weight is not a number".to_string())
        })
        .collect()
}

/// Soft budget and hard limit of one move in seconds: 15 % of the time left, between 1.5 and 40 s
/// and at most half of it; the hard limit is 1.5 times the budget minus 0.3 s.
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
            _ => {} // winc and binc: ignored (see the module comment)
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

fn run_search(memory: &mut Memory, eval: &Eval, job: Job) -> Result<String, String> {
    let mut state = GameState::from_moves(&job.position.fen, &job.position.moves)?;
    let options = Options {
        techniques: job.techniques,
        node_limit: job.limits.nodes,
        ..Options::default()
    };
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
        "Model" if value == MODEL_NAME => settings.eval = Arc::new(load_model()?),
        "Model" => {
            let text = std::fs::read_to_string(value)
                .map_err(|e| format!("Model is {MODEL_NAME} or a JSON file: {value}: {e}"))?;
            settings.eval = Arc::new(model_from_json(&text)?);
        }
        "Book" => {
            settings.book = match value {
                "" => None,
                path => Some(OpeningBook::load(Path::new(path))?),
            };
            settings.in_book = settings.book.is_some();
        }
        "Tablebase" => {
            settings.tablebase = match value {
                "" => None,
                path => Some(EndgameTablebase::open(Path::new(path))?),
            };
        }
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

/// The move the book or the tablebases give for the position, with the name of its source. The book
/// is followed while it has the position; the first miss ends it until the next game. `None` leaves
/// the move to the search.
fn known_move(settings: &mut Settings, position: &Position) -> Option<(String, &'static str)> {
    let state = GameState::from_moves(&position.fen, &position.moves).ok()?;
    if settings.in_book {
        let found = settings
            .book
            .as_ref()
            .and_then(|book| book.lookup(state.pos()));
        match found {
            Some(m) => return Some((uci_string(&m), "book")),
            None => settings.in_book = false,
        }
    }
    let m = settings.tablebase.as_ref()?.best_move(state.pos())?;
    Some((uci_string(&m), "tablebase"))
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

/// Runs `bench_once` `repetitions` times: the node counts (stdout) come from the first run, the
/// times and the nodes per second (stderr) from all of them, with their median.
fn bench(eval: &Eval, repetitions: usize) {
    let mut seconds: Vec<f64> = Vec::new();
    let mut total_nodes = 0;
    for run in 0..repetitions {
        let (nodes, elapsed) = bench_once(eval, run == 0);
        total_nodes = nodes;
        seconds.push(elapsed);
    }
    seconds.sort_by(f64::total_cmp);
    let median = seconds[seconds.len() / 2];
    eprintln!(
        "runs {repetitions} min {:.3} s median {median:.3} s nps(median) {:.0}",
        seconds[0],
        total_nodes as f64 / median
    );
}

/// Searches the fixed positions one by one with fresh memory and a fixed depth. When `print` is
/// set, the node counts go to stdout (the same on every run). Returns the total nodes and the time.
fn bench_once(eval: &Eval, print: bool) -> (u64, f64) {
    let techniques = Techniques::from_names(&DEFAULT_TECHNIQUES).expect("default techniques");
    let started = Instant::now();
    let mut total_nodes = 0;
    let mut nodes_by_depth = [0u64; BENCH_DEPTH + 1];
    for fen in BENCH_FENS {
        let mut memory = Memory::new(DEFAULT_MAX_TT_ENTRIES);
        let mut state = GameState::from_moves(fen, &[]).expect("bench position is valid");
        let options = Options {
            techniques,
            ..Options::default()
        };
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
        if print {
            println!("{fen} nodes {} bestmove {best}", stats.nodes_visited);
        }
        total_nodes += stats.nodes_visited;
        for iteration in &stats.iterations {
            nodes_by_depth[iteration.depth] += iteration.nodes;
        }
    }
    if print {
        println!("total nodes {total_nodes}");
        for (depth, nodes) in nodes_by_depth.iter().enumerate().skip(1) {
            println!("depth {depth} nodes {nodes}");
        }
    }
    (total_nodes, started.elapsed().as_secs_f64())
}

/// `uci ttd <depth> <positions> [limit_s]`: searches every position of the file (one FEN per line,
/// `#` starts a comment) to `depth` with fresh memory and one thread, and prints the time and the
/// nodes of each and what the search counted over all of them. A position that does not finish
/// within the limit counts as the limit in the median. `TTD_TABLE=shared` searches with the table
/// of several threads (64 MB) instead of the one of a single thread.
fn time_to_depth(eval: &Eval, depth: usize, path: &str, limit: f64) -> Result<(), String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
    let fens: Vec<String> = text
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .map(|line| {
            line.split_whitespace()
                .take(6)
                .collect::<Vec<_>>()
                .join(" ")
        })
        .collect();
    if fens.is_empty() {
        return Err(format!("{path}: no positions"));
    }
    let shared = std::env::var("TTD_TABLE").is_ok_and(|table| table == "shared");
    // Entries of the local table, to measure how its size matters (the bench only).
    let entries = std::env::var("TTD_ENTRIES")
        .ok()
        .and_then(|n| n.parse().ok())
        .unwrap_or(DEFAULT_MAX_TT_ENTRIES);
    let options = Options {
        techniques: Techniques::from_names(&DEFAULT_TECHNIQUES)?,
        ..Options::default()
    };
    let mut counters = Counters::default();
    let mut seconds: Vec<f64> = Vec::new();
    let (mut total_nodes, mut total_seconds, mut finished) = (0u64, 0.0, 0);
    let mut nodes_by_depth = vec![0u64; depth + 1];
    for (number, fen) in fens.iter().enumerate() {
        let mut memory = if shared {
            // The shared table without its helper threads.
            let mut memory = Memory::with_threads(DEFAULT_MAX_TT_ENTRIES, 2, DEFAULT_HASH_MB);
            memory.threads = 1;
            memory
        } else {
            Memory::new(entries)
        };
        memory.counters = counters;
        let mut state = GameState::from_moves(fen, &[])?;
        let origin = Instant::now();
        let clock = || origin.elapsed().as_secs_f64();
        let control = Control {
            start: 0.0,
            hard_limit: limit,
            clock: &clock,
            stop: None,
            progress: None,
            rotation: 0,
        };
        let (_, stats) = search::iterative_deepening(
            &mut state,
            eval,
            &mut memory,
            &options,
            depth,
            f64::INFINITY,
            &control,
        )?;
        let elapsed = clock();
        counters = memory.counters;
        let done = stats.completed_depth >= depth;
        if done {
            finished += 1;
            for iteration in &stats.iterations {
                nodes_by_depth[iteration.depth] += iteration.nodes;
            }
        }
        total_nodes += stats.nodes_visited;
        total_seconds += elapsed;
        seconds.push(if done { elapsed } else { limit });
        let by_iteration: Vec<String> = stats
            .iterations
            .iter()
            .map(|iteration| format!("{}:{}", iteration.depth, iteration.nodes))
            .collect();
        println!(
            "{:2} depth {} nodes {} seconds {elapsed:.2} iterations {}",
            number + 1,
            stats.completed_depth,
            stats.nodes_visited,
            by_iteration.join(" ")
        );
    }
    seconds.sort_by(f64::total_cmp);
    let percent = |part: u64, whole: u64| 100.0 * part as f64 / whole.max(1) as f64;
    println!(
        "positions {} finished {finished} depth {depth} table {}",
        fens.len(),
        if shared { "shared" } else { "local" }
    );
    println!("median seconds {:.2}", seconds[seconds.len() / 2]);
    println!(
        "nodes {total_nodes} nodes/s {:.0}",
        total_nodes as f64 / total_seconds
    );
    if depth >= 3 && nodes_by_depth[depth - 2] > 0 {
        // Over two plies, so that the odd and even iterations do not alternate.
        let growth = nodes_by_depth[depth] as f64 / nodes_by_depth[depth - 2] as f64;
        println!(
            "branching factor {:.2} (depth {} to {depth}, per ply, finished positions)",
            growth.sqrt(),
            depth - 2
        );
    }
    let c = counters;
    println!(
        "quiescence nodes {:.1} %",
        percent(c.quiescence_nodes, total_nodes)
    );
    println!(
        "first-move cuts {:.1} % of {} cuts",
        percent(c.first_move_cuts, c.cuts),
        c.cuts
    );
    println!(
        "killer cuts {:.1} % of cuts, by slot (oldest first) {:.1} % {:.1} % {:.1} %",
        percent(c.killer_cuts.iter().sum(), c.cuts),
        percent(c.killer_cuts[0], c.cuts),
        percent(c.killer_cuts[1], c.cuts),
        percent(c.killer_cuts[2], c.cuts)
    );
    println!(
        "table hits {:.1} % of {} probes, ending the node {:.1} %",
        percent(c.table_hits, c.table_probes),
        c.table_probes,
        percent(c.table_cuts, c.table_probes)
    );
    println!(
        "table stores {} dropping another position {:.1} %, a deeper one {:.1} %",
        c.stores,
        percent(c.evictions, c.stores),
        percent(c.deeper_evictions, c.stores)
    );
    println!(
        "repetition draws {} stores over a repetition {} ({:.3} % of stores)",
        c.repetition_draws,
        c.stores_over_repetition,
        percent(c.stores_over_repetition, c.stores)
    );
    Ok(())
}

fn main() {
    let eval = Arc::new(load_model().expect("the compiled-in model is valid"));
    // uci ttd <depth> <positions> [limit_s]
    if std::env::args().nth(1).as_deref() == Some("ttd") {
        let args: Vec<String> = std::env::args().skip(2).collect();
        let depth = args.first().and_then(|depth| depth.parse().ok());
        let limit = args.get(2).map_or(Some(f64::INFINITY), |s| s.parse().ok());
        let result = match (depth, args.get(1), limit) {
            (Some(depth), Some(path), Some(limit)) => time_to_depth(&eval, depth, path, limit),
            _ => Err("usage: uci ttd <depth> <positions> [limit_s]".to_string()),
        };
        if let Err(error) = result {
            eprintln!("{error}");
            std::process::exit(1);
        }
        return;
    }
    // uci bench [repetitions] [model.json]
    if std::env::args().nth(1).as_deref() == Some("bench") {
        let repetitions = std::env::args()
            .nth(2)
            .and_then(|n| n.parse().ok())
            .unwrap_or(5);
        match std::env::args().nth(3) {
            Some(path) => bench(
                &std::fs::read_to_string(&path)
                    .map_err(|e| e.to_string())
                    .and_then(|text| model_from_json(&text))
                    .unwrap_or_else(|e| panic!("bench model {path}: {e}")),
                repetitions,
            ),
            None => bench(&eval, repetitions),
        }
        return;
    }
    let mut settings = Settings {
        threads: 1,
        hash_mb: DEFAULT_HASH_MB,
        techniques: Techniques::from_names(&DEFAULT_TECHNIQUES).expect("default techniques"),
        eval: eval.clone(),
        book: None,
        in_book: false,
        tablebase: None,
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
                println!("option name Book type string default");
                println!("option name Tablebase type string default");
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
                settings.in_book = settings.book.is_some();
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
            "eval" => match GameState::from_moves(&position.fen, &position.moves) {
                Ok(state) => println!(
                    "info string eval {:.6}",
                    settings.eval.evaluate(state.pos(), Color::White)
                ),
                Err(error) => println!("info string {error}"),
            },
            "position" => match args.first() {
                Some(&"startpos") => position = parse_position(START_FEN.into(), &args[1..]),
                Some(&"fen") if args.len() >= 7 => {
                    position = parse_position(args[1..7].join(" "), &args[7..])
                }
                _ => println!("info string position needs startpos or fen"),
            },
            "go" => {
                finish(&mut running, &mut memory);
                if let Some((best, source)) = known_move(&mut settings, &position) {
                    println!("info string {source}\nbestmove {best}");
                    continue;
                }
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
                let eval = settings.eval.clone();
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

#[cfg(test)]
mod tests {
    use super::*;

    fn net_json(fan_in: usize, extra: &str) -> String {
        let first = format!(
            r#"{{"fan_in":{fan_in},"fan_out":2,"weights":{:?},"bias":[0.5,0.5]}}"#,
            vec![0.0; fan_in * 2]
        );
        let last = r#"{"fan_in":2,"fan_out":1,"weights":[1.0,1.0],"bias":[0.0]}"#;
        format!(r#"{{"features":"pst_slotted",{extra}"layers":[{first},{last}]}}"#)
    }

    #[test]
    fn a_network_file_is_loaded_with_or_without_the_old_perspective_field() {
        assert!(matches!(
            model_from_json(&net_json(768, "")),
            Ok(Eval::Net(_))
        ));
        let old = r#""perspective":"side_to_move","#;
        assert!(matches!(
            model_from_json(&net_json(768, old)),
            Ok(Eval::Net(_))
        ));
        let white = r#""perspective":"white","#;
        assert!(model_from_json(&net_json(768, white)).is_err());
    }

    #[test]
    fn a_network_with_the_wrong_input_size_is_rejected() {
        assert!(model_from_json(&net_json(10, "")).is_err());
    }

    #[test]
    fn the_compiled_in_model_is_a_network() {
        assert!(matches!(load_model(), Ok(Eval::Net(_))));
    }

    fn settings_with_book() -> Settings {
        let start = Position {
            fen: START_FEN.into(),
            moves: vec![],
        };
        let key = _core::knowledge::book::position_key(&start.fen);
        Settings {
            threads: 1,
            hash_mb: 1,
            techniques: Techniques::from_names(&DEFAULT_TECHNIQUES).unwrap(),
            eval: Arc::new(load_model().unwrap()),
            book: Some(OpeningBook::from_entries([(key, "e2e4".to_string())])),
            in_book: true,
            tablebase: None,
        }
    }

    #[test]
    fn the_book_answers_until_the_first_miss() {
        let mut settings = settings_with_book();
        let start = Position {
            fen: START_FEN.into(),
            moves: vec![],
        };
        assert_eq!(
            known_move(&mut settings, &start),
            Some(("e2e4".to_string(), "book"))
        );
        let off_book = Position {
            fen: START_FEN.into(),
            moves: vec!["a2a3".into()],
        };
        assert_eq!(known_move(&mut settings, &off_book), None);
        // After the miss the book stays closed, even for a position it has.
        assert!(!settings.in_book);
        assert_eq!(known_move(&mut settings, &start), None);
    }

    #[test]
    fn without_book_or_tables_the_search_decides() {
        let mut settings = settings_with_book();
        settings.book = None;
        let start = Position {
            fen: START_FEN.into(),
            moves: vec![],
        };
        assert_eq!(known_move(&mut settings, &start), None);
    }
}
