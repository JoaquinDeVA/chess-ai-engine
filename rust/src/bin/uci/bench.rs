//! `uci bench` and `uci ttd`: the fixed node count of the search and the time to reach a depth.

use std::time::Instant;

use _core::board::{uci_string, GameState};
use _core::eval::Net;
use _core::search::{
    self, Control, Counters, Memory, Options, Techniques, DEFAULT_HASH_MB, DEFAULT_TECHNIQUES,
};

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
pub fn bench(network: &Net, repetitions: usize) {
    let mut elapsed_seconds: Vec<f64> = Vec::new();
    let mut total_nodes = 0u64;
    for run in 0..repetitions {
        let (nodes, elapsed) = bench_once(network, run == 0);
        total_nodes = nodes; //Son iguales todas las ejecuciones, solo varía el tiempo
        elapsed_seconds.push(elapsed);
    }
    elapsed_seconds.sort_by(f64::total_cmp);
    let median = elapsed_seconds[elapsed_seconds.len() / 2];
    eprintln!(
        "runs {repetitions} min {:.3} s median {median:.3} s nps(median) {:.0}",
        elapsed_seconds[0],
        total_nodes as f64 / median
    );
}

/// Searches the fixed positions one by one with cleared memory and a fixed depth. When
/// `print_positions` is set, the node counts go to stdout (the same on every run). Returns the
/// total nodes and the time of the searches: the table is cleared outside the clock, as a game does
/// it once and not per move (clearing also brings its pages into memory, which a fresh table would
/// do mid-search).
fn bench_once(network: &Net, print_positions: bool) -> (u64, f64) {
    
    let techniques = Techniques::from_names(&DEFAULT_TECHNIQUES).expect("default techniques");
    let search_options = Options {
        techniques,
        ..Options::default()
    };
    let unlimited_clock = || 0.0;
    let search_control = Control {
        start: 0.0,
        hard_limit: f64::INFINITY,
        clock: &unlimited_clock,
        stop: None,
        progress: None,
        rotation: 0,
    };
    let mut memory = Memory::new(1, DEFAULT_HASH_MB);
    let mut elapsed_seconds = 0.0;
    let mut total_nodes = 0u64;
    let mut nodes_by_depth = [0u64; BENCH_DEPTH + 1];

    for position_fen in BENCH_FENS {
        memory.clear();
        memory.counters = Counters::default();
        let mut game_state =
            GameState::from_moves(position_fen, &[]).expect("bench position is valid");

        let search_started = Instant::now();
        let (best_move, stats) = search::iterative_deepening(
            &mut game_state,
            network,
            &mut memory,
            &search_options,
            BENCH_DEPTH,
            f64::INFINITY,
            &search_control,
        )
        .expect("bench search");

        elapsed_seconds += search_started.elapsed().as_secs_f64();
        let best_move_uci = best_move.as_ref().map_or("0000".to_string(), uci_string);
        if print_positions {
            println!(
                "{position_fen} nodes {} bestmove {best_move_uci}",
                stats.nodes_visited
            );
        }
        total_nodes += stats.nodes_visited;
        for depth_stats in &stats.iterations {
            nodes_by_depth[depth_stats.depth] += depth_stats.nodes;
        }
    }

    if print_positions {
        println!("total nodes {total_nodes}");
        for (depth, nodes) in nodes_by_depth.iter().enumerate().skip(1) {
            println!("depth {depth} nodes {nodes}");
        }
    }
    (total_nodes, elapsed_seconds)
}

/// `uci ttd <depth> <positions> [limit_s]`: searches every position of the file (one FEN per line,
/// `#` starts a comment) to `depth` with fresh memory and one thread, and prints the time and the
/// nodes of each and what the search counted over all of them. A position that does not finish
/// within the limit counts as the limit in the median.
pub fn time_to_depth(eval: &Net, depth: usize, path: &str, limit: f64) -> Result<(), String> {
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
    // Techniques to measure, comma separated (`TTD_TECHNIQUES=lmr,futility`); the default set otherwise.
    let names = std::env::var("TTD_TECHNIQUES").unwrap_or_else(|_| DEFAULT_TECHNIQUES.join(","));
    let options = Options {
        techniques: Techniques::from_names(
            &names
                .split(',')
                .filter(|s| !s.is_empty())
                .collect::<Vec<_>>(),
        )?,
        ..Options::default()
    };
    let mut counters = Counters::default();
    let mut seconds: Vec<f64> = Vec::new();
    let (mut total_nodes, mut total_seconds, mut finished) = (0u64, 0.0, 0);
    let mut nodes_by_depth = vec![0u64; depth + 1];
    for (number, fen) in fens.iter().enumerate() {
        let mut memory = Memory::new(1, DEFAULT_HASH_MB);
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
    println!("positions {} finished {finished} depth {depth}", fens.len());
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
