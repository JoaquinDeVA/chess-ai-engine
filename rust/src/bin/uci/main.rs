//! Entry point for the UCI engine and its standalone measurement tools.

use std::sync::Arc;

use _core::eval::Net;

use bench::{bench, time_to_depth};
use engine::{load_model, UciEngine};

mod bench;
mod clock;
mod engine;
mod limits;
mod position;
mod search_job;

/// Selects the executable mode from process arguments.
///
/// The executable name is argument 0 and the mode is argument 1:
///
/// ```text
/// uci bench [repetitions] [model.json]
/// uci ttd <depth> <positions> [limit_s]
/// uci
/// ```
///
/// `bench` measures fixed-position search performance, `ttd` measures the time to reach a
/// depth for one FEN per line, and no mode starts the interactive UCI protocol on `stdin`.
fn main() {
    let eval = Arc::new(load_model().expect("the compiled-in model is valid"));
    match std::env::args().nth(1).as_deref() {
        Some("ttd") => run_time_to_depth(&eval),
        Some("bench") => run_bench(&eval),
        _ => UciEngine::new(eval).run(),
    }
}

/// Runs the `ttd` tool.
///
/// Arguments begin at process argument 2 because argument 0 is the executable and argument 1 is
/// `ttd`:
///
/// ```text
/// uci ttd <depth> <positions> [limit_s]
/// ```
///
/// `<depth>` is the target search depth, `<positions>` is a file containing one FEN per line, and
/// `[limit_s]` is an optional per-position time limit in seconds.
fn run_time_to_depth(eval: &Net) {
    let args: Vec<String> = std::env::args().skip(2).collect();
    let depth = args.first().and_then(|depth| depth.parse().ok());
    let limit = args.get(2).map_or(Some(f64::INFINITY), |s| s.parse().ok());
    let result = match (depth, args.get(1), limit) {
        (Some(depth), Some(path), Some(limit)) => time_to_depth(eval, depth, path, limit),
        _ => Err("usage: uci ttd <depth> <positions> [limit_s]".to_string()),
    };
    if let Err(error) = result {
        eprintln!("{error}");
        std::process::exit(1);
    }
}

/// Runs the `bench` tool.
///
/// Arguments begin at process argument 2 because argument 0 is the executable and argument 1 is
/// `bench`:
///
/// ```text
/// uci bench [repetitions] [model.json]
/// ```
///
/// `[repetitions]` defaults to 5. `[model.json]`, when present, replaces the compiled-in network.
fn run_bench(eval: &Net) {
    let repetitions = std::env::args()
        .nth(2)
        .and_then(|n| n.parse().ok())
        .unwrap_or(5);
    match std::env::args().nth(3) {
        Some(path) => bench(
            &std::fs::read_to_string(&path)
                .map_err(|e| e.to_string())
                .and_then(|text| Net::from_json(&text))
                .unwrap_or_else(|e| panic!("bench model {path}: {e}")),
            repetitions,
        ),
        None => bench(eval, repetitions),
    }
}
