use std::io::{self, BufRead};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering::Relaxed};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::Instant;

use _core::board::{move_id, ordered_moves, uci_string};
use _core::eval::Net;
use _core::knowledge::book::OpeningBook;
use _core::knowledge::tablebase::EndgameTablebase;
use _core::search::{self, Control, Memory, Options, DEFAULT_HASH_MB};
use shakmaty::Color;

use crate::limits::{parse_go, SearchLimits};
use crate::position::{self, Position};

pub const MODEL_NAME: &str = "net_pst_w64";
const MODEL_JSON: &str = include_str!("../../../models/net_pst_w64.json");
const START_FEN: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

/// Settings changed with `setoption`.
struct EngineSettings {
    threads: usize,
    hash_mb: usize,
    eval: Arc<Net>,
    book: Option<OpeningBook>,
    in_book: bool,
    tablebase: Option<EndgameTablebase>,
}

struct SearchJob {
    position: Position,
    limits: SearchLimits,
    stop: Arc<AtomicBool>,
    root_moves: Option<Vec<u32>>,
}

type RunningSearch = (JoinHandle<Memory>, Arc<AtomicBool>);

pub fn load_model() -> Result<Net, String> {
    Net::from_json(MODEL_JSON)
}

pub struct UciEngine {
    settings: EngineSettings,
    memory: Option<Memory>,
    position: Position,
    running: Option<RunningSearch>,
}

impl UciEngine {
    pub fn new(eval: Arc<Net>) -> Self {
        let settings = EngineSettings {
            threads: 1,
            hash_mb: DEFAULT_HASH_MB,
            eval,
            book: None,
            in_book: false,
            tablebase: None,
        };
        let memory = Some(Memory::new(settings.threads, settings.hash_mb));
        Self {
            settings,
            memory,
            position: Position {
                fen: START_FEN.to_string(),
                moves: Vec::new(),
            },
            running: None,
        }
    }

    pub fn run(mut self) {
        for line in io::stdin().lock().lines() {
            let line = line.expect("stdin is readable");
            let words: Vec<&str> = line.split_whitespace().collect();
            let Some((&command, args)) = words.split_first() else {
                continue;
            };
            if self.handle_command(command, args) {
                return;
            }
        }
        self.finish_search();
    }

    fn handle_command(&mut self, command: &str, args: &[&str]) -> bool {
        match command {
            "uci" => self.identify(),
            "isready" => println!("readyok"),
            "ucinewgame" => self.new_game(),
            "setoption" => self.set_option(args),
            "eval" => self.print_evaluation(),
            "position" => self.set_position(args),
            "go" => self.start_search(args),
            "stop" => self.stop_search(),
            "quit" => {
                self.stop_search();
                self.finish_search();
                return true;
            }
            _ => {}
        }
        false
    }

    fn identify(&self) {
        println!("id name chess_core");
        println!("option name Hash type spin default {DEFAULT_HASH_MB} min 1 max 65536");
        println!("option name Threads type spin default 1 min 1 max 64");
        println!("option name Book type string default");
        println!("option name Tablebase type string default");
        println!("option name Model type string default {MODEL_NAME}");
        println!("uciok");
    }

    fn new_game(&mut self) {
        self.finish_search();
        self.settings.in_book = self.settings.book.is_some();
        if let Some(memory) = self.memory.as_mut() {
            memory.clear();
        }
    }

    fn set_option(&mut self, args: &[&str]) {
        self.finish_search();
        match self.apply_option(args) {
            Ok(true) => {
                self.memory = Some(Memory::new(self.settings.threads, self.settings.hash_mb));
            }
            Ok(false) => {}
            Err(error) => println!("info string {error}"),
        }
    }

    fn apply_option(&mut self, args: &[&str]) -> Result<bool, String> {
        let text = args.join(" ");
        let (name, value) = match text.split_once(" value ") {
            Some((name, value)) => (name, value.trim()),
            None => (text.as_str(), ""),
        };
        let name = name.strip_prefix("name ").unwrap_or(name).trim();
        match name {
            "Hash" => self.settings.hash_mb = value.parse().map_err(|_| "Hash needs a number")?,
            "Threads" => {
                self.settings.threads = value
                    .parse::<usize>()
                    .map_err(|_| "Threads needs a number")?
                    .max(1)
            }
            "Model" if value == MODEL_NAME => self.settings.eval = Arc::new(load_model()?),
            "Model" => {
                let text = std::fs::read_to_string(value)
                    .map_err(|e| format!("Model is {MODEL_NAME} or a JSON file: {value}: {e}"))?;
                self.settings.eval = Arc::new(Net::from_json(&text)?);
            }
            "Book" => {
                self.settings.book = match value {
                    "" => None,
                    path => Some(OpeningBook::load(Path::new(path))?),
                };
                self.settings.in_book = self.settings.book.is_some();
            }
            "Tablebase" => {
                self.settings.tablebase = match value {
                    "" => None,
                    path => Some(EndgameTablebase::open(Path::new(path))?),
                };
            }
            other => return Err(format!("unknown option {other:?}")),
        }
        Ok(name == "Hash" || name == "Threads")
    }

    fn print_evaluation(&self) {
        match self.position.state() {
            Ok(state) => println!(
                "info string eval {:.6}",
                self.settings.eval.evaluate(state.pos(), Color::White)
            ),
            Err(error) => println!("info string {error}"),
        }
    }

    fn set_position(&mut self, args: &[&str]) {
        match args.first() {
            Some(&"startpos") => self.position = position::parse(START_FEN.into(), &args[1..]),
            Some(&"fen") if args.len() >= 7 => {
                self.position = position::parse(args[1..7].join(" "), &args[7..])
            }
            _ => println!("info string position needs startpos or fen"),
        }
    }

    fn start_search(&mut self, args: &[&str]) {
        self.finish_search();
        if let Some(best) = self.book_move() {
            println!("info string book\nbestmove {best}");
            return;
        }
        let root_moves = match self.tablebase_root_moves() {
            Some((names, _)) if names.len() == 1 => {
                println!("info string tablebase\nbestmove {}", names[0]);
                return;
            }
            found => found.map(|(_, ids)| ids),
        };
        let white_to_move = self
            .position
            .state()
            .map_or(true, |state| state.turn() == Color::White);
        let job = SearchJob {
            position: self.position.clone(),
            limits: parse_go(args, white_to_move, self.position.move_number()),
            stop: Arc::new(AtomicBool::new(false)),
            root_moves,
        };
        let stop = job.stop.clone();
        let memory = self.memory.take().expect("memory is back after finish");
        let eval = self.settings.eval.clone();
        let handle = std::thread::spawn(move || search_job(memory, eval, job));
        self.running = Some((handle, stop));
    }

    fn stop_search(&self) {
        if let Some((_, stop)) = &self.running {
            stop.store(true, Relaxed);
        }
    }

    fn finish_search(&mut self) {
        if let Some((handle, _)) = self.running.take() {
            self.memory = Some(handle.join().expect("search thread panicked"));
        }
    }

    fn book_move(&mut self) -> Option<String> {
        let state = self.position.state().ok()?;
        if self.settings.in_book {
            match self
                .settings
                .book
                .as_ref()
                .and_then(|book| book.lookup(state.pos()))
            {
                Some(mv) => return Some(uci_string(&mv)),
                None => self.settings.in_book = false,
            }
        }
        None
    }

    fn tablebase_root_moves(&self) -> Option<(Vec<String>, Vec<u32>)> {
        let state = self.position.state().ok()?;
        let moves = self.settings.tablebase.as_ref()?.root_moves(state.pos())?;
        Some((
            moves.iter().map(uci_string).collect(),
            moves.iter().map(move_id).collect(),
        ))
    }
}

fn search_job(mut memory: Memory, eval: Arc<Net>, job: SearchJob) -> Memory {
    match run_search(&mut memory, &eval, job) {
        Ok(best) => println!("bestmove {best}"),
        Err(error) => println!("info string {error}\nbestmove 0000"),
    }
    memory
}

fn run_search(memory: &mut Memory, eval: &Net, job: SearchJob) -> Result<String, String> {
    let mut state = job.position.state()?;
    let options = Options {
        node_limit: job.limits.node_limit,
        root_moves: job.root_moves,
        ..Options::default()
    };
    let origin = Instant::now();
    let clock = || origin.elapsed().as_secs_f64();
    let control = Control {
        start: 0.0,
        hard_limit: job.limits.hard_time,
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
    Ok(match best {
        Some(mv) => uci_string(&mv),
        None => ordered_moves(state.pos())
            .first()
            .map_or("0000".to_string(), uci_string),
    })
}
