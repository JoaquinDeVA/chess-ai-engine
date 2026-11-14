use std::io::{self, BufRead};
use std::path::Path;
use std::sync::Arc;

use _core::board::{move_id, uci_string};
use _core::eval::Net;
use _core::knowledge::book::OpeningBook;
use _core::knowledge::tablebase::EndgameTablebase;
use _core::search::{Memory, DEFAULT_HASH_MB};
use shakmaty::Color;

use crate::limits::parse_go;
use crate::position::{self, Position};
use crate::search_job::RunningSearch;

pub const MODEL_NAME: &str = "net_pst_w64";
const MODEL_JSON: &str = include_str!("../../../models/net_pst_w64.json");
pub(crate) const START_FEN: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

/// Settings changed with `setoption`.
struct EngineSettings {
    threads: usize,
    hash_mb: usize,
    eval: Arc<Net>,
    book: Option<OpeningBook>,
    in_book: bool,
    tablebase: Option<EndgameTablebase>,
}

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
            "ponderhit" => self.ponder_hit(),
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
        println!("option name Ponder type check default false");
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
            "Ponder" => {
                value
                    .parse::<bool>()
                    .map_err(|_| "Ponder needs true or false")?;
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
        let white_to_move = self
            .position
            .state()
            .map_or(true, |state| state.turn() == Color::White);
        let limits = parse_go(args, white_to_move, self.position.move_number());
        if !limits.ponder {
            if let Some(best) = self.book_move() {
                println!("info string book\nbestmove {best}");
                return;
            }
        }
        let root_moves = if limits.ponder {
            None
        } else {
            match self.tablebase_root_moves() {
                Some((names, _)) if names.len() == 1 => {
                    println!("info string tablebase\nbestmove {}", names[0]);
                    return;
                }
                found => found.map(|(_, ids)| ids),
            }
        };
        let memory = self.memory.take().expect("memory is back after finish");
        self.running = Some(RunningSearch::start(
            memory,
            self.settings.eval.clone(),
            self.position.clone(),
            limits,
            root_moves,
        ));
    }

    fn stop_search(&self) {
        if let Some(search) = &self.running {
            search.stop();
        }
    }

    fn ponder_hit(&self) {
        if let Some(search) = &self.running {
            search.ponder_hit();
        }
    }

    fn finish_search(&mut self) {
        if let Some(search) = self.running.take() {
            self.memory = Some(search.finish());
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
