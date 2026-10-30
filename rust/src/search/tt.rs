//! Transposition table, killer moves and move ordering.

use std::collections::{HashMap, VecDeque};
use std::hash::{BuildHasherDefault, Hash, Hasher};

use shakmaty::{Color, Move, Role};

use crate::board::{move_id, Key};

/// Number of killer moves remembered per depth.
pub const MAX_KILLERS_PER_DEPTH: usize = 3;
/// Default maximum number of entries of a table.
pub const DEFAULT_MAX_TT_ENTRIES: usize = 500_000;

/// What an entry says about the value of the position.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Flag {
    Exact,
    LowerBound,
    UpperBound,
}

/// One entry of the table.
#[derive(Clone, Copy, Debug)]
pub struct Entry {
    pub value: f64,
    /// Identity of the best move (see `board::move_id`), if any.
    pub best: Option<u32>,
    pub depth: i32,
    pub flag: Flag,
}

/// The keys are hashes already: use them as they are.
#[derive(Default)]
pub struct KeyHasher(u64);

impl Hasher for KeyHasher {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, _bytes: &[u8]) {
        unreachable!("only integers are hashed");
    }
    fn write_u64(&mut self, value: u64) {
        self.0 ^= value;
    }
    fn write_u32(&mut self, value: u32) {
        self.0 ^= (value as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    }
}

impl Hash for Key {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_u64(self.zobrist);
        state.write_u32(self.halfmoves);
        state.write_u32(self.fullmoves.rotate_left(7));
    }
}

type Map = HashMap<Key, (Entry, u64), BuildHasherDefault<KeyHasher>>;

/// Transposition table of bounded size: when it is full the entries stored longest ago are
/// dropped (a refreshed entry counts as new), which keeps memory bounded during long games.
pub struct Table {
    map: Map,
    order: VecDeque<(Key, u64)>,
    sequence: u64,
    capacity: usize,
}

impl Table {
    pub fn new(capacity: usize) -> Table {
        Table {
            map: Map::default(),
            order: VecDeque::new(),
            sequence: 0,
            capacity,
        }
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    pub fn clear(&mut self) {
        self.map.clear();
        self.order.clear();
    }

    pub fn set_capacity(&mut self, capacity: usize) {
        self.capacity = capacity;
    }

    pub fn get(&self, key: &Key) -> Option<Entry> {
        self.map.get(key).map(|(entry, _)| *entry)
    }

    /// Store a result, classifying it against the original window of the search.
    pub fn store(
        &mut self,
        key: Key,
        value: f64,
        best: Option<u32>,
        depth: i32,
        alpha_orig: f64,
        beta_orig: f64,
    ) {
        let flag = if value <= alpha_orig {
            Flag::UpperBound
        } else if value >= beta_orig {
            Flag::LowerBound
        } else {
            Flag::Exact
        };
        self.sequence += 1;
        self.map.insert(
            key,
            (
                Entry {
                    value,
                    best,
                    depth,
                    flag,
                },
                self.sequence,
            ),
        );
        self.order.push_back((key, self.sequence));
        while self.map.len() > self.capacity {
            let Some((old, seq)) = self.order.pop_front() else {
                break;
            };
            if self.map.get(&old).is_some_and(|(_, s)| *s == seq) {
                self.map.remove(&old);
            }
        }
        if self.order.len() > 4 * self.map.len() + 1024 {
            let map = &self.map;
            self.order
                .retain(|(key, seq)| map.get(key).is_some_and(|(_, s)| s == seq));
        }
    }
}

/// The transposition table of a search: the exact one of a single thread (bounded, oldest entries
/// dropped first, as in the Python engine) or the lock-free one shared by several threads.
pub enum Tt {
    Local(Table),
    Shared(std::sync::Arc<super::shared::SharedTable>),
}

impl Tt {
    pub fn get(&self, key: &Key) -> Option<Entry> {
        match self {
            Tt::Local(table) => table.get(key),
            Tt::Shared(table) => table.get(key),
        }
    }

    pub fn store(
        &mut self,
        key: Key,
        value: f64,
        best: Option<u32>,
        depth: i32,
        alpha_orig: f64,
        beta_orig: f64,
    ) {
        match self {
            Tt::Local(table) => table.store(key, value, best, depth, alpha_orig, beta_orig),
            Tt::Shared(table) => table.store(key, value, best, depth, alpha_orig, beta_orig),
        }
    }

    pub fn clear(&mut self) {
        match self {
            Tt::Local(table) => table.clear(),
            Tt::Shared(table) => table.clear(),
        }
    }

    /// Entries held (an estimate for the shared table).
    pub fn len(&self) -> usize {
        match self {
            Tt::Local(table) => table.len(),
            Tt::Shared(table) => table.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn set_capacity(&mut self, capacity: usize) {
        if let Tt::Local(table) = self {
            table.set_capacity(capacity);
        }
    }
}

/// Killer moves by remaining depth, identified by their squares and promotion (as `chess.Move`
/// compares them), not by what they capture.
#[derive(Default)]
pub struct Killers {
    by_depth: Vec<Vec<u32>>,
}

impl Killers {
    pub fn clear(&mut self) {
        self.by_depth.clear();
    }

    pub fn get(&self, depth: i32) -> &[u32] {
        self.by_depth
            .get(depth as usize)
            .map_or(&[], |v| v.as_slice())
    }

    /// Remember `m` as a killer of `depth` (at most three per depth).
    pub fn record(&mut self, depth: i32, m: Move) {
        let depth = depth as usize;
        if self.by_depth.len() <= depth {
            self.by_depth.resize_with(depth + 1, Vec::new);
        }
        let moves = &mut self.by_depth[depth];
        let id = move_id(&m);
        if moves.contains(&id) {
            return;
        }
        moves.push(id);
        if moves.len() > MAX_KILLERS_PER_DEPTH {
            moves.remove(0);
        }
    }
}

fn piece_order(role: Role) -> i32 {
    role as i32
}

/// Most-Valuable-Victim / Least-Valuable-Attacker score of a capture (0 for quiet moves).
pub fn mvv_lva_score(m: &Move) -> i32 {
    let victim = match *m {
        Move::Normal {
            capture: Some(role),
            ..
        } => role,
        Move::EnPassant { .. } => Role::Pawn,
        _ => return 0,
    };
    10 * piece_order(victim) - piece_order(m.role())
}

/// Largest identity of a move (`board::move_id`) plus one.
const MOVE_IDS: usize = 1 << 15;
/// Value above which the history scores are halved.
const HISTORY_LIMIT: i32 = 1 << 24;

/// How often each quiet move (by mover and `move_id`) caused a cut-off, weighted by depth.
pub struct History {
    scores: Vec<i32>,
}

impl Default for History {
    fn default() -> History {
        History {
            scores: vec![0; 2 * MOVE_IDS],
        }
    }
}

impl History {
    fn slot(color: Color, m: &Move) -> usize {
        usize::from(color == Color::Black) * MOVE_IDS + move_id(m) as usize
    }

    pub fn clear(&mut self) {
        self.scores.fill(0);
    }

    pub fn score(&self, color: Color, m: &Move) -> i32 {
        self.scores[History::slot(color, m)]
    }

    /// A quiet move caused a cut-off with `depth` plies left.
    pub fn reward(&mut self, color: Color, m: &Move, depth: i32) {
        let slot = History::slot(color, m);
        self.scores[slot] += depth * depth;
        if self.scores[slot] > HISTORY_LIMIT {
            for score in &mut self.scores {
                *score /= 2;
            }
        }
    }
}

/// Put the move with the given identity first, if it is among `moves`.
pub fn move_to_front(moves: &mut [Move], id: Option<u32>) {
    if let Some(id) = id {
        if let Some(position) = moves.iter().position(|m| move_id(m) == id) {
            moves[..=position].rotate_right(1);
        }
    }
}

/// The moves in the order the search tries them: killers first, then captures (best MVV-LVA first)
/// and then the quiet moves, each group keeping its incoming order. With a history the quiet moves
/// that are not killers come by their score instead, best first.
pub fn order(
    moves: Vec<Move>,
    killers: &[u32],
    capture_order: bool,
    history: Option<(&History, Color)>,
) -> Vec<Move> {
    let mut front: Vec<Move> = Vec::with_capacity(moves.len());
    let mut rest: Vec<Move> = Vec::new();
    for m in moves {
        if killers.contains(&move_id(&m)) {
            front.push(m);
        } else {
            rest.push(m);
        }
    }
    front.extend(rest);
    if !capture_order {
        return front;
    }
    let (mut captures, quiets): (Vec<Move>, Vec<Move>) =
        front.into_iter().partition(|m| m.is_capture());
    captures.sort_by_key(|m| std::cmp::Reverse(mvv_lva_score(m)));
    match history {
        None => captures.extend(quiets),
        Some((history, color)) => {
            let (killer_moves, mut others): (Vec<Move>, Vec<Move>) = quiets
                .into_iter()
                .partition(|m| killers.contains(&move_id(m)));
            others.sort_by_key(|m| std::cmp::Reverse(history.score(color, m)));
            captures.extend(killer_moves);
            captures.extend(others);
        }
    }
    captures
}

#[cfg(test)]
mod tests {
    use super::*;
    use shakmaty::Square;

    fn quiet(from: Square, to: Square) -> Move {
        Move::Normal {
            role: Role::Knight,
            from,
            capture: None,
            to,
            promotion: None,
        }
    }

    fn key(n: u64) -> Key {
        Key {
            zobrist: n,
            halfmoves: 0,
            fullmoves: 1,
        }
    }

    #[test]
    fn bounds_are_classified_against_the_original_window() {
        let mut table = Table::new(10);
        table.store(key(1), 0.2, None, 3, 0.3, 0.9);
        table.store(key(2), 0.95, None, 3, 0.3, 0.9);
        table.store(key(3), 0.5, None, 3, 0.3, 0.9);
        assert_eq!(table.get(&key(1)).unwrap().flag, Flag::UpperBound);
        assert_eq!(table.get(&key(2)).unwrap().flag, Flag::LowerBound);
        assert_eq!(table.get(&key(3)).unwrap().flag, Flag::Exact);
    }

    #[test]
    fn the_oldest_entries_are_dropped_and_a_refreshed_one_survives() {
        let mut table = Table::new(3);
        for n in 1..=3 {
            table.store(key(n), 0.5, None, 1, 0.0, 1.0);
        }
        table.store(key(1), 0.6, None, 2, 0.0, 1.0); // refresh
        table.store(key(4), 0.5, None, 1, 0.0, 1.0); // evicts 2, the oldest
        assert!(table.get(&key(1)).is_some() && table.get(&key(2)).is_none());
        assert!(table.get(&key(3)).is_some() && table.get(&key(4)).is_some());
        assert_eq!(table.len(), 3);
    }

    #[test]
    fn killers_keep_three_per_depth_without_repeats() {
        let mut killers = Killers::default();
        let moves: Vec<Move> = (0..5)
            .map(|i| quiet(Square::new(i), Square::new(i + 16)))
            .collect();
        for m in &moves {
            killers.record(4, *m);
        }
        killers.record(4, moves[4]);
        assert_eq!(
            killers.get(4),
            &moves[2..5].iter().map(move_id).collect::<Vec<_>>()[..]
        );
        assert!(killers.get(2).is_empty());
    }

    #[test]
    fn the_table_move_goes_first() {
        let a = quiet(Square::B1, Square::C3);
        let b = quiet(Square::G1, Square::F3);
        let c = quiet(Square::A2, Square::A3);
        let mut moves = vec![a, b, c];
        move_to_front(&mut moves, Some(move_id(&c)));
        assert_eq!(moves, vec![c, a, b]);
        move_to_front(&mut moves, Some(99)); // not among the moves: nothing changes
        move_to_front(&mut moves, None);
        assert_eq!(moves, vec![c, a, b]);
    }

    #[test]
    fn history_orders_quiet_moves_by_cuts_after_the_killers() {
        let a = quiet(Square::B1, Square::C3);
        let b = quiet(Square::G1, Square::F3);
        let c = quiet(Square::A2, Square::A3);
        let mut history = History::default();
        history.reward(Color::White, &c, 4);
        history.reward(Color::White, &b, 2);
        let ordered = order(
            vec![a, b, c],
            &[move_id(&a)],
            true,
            Some((&history, Color::White)),
        );
        assert_eq!(ordered, vec![a, c, b]);
        assert_eq!(
            history.score(Color::Black, &c),
            0,
            "the colours have their own scores"
        );
        history.clear();
        assert_eq!(history.score(Color::White, &c), 0);
    }

    #[test]
    fn captures_come_first_best_victim_first_then_quiet_moves_with_killers_in_front() {
        let pawn_takes_queen = Move::Normal {
            role: Role::Pawn,
            from: Square::A2,
            capture: Some(Role::Queen),
            to: Square::B3,
            promotion: None,
        };
        let queen_takes_pawn = Move::Normal {
            role: Role::Queen,
            from: Square::D1,
            capture: Some(Role::Pawn),
            to: Square::D4,
            promotion: None,
        };
        let a = quiet(Square::B1, Square::C3);
        let b = quiet(Square::G1, Square::F3);
        let ordered = order(
            vec![a, queen_takes_pawn, b, pawn_takes_queen],
            &[move_id(&b)],
            true,
            None,
        );
        assert_eq!(ordered, vec![pawn_takes_queen, queen_takes_pawn, b, a]);
        let plain = order(
            vec![a, queen_takes_pawn, b, pawn_takes_queen],
            &[move_id(&b)],
            false,
            None,
        );
        assert_eq!(plain, vec![b, a, queen_takes_pawn, pawn_takes_queen]);
    }
}
