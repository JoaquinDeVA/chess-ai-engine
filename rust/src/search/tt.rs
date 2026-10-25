//! Transposition table, killer moves and move ordering. Port of `search/search_utils.py`.

use std::collections::{HashMap, VecDeque};
use std::hash::{BuildHasherDefault, Hash, Hasher};

use shakmaty::{Move, Role};

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
    pub best: Option<Move>,
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

    pub fn get(&self, key: &Key) -> Option<&Entry> {
        self.map.get(key).map(|(entry, _)| entry)
    }

    /// Store a result, classifying it against the original window of the search.
    pub fn store(
        &mut self,
        key: Key,
        value: f64,
        best: Option<Move>,
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

/// The moves in the order the search tries them: killers first, then captures (best MVV-LVA first)
/// and then the quiet moves, each group keeping its incoming order.
pub fn order(moves: Vec<Move>, killers: &[u32], capture_order: bool) -> Vec<Move> {
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
    captures.extend(quiets);
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
        );
        assert_eq!(ordered, vec![pawn_takes_queen, queen_takes_pawn, b, a]);
        let plain = order(
            vec![a, queen_takes_pawn, b, pawn_takes_queen],
            &[move_id(&b)],
            false,
        );
        assert_eq!(plain, vec![b, a, queen_takes_pawn, pawn_takes_queen]);
    }
}
