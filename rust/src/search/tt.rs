//! Transposition table.

use std::collections::{HashMap, VecDeque};
use std::hash::{BuildHasherDefault, Hash, Hasher};

use crate::board::Key;

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

#[cfg(test)]
mod tests {
    use super::*;

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
}
