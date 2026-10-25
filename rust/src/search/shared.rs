//! A transposition table that several threads read and write without locks.
//!
//! Each slot is three words: `check = hash ^ data1 ^ data2`, the value and a packed word with the
//! best move, depth, bound, age and a validity bit. A reader accepts the slot only if the check
//! matches, so a slot torn by two writers at once is simply ignored (the usual XOR trick). The key
//! is a 64-bit hash of the position and the two counters, so two different positions can collide
//! with probability 2^-64 per pair; the single-thread table compares the full key instead.

use std::sync::atomic::{AtomicU64, AtomicU8, Ordering::Relaxed};

use super::tt::{Entry, Flag};
use crate::board::Key;

/// Bytes of one slot.
const SLOT_BYTES: usize = 24;
const VALID: u64 = 1 << 63;

fn hash(key: &Key) -> u64 {
    key.zobrist
        ^ (key.halfmoves as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ (key.fullmoves as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F)
}

fn flag_bits(flag: Flag) -> u64 {
    match flag {
        Flag::Exact => 0,
        Flag::LowerBound => 1,
        Flag::UpperBound => 2,
    }
}

/// Lock-free transposition table with one slot per index.
pub struct SharedTable {
    slots: Vec<[AtomicU64; 3]>,
    age: AtomicU8,
}

impl SharedTable {
    /// A table of about `megabytes` MB.
    pub fn with_megabytes(megabytes: usize) -> SharedTable {
        SharedTable::with_slots((megabytes.max(1) * 1024 * 1024) / SLOT_BYTES)
    }

    /// A table with `count` slots (at least 1024).
    pub fn with_slots(count: usize) -> SharedTable {
        let count = count.max(1024);
        SharedTable {
            slots: (0..count)
                .map(|_| [AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)])
                .collect(),
            age: AtomicU8::new(0),
        }
    }

    pub fn capacity(&self) -> usize {
        self.slots.len()
    }

    /// Mark the start of a new search: old entries become easier to replace.
    pub fn new_search(&self) {
        self.age.fetch_add(1, Relaxed);
    }

    pub fn clear(&self) {
        for slot in &self.slots {
            for word in slot {
                word.store(0, Relaxed);
            }
        }
    }

    /// Number of valid slots, counted by scanning (slow: for statistics only).
    pub fn len(&self) -> usize {
        self.slots
            .iter()
            .filter(|s| s[2].load(Relaxed) & VALID != 0)
            .count()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn index(&self, hash: u64) -> usize {
        ((hash as u128 * self.slots.len() as u128) >> 64) as usize
    }

    fn read(&self, index: usize, hash: u64) -> Option<(Entry, u8)> {
        let slot = &self.slots[index];
        let (check, value, data) = (
            slot[0].load(Relaxed),
            slot[1].load(Relaxed),
            slot[2].load(Relaxed),
        );
        if data & VALID == 0 || check ^ value ^ data != hash {
            return None;
        }
        let best = (data & 0xFFFF) as u32;
        let flag = match (data >> 24) & 3 {
            0 => Flag::Exact,
            1 => Flag::LowerBound,
            _ => Flag::UpperBound,
        };
        let entry = Entry {
            value: f64::from_bits(value),
            best: if best == 0 { None } else { Some(best - 1) },
            depth: ((data >> 16) & 0xFF) as i32,
            flag,
        };
        Some((entry, ((data >> 32) & 0xFF) as u8))
    }

    pub fn get(&self, key: &Key) -> Option<Entry> {
        let hash = hash(key);
        self.read(self.index(hash), hash).map(|(entry, _)| entry)
    }

    /// Store a result, classifying it against the original window of the search. A slot holding
    /// a deeper entry of the current search for another position is kept.
    pub fn store(
        &self,
        key: Key,
        value: f64,
        best: Option<u32>,
        depth: i32,
        alpha_orig: f64,
        beta_orig: f64,
    ) {
        let hash = hash(&key);
        let index = self.index(hash);
        let age = self.age.load(Relaxed);
        if let Some((old, old_age)) = self.read_any(index) {
            if old_age == age && old.0 != hash && old.1.depth > depth {
                return;
            }
        }
        let flag = if value <= alpha_orig {
            Flag::UpperBound
        } else if value >= beta_orig {
            Flag::LowerBound
        } else {
            Flag::Exact
        };
        let data = VALID
            | best.map_or(0, |id| u64::from(id) + 1)
            | ((depth.clamp(0, 255) as u64) << 16)
            | (flag_bits(flag) << 24)
            | ((age as u64) << 32);
        let value = value.to_bits();
        let slot = &self.slots[index];
        slot[1].store(value, Relaxed);
        slot[2].store(data, Relaxed);
        slot[0].store(hash ^ value ^ data, Relaxed);
    }

    /// The valid slot at `index` whatever its key: `(key hash, entry)` and its age.
    fn read_any(&self, index: usize) -> Option<((u64, Entry), u8)> {
        let slot = &self.slots[index];
        let (check, value, data) = (
            slot[0].load(Relaxed),
            slot[1].load(Relaxed),
            slot[2].load(Relaxed),
        );
        if data & VALID == 0 {
            return None;
        }
        let hash = check ^ value ^ data;
        self.read(index, hash)
            .map(|(entry, age)| ((hash, entry), age))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(n: u64) -> Key {
        Key {
            zobrist: n.wrapping_mul(0x9E37_79B9_7F4A_7C15),
            halfmoves: 0,
            fullmoves: 1,
        }
    }

    #[test]
    fn an_entry_comes_back_as_it_was_stored() {
        let table = SharedTable::with_slots(4096);
        table.store(key(1), 0.75, Some(1234), 5, 0.0, 1.0);
        let entry = table.get(&key(1)).unwrap();
        assert_eq!(
            (entry.value, entry.best, entry.depth, entry.flag),
            (0.75, Some(1234), 5, Flag::Exact)
        );
        table.store(key(2), 0.9, None, 3, 0.0, 0.5);
        let entry = table.get(&key(2)).unwrap();
        assert_eq!((entry.best, entry.flag), (None, Flag::LowerBound));
        assert!(table.get(&key(3)).is_none());
    }

    #[test]
    fn a_deeper_entry_of_the_same_search_is_not_replaced_by_a_shallower_one_of_another_position() {
        let table = SharedTable::with_slots(1024);
        // find two keys that share a slot
        let first = key(1);
        let slot = table.index(hash(&first));
        let second = (2..100_000u64)
            .map(key)
            .find(|k| table.index(hash(k)) == slot)
            .unwrap();
        table.store(first, 0.5, None, 6, 0.0, 1.0);
        table.store(second, 0.6, None, 2, 0.0, 1.0);
        assert!(table.get(&first).is_some() && table.get(&second).is_none());
        table.new_search();
        table.store(second, 0.6, None, 2, 0.0, 1.0);
        assert!(
            table.get(&second).is_some(),
            "an old entry yields to a new search"
        );
    }

    #[test]
    fn threads_writing_the_same_slots_never_produce_a_wrong_entry() {
        let table = SharedTable::with_slots(1024);
        let expected = |n: u64| (n % 1000) as f64 / 1000.0;
        std::thread::scope(|scope| {
            for thread in 0..8u64 {
                let table = &table;
                scope.spawn(move || {
                    for round in 0..20_000u64 {
                        let n = (round * 8 + thread) % 5000;
                        table.store(
                            key(n),
                            expected(n),
                            Some((n % 1000) as u32),
                            (n % 30) as i32,
                            0.0,
                            2.0,
                        );
                        let probe = (round * 31 + thread) % 5000;
                        if let Some(entry) = table.get(&key(probe)) {
                            assert_eq!(entry.value, expected(probe));
                            assert_eq!(entry.best, Some((probe % 1000) as u32));
                        }
                    }
                });
            }
        });
    }
}
