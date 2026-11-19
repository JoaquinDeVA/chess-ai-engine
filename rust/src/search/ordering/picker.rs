//! The moves of a node handed out one at a time, best first, so a node that cuts on its first
//! moves never orders the rest.
//!
//! The sequence is the one a full sort would give: the table move; the captures that do not lose
//! material (best MVV-LVA first, killers first among equals); the quiet killers;
//! the other quiet moves by history score; and last the captures that lose material. Every key
//! carries the position the move had before sorting, so keys never tie and a selection gives the
//! same order as a stable sort. History scores are read when the picker is built: the subtrees
//! searched while it hands out moves change the history, and the order must not follow them.

use shakmaty::{Chess, Color, Move, MoveList};

use super::{mvv_lva_score, see_ge, ContinuationHistory, History};
use crate::board::move_id;

/// Most moves of a position (the capacity of a `MoveList`).
const MAX_MOVES: usize = 256;
/// Bits of a key that hold the position before sorting.
const RANK_BITS: u32 = 8;
/// Key of a capture found to lose material, until the last stage hands it out.
const LOSING: i64 = i64::MIN;

#[derive(Clone, Copy)]
enum Stage {
    Table,
    GoodCaptures,
    QuietKillers,
    Quiets,
    LosingCaptures,
    /// A list built by the caller, handed out as it is.
    Listed,
}

pub struct MovePicker {
    /// Captures, then quiet killers, then the other quiet moves, each group in incoming order.
    moves: MoveList,
    /// Selection keys of the captures and of the other quiet moves, by position in `moves`.
    keys: [i64; MAX_MOVES],
    captures_end: usize,
    killers_end: usize,
    cursor: usize,
    stage: Stage,
    table: Option<Move>,
}

/// Key of the move that was `rank`-th of its group: the higher `score`, the earlier; at equal
/// scores the lower `rank`.
fn key(score: i32, rank: usize) -> i64 {
    (i64::from(score) << RANK_BITS) | (MAX_MOVES - 1 - rank) as i64
}

impl MovePicker {
    /// The picker of `moves` (the legal moves of the position of `color` to move).
    pub fn new(
        moves: &[Move],
        killers: &[u32],
        table: Option<u32>,
        (history, color): (&History, Color),
        continuation: Option<(&ContinuationHistory, usize)>,
    ) -> MovePicker {
        let mut killer = [false; MAX_MOVES];
        let mut table_move = None;
        if !killers.is_empty() || table.is_some() {
            for (k, m) in moves.iter().enumerate() {
                let id = move_id(m);
                killer[k] = killers.contains(&id);
                if table_move.is_none() && table == Some(id) {
                    table_move = Some(*m);
                }
            }
        }
        let mut picker = MovePicker {
            moves: MoveList::new(),
            keys: [0; MAX_MOVES],
            captures_end: 0,
            killers_end: 0,
            cursor: 0,
            stage: Stage::Table,
            table: None,
        };
        // Captures: killers first, then the rest.
        picker.moves.extend(
            moves
                .iter()
                .zip(&killer)
                .filter(|&(m, &k)| m.is_capture() && k)
                .map(|(m, _)| *m),
        );
        picker.moves.extend(
            moves
                .iter()
                .zip(&killer)
                .filter(|&(m, &k)| m.is_capture() && !k)
                .map(|(m, _)| *m),
        );
        for (rank, (slot, m)) in picker.keys.iter_mut().zip(picker.moves.iter()).enumerate() {
            *slot = key(mvv_lva_score(m), rank);
        }
        picker.captures_end = picker.moves.len();
        picker.moves.extend(
            moves
                .iter()
                .zip(&killer)
                .filter(|&(m, &k)| !m.is_capture() && k)
                .map(|(m, _)| *m),
        );
        picker.killers_end = picker.moves.len();
        picker.moves.extend(
            moves
                .iter()
                .zip(&killer)
                .filter(|&(m, &k)| !m.is_capture() && !k)
                .map(|(m, _)| *m),
        );
        let others = picker.killers_end;
        for (rank, (slot, m)) in picker.keys[others..]
            .iter_mut()
            .zip(picker.moves[others..].iter())
            .enumerate()
        {
            let score = history.score(color, m)
                + continuation.map_or(0, |(cont, previous)| cont.score(color, previous, m));
            *slot = key(score, rank);
        }
        picker.table = table_move;
        picker
    }

    /// The moves in exactly `list`, in that order (the root, which filters and rotates them).
    pub fn replace_with(&mut self, list: MoveList) {
        self.moves = list;
        self.cursor = 0;
        self.stage = Stage::Listed;
    }

    /// The best of the moves from `cursor` to `end` by key, moved to `cursor`.
    fn select(&mut self, end: usize) -> Move {
        let mut best = self.cursor;
        for i in self.cursor + 1..end {
            if self.keys[i] > self.keys[best] {
                best = i;
            }
        }
        self.moves.swap(self.cursor, best);
        self.keys.swap(self.cursor, best);
        self.cursor += 1;
        self.moves[self.cursor - 1]
    }

    /// The next move of `pos` (the position the picker was built for), `None` when all were given.
    pub fn next(&mut self, pos: &Chess) -> Option<Move> {
        loop {
            match self.stage {
                Stage::Table => {
                    self.stage = Stage::GoodCaptures;
                    if self.table.is_some() {
                        return self.table;
                    }
                }
                Stage::GoodCaptures => {
                    while self.cursor < self.captures_end {
                        let m = self.select(self.captures_end);
                        if Some(m) == self.table {
                            continue;
                        }
                        if see_ge(pos, &m, 0) {
                            return Some(m);
                        }
                        self.keys[self.cursor - 1] = LOSING;
                    }
                    self.stage = Stage::QuietKillers;
                }
                Stage::QuietKillers => {
                    while self.cursor < self.killers_end {
                        self.cursor += 1;
                        let m = self.moves[self.cursor - 1];
                        if Some(m) != self.table {
                            return Some(m);
                        }
                    }
                    self.stage = Stage::Quiets;
                }
                Stage::Quiets => {
                    while self.cursor < self.moves.len() {
                        let m = self.select(self.moves.len());
                        if Some(m) != self.table {
                            return Some(m);
                        }
                    }
                    self.cursor = 0;
                    self.stage = Stage::LosingCaptures;
                }
                Stage::LosingCaptures => {
                    while self.cursor < self.captures_end {
                        self.cursor += 1;
                        let m = self.moves[self.cursor - 1];
                        if self.keys[self.cursor - 1] == LOSING && Some(m) != self.table {
                            return Some(m);
                        }
                    }
                    return None;
                }
                Stage::Listed => {
                    let m = self.moves.get(self.cursor).copied();
                    self.cursor += 1;
                    return m;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_support::{position, quiet};
    use super::*;
    use shakmaty::{Position, Role, Square};

    fn all(mut picker: MovePicker, pos: &Chess) -> Vec<Move> {
        std::iter::from_fn(|| picker.next(pos)).collect()
    }

    fn picker_of(moves: &[Move], killers: &[u32], history: &History) -> MovePicker {
        MovePicker::new(moves, killers, None, (history, Color::White), None)
    }

    #[test]
    fn history_orders_quiet_moves_by_cuts_after_the_killers() {
        let a = quiet(Square::B1, Square::C3);
        let b = quiet(Square::G1, Square::F3);
        let c = quiet(Square::A2, Square::A3);
        let history = History::default();
        history.reward(Color::White, &c, 4);
        history.reward(Color::White, &b, 2);
        let picker = picker_of(&[a, b, c], &[move_id(&a)], &history);
        assert_eq!(all(picker, &Chess::default()), vec![a, c, b]);
    }

    #[test]
    fn equal_scores_keep_the_incoming_order() {
        let moves: Vec<Move> = (0..6)
            .map(|i| quiet(Square::new(i), Square::new(i + 16)))
            .collect();
        let picker = picker_of(&moves, &[], &History::default());
        assert_eq!(all(picker, &Chess::default()), moves);
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
        let picker = picker_of(
            &[a, queen_takes_pawn, b, pawn_takes_queen],
            &[move_id(&b)],
            &History::default(),
        );
        let order = all(picker, &position("4k3/8/8/8/3p4/1q6/P7/3QK3 w - - 0 1"));
        assert_eq!(order, vec![pawn_takes_queen, queen_takes_pawn, b, a]);
    }

    #[test]
    fn a_capture_that_loses_material_comes_after_the_quiet_moves() {
        // The white queen takes a pawn that a pawn defends: SEE < 0.
        let pos = position("4k3/8/2p5/3p4/8/8/8/3QK3 w - - 0 1");
        let moves: Vec<Move> = pos.legal_moves().into_iter().collect();
        let losing = *moves.iter().find(|m| m.is_capture()).unwrap();
        let picker = picker_of(&moves, &[], &History::default());
        let order = all(picker, &pos);
        assert_eq!(order.len(), moves.len());
        assert_eq!(*order.last().unwrap(), losing);
    }

    #[test]
    fn a_listed_picker_hands_out_the_list_as_it_is() {
        let a = quiet(Square::B1, Square::C3);
        let b = quiet(Square::G1, Square::F3);
        let mut picker = picker_of(&[a, b], &[], &History::default());
        picker.replace_with([b, a].into_iter().collect());
        assert_eq!(all(picker, &Chess::default()), vec![b, a]);
    }
}
