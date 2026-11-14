//! History heuristics: how often quiet moves caused a cut-off, alone and after the opponent's move.

use std::sync::atomic::{AtomicI32, Ordering::Relaxed};
use std::sync::Arc;

use shakmaty::{Color, Move};

use super::MOVE_IDS;
use crate::board::move_id;

/// Bound of the history scores: a bonus pushes a score less the closer it is to it.
const HISTORY_LIMIT: i32 = 1 << 14;
/// Value above which the continuation history scores are halved.
const CONTINUATION_LIMIT: i32 = 1 << 24;

/// How often each quiet move (by mover and `move_id`) caused a cut-off, weighted by depth. A clone
/// is another handle to the same scores: the threads of a search share them, with relaxed atomics
/// (a lost update only blurs a heuristic).
#[derive(Clone)]
pub struct History {
    scores: Arc<[AtomicI32]>,
}

impl Default for History {
    fn default() -> History {
        History {
            scores: (0..2 * MOVE_IDS).map(|_| AtomicI32::new(0)).collect(),
        }
    }
}

impl History {
    fn slot(color: Color, m: &Move) -> usize {
        usize::from(color == Color::Black) * MOVE_IDS + move_id(m) as usize
    }

    pub fn clear(&self) {
        for score in self.scores.iter() {
            score.store(0, Relaxed);
        }
    }

    pub fn score(&self, color: Color, m: &Move) -> i32 {
        self.scores[History::slot(color, m)].load(Relaxed)
    }

    /// A quiet move caused a cut-off with `depth` plies left.
    pub fn reward(&self, color: Color, m: &Move, depth: i32) {
        self.add(color, m, depth * depth);
    }

    /// A quiet move was tried before the one that cut, and failed to.
    pub fn penalize(&self, color: Color, m: &Move, depth: i32) {
        self.add(color, m, -depth * depth);
    }

    /// Move the score towards the bonus, the more slowly the closer it is to `HISTORY_LIMIT`
    /// (history gravity): the scores stay inside `±HISTORY_LIMIT` with no halving of the table.
    fn add(&self, color: Color, m: &Move, amount: i32) {
        let score = &self.scores[History::slot(color, m)];
        let old = score.load(Relaxed);
        score.store(old + amount - old * amount.abs() / HISTORY_LIMIT, Relaxed);
    }
}

/// Piece-and-target squares: 6 roles by 64 squares.
pub const PIECE_TO: usize = 6 * 64;

/// Index of a move by the piece that moves and where it lands.
pub fn piece_to(m: &Move) -> usize {
    (m.role() as usize - 1) * 64 + m.to() as usize
}

/// How often a quiet move (piece and target) caused a cut-off right after the opponent's move
/// (piece and target), weighted by depth. Shared by the threads like [`History`].
#[derive(Clone)]
pub struct ContinuationHistory {
    scores: Arc<[AtomicI32]>,
}

impl Default for ContinuationHistory {
    fn default() -> ContinuationHistory {
        ContinuationHistory {
            scores: (0..2 * PIECE_TO * PIECE_TO)
                .map(|_| AtomicI32::new(0))
                .collect(),
        }
    }
}

impl ContinuationHistory {
    fn slot(color: Color, previous: usize, m: &Move) -> usize {
        (usize::from(color == Color::Black) * PIECE_TO + previous) * PIECE_TO + piece_to(m)
    }

    pub fn clear(&self) {
        for score in self.scores.iter() {
            score.store(0, Relaxed);
        }
    }

    pub fn score(&self, color: Color, previous: usize, m: &Move) -> i32 {
        self.scores[ContinuationHistory::slot(color, previous, m)].load(Relaxed)
    }

    /// A quiet move caused a cut-off with `depth` plies left, right after the move `previous`.
    pub fn reward(&self, color: Color, previous: usize, m: &Move, depth: i32) {
        let score = &self.scores[ContinuationHistory::slot(color, previous, m)];
        if (score.fetch_add(depth * depth, Relaxed) + depth * depth).abs() > CONTINUATION_LIMIT {
            for score in self.scores.iter() {
                score.store(score.load(Relaxed) / 2, Relaxed);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::search::ordering::test_support::quiet;
    use shakmaty::Square;

    #[test]
    fn history_scores_each_colour_apart_and_clears() {
        let c = quiet(Square::A2, Square::A3);
        let history = History::default();
        history.reward(Color::White, &c, 4);
        assert!(history.score(Color::White, &c) > 0);
        assert_eq!(
            history.score(Color::Black, &c),
            0,
            "the colours have their own scores"
        );
        history.clear();
        assert_eq!(history.score(Color::White, &c), 0);
    }
}
