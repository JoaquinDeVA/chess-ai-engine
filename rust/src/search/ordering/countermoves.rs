//! Countermoves: the quiet reply that last cut after each move of the opponent.

use shakmaty::{Color, Move};

use super::MOVE_IDS;
use crate::board::move_id;

/// The quiet reply that last cut after each move (by mover and `move_id` of the move answered).
pub struct Countermoves {
    replies: Vec<u32>,
}

impl Default for Countermoves {
    fn default() -> Countermoves {
        Countermoves {
            replies: vec![NO_MOVE; 2 * MOVE_IDS],
        }
    }
}

/// Marks a slot with no reply (no move has this identity).
const NO_MOVE: u32 = u32::MAX;

impl Countermoves {
    fn slot(replier: Color, previous: u32) -> usize {
        usize::from(replier == Color::Black) * MOVE_IDS + previous as usize
    }

    pub fn clear(&mut self) {
        self.replies.fill(NO_MOVE);
    }

    pub fn get(&self, replier: Color, previous: Option<u32>) -> Option<u32> {
        let reply = self.replies[Countermoves::slot(replier, previous?)];
        (reply != NO_MOVE).then_some(reply)
    }

    pub fn record(&mut self, replier: Color, previous: Option<u32>, reply: &Move) {
        if let Some(previous) = previous {
            self.replies[Countermoves::slot(replier, previous)] = move_id(reply);
        }
    }
}
