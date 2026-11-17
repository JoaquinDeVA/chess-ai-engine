//! Move ordering: table move, captures by MVV-LVA, killer moves and the history of cuts.

mod history;
mod killers;
mod picker;
mod scoring;
mod see;

pub use history::{piece_to, ContinuationHistory, History};
pub use killers::{Killers, MAX_KILLERS_PER_PLY};
pub use picker::MovePicker;
pub use scoring::{mvv_lva_score, order_noisy};
pub use see::see_ge;

use shakmaty::Move;

use crate::board::move_id;

/// Put the move with the given identity first, if it is among `moves`.
pub fn move_to_front(moves: &mut [Move], id: Option<u32>) {
    if let Some(id) = id {
        if let Some(position) = moves.iter().position(|m| move_id(m) == id) {
            moves[..=position].rotate_right(1);
        }
    }
}

/// Sort `moves` by descending `keys`, keeping the incoming order among equal keys (insertion sort:
/// stable and with no allocation, for the few dozen moves of a position).
fn sort_descending(moves: &mut [Move], keys: &mut [i32]) {
    for i in 1..moves.len() {
        let mut j = i;
        while j > 0 && keys[j - 1] < keys[j] {
            moves.swap(j - 1, j);
            keys.swap(j - 1, j);
            j -= 1;
        }
    }
}

#[cfg(test)]
mod test_support {
    use shakmaty::{fen::Fen, CastlingMode, Chess, Move, Role, Square};

    pub fn position(fen: &str) -> Chess {
        fen.parse::<Fen>()
            .unwrap()
            .into_position(CastlingMode::Standard)
            .unwrap()
    }

    pub fn quiet(from: Square, to: Square) -> Move {
        Move::Normal {
            role: Role::Knight,
            from,
            capture: None,
            to,
            promotion: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::test_support::quiet;
    use super::*;
    use shakmaty::Square;

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
}
