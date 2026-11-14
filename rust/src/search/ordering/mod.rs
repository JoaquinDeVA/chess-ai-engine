//! Move ordering: table move, captures by MVV-LVA, killer moves and the history of cuts.

mod countermoves;
mod history;
mod killers;
mod scoring;
mod see;

pub use countermoves::Countermoves;
pub use history::{piece_to, ContinuationHistory, History};
pub use killers::{Killers, MAX_KILLERS_PER_PLY};
pub use scoring::{mvv_lva_score, order_noisy};
pub use see::see_ge;

use shakmaty::{Color, Move, MoveList};

use crate::board::move_id;

/// Largest identity of a move (`board::move_id`) plus one.
const MOVE_IDS: usize = 1 << 15;

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

/// The moves in the order the search tries them: killers first, then captures (best MVV-LVA first)
/// and then the quiet moves, each group keeping its incoming order. With a history the quiet moves
/// that are not killers come by their score instead, best first.
pub fn order(
    moves: &[Move],
    killers: &[u32],
    counter: Option<u32>,
    history: Option<(&History, Color)>,
    continuation: Option<(&ContinuationHistory, usize)>,
) -> MoveList {
    let is_killer = |m: &Move| killers.contains(&move_id(m));
    let mut ordered = MoveList::new();
    // Captures: killers first, then the rest, and the whole group sorted by a stable sort.
    ordered.extend(
        moves
            .iter()
            .filter(|m| m.is_capture() && is_killer(m))
            .copied(),
    );
    ordered.extend(
        moves
            .iter()
            .filter(|m| m.is_capture() && !is_killer(m))
            .copied(),
    );
    let mut keys = [0i32; 256];
    for (key, m) in keys.iter_mut().zip(ordered.iter()) {
        *key = mvv_lva_score(m);
    }
    let captures = ordered.len();
    sort_descending(&mut ordered, &mut keys[..captures]);
    // Quiet moves: killers first, then the others (by history score when there is one).
    ordered.extend(
        moves
            .iter()
            .filter(|m| !m.is_capture() && is_killer(m))
            .copied(),
    );
    let others_start = ordered.len();
    ordered.extend(
        moves
            .iter()
            .filter(|m| !m.is_capture() && !is_killer(m))
            .copied(),
    );
    if let Some((history, color)) = history {
        let others = &mut ordered[others_start..];
        for (key, m) in keys.iter_mut().zip(others.iter()) {
            *key = history.score(color, m)
                + continuation.map_or(0, |(cont, previous)| cont.score(color, previous, m));
        }
        let count = others.len();
        sort_descending(others, &mut keys[..count]);
    }
    // The countermove goes right after the killers.
    if let Some(counter) = counter {
        move_to_front(&mut ordered[others_start..], Some(counter));
    }
    ordered
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
    use shakmaty::{Role, Square};

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
        let history = History::default();
        history.reward(Color::White, &c, 4);
        history.reward(Color::White, &b, 2);
        let ordered = order(
            &[a, b, c],
            &[move_id(&a)],
            None,
            Some((&history, Color::White)),
            None,
        );
        assert_eq!(ordered.as_slice(), &[a, c, b]);
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
            &[a, queen_takes_pawn, b, pawn_takes_queen],
            &[move_id(&b)],
            None,
            None,
            None,
        );
        assert_eq!(
            ordered.as_slice(),
            &[pawn_takes_queen, queen_takes_pawn, b, a]
        );
    }
}
