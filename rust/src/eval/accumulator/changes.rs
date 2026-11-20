//! The pieces a move takes off and puts on the board, which the accumulator follows.

use shakmaty::{Color, Move, Piece, Role, Square};

/// The pieces a move takes off and puts on the board.
pub(super) fn changes(mover: Color, m: &Move) -> ([(Piece, Square, bool); 4], usize) {
    let own = |role| Piece { color: mover, role };
    let other = |role| Piece {
        color: !mover,
        role,
    };
    let none = (own(Role::Pawn), Square::A1, false);
    match *m {
        Move::Normal {
            role,
            from,
            capture,
            to,
            promotion,
        } => {
            let taken = capture.map_or(none, |taken| (other(taken), to, false));
            (
                [
                    (own(role), from, false),
                    (own(promotion.unwrap_or(role)), to, true),
                    taken,
                    none,
                ],
                2 + usize::from(capture.is_some()),
            )
        }
        Move::EnPassant { from, to } => (
            [
                (own(Role::Pawn), from, false),
                (own(Role::Pawn), to, true),
                (
                    other(Role::Pawn),
                    Square::from_coords(to.file(), from.rank()),
                    false,
                ),
                none,
            ],
            3,
        ),
        Move::Castle { king, rook } => {
            let kingside = rook.file() > king.file();
            let (king_file, rook_file) = if kingside {
                (shakmaty::File::G, shakmaty::File::F)
            } else {
                (shakmaty::File::C, shakmaty::File::D)
            };
            let king_to = Square::from_coords(king_file, king.rank());
            (
                [
                    (own(Role::King), king, false),
                    (own(Role::King), king_to, true),
                    (own(Role::Rook), rook, false),
                    (
                        own(Role::Rook),
                        Square::from_coords(rook_file, rook.rank()),
                        true,
                    ),
                ],
                4,
            )
        }
        Move::Put { .. } => unreachable!("there are no drops in chess"),
    }
}
