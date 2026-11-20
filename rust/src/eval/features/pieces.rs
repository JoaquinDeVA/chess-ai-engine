//! The piece inputs, the base of every network: one entry per piece, color and square, the mover's
//! pieces in the first block and the opponent's in the second, each color seen from its own home
//! rank. They are the inputs of the networks saved as `pst_slotted` (`net_pst_w64_both` and its
//! wider versions) and the first entries of the other two sets.

use shakmaty::{Chess, Color, Piece, Position, Role, Square};

pub(in crate::eval) const OPPONENT_PIECE_BLOCK: usize = 6 * 64;
/// Piece inputs: one per piece, color and square.
pub const PIECE_INPUTS: usize = 2 * OPPONENT_PIECE_BLOCK;

/// Index of a piece among the piece inputs when `mover` is the side to move: the pieces of the
/// mover come first, then the opponent's, each color seen from its own home rank.
pub(in crate::eval) fn input_index(
    mover: Color,
    color: Color,
    role: Role,
    square: Square,
) -> usize {
    let mut index = (role as usize - 1) * 64;
    if color != mover {
        index += OPPONENT_PIECE_BLOCK;
    }
    let square = if color == Color::Black {
        square.flip_vertical()
    } else {
        square
    };
    index + u32::from(square) as usize
}

/// Pieces of the board from the highest square to the lowest, as `python-chess` lists them.
fn pieces(pos: &Chess) -> impl Iterator<Item = (Square, Piece)> + '_ {
    let board = pos.board();
    board
        .occupied()
        .into_iter()
        .rev()
        .map(move |sq| (sq, board.piece_at(sq).expect("occupied")))
}

fn mirror(sq: Square) -> usize {
    u32::from(sq.flip_vertical()) as usize
}

/// One entry per piece: the mover's pieces in the first block, the opponent's in the second, each
/// color seen from its own home rank.
pub(super) fn slotted(pos: &Chess, indices: &mut Vec<usize>, values: &mut Vec<f64>) {
    let mover = pos.turn();
    for (square, piece) in pieces(pos) {
        let mut offset = (piece.role as usize - 1) * 64;
        if piece.color != mover {
            offset += OPPONENT_PIECE_BLOCK;
        }
        let oriented = if piece.color == Color::Black {
            mirror(square)
        } else {
            u32::from(square) as usize
        };
        indices.push(offset + oriented);
    }
    values.resize(indices.len(), 1.0);
}
