//! The piece inputs, the base of every network: one entry per piece, color and square, the mover's
//! pieces in the first block and the opponent's in the second, each color seen from its own home
//! rank. They are the inputs of the networks saved as `pst_slotted` (`net_pst_w64_both` and its
//! wider versions) and the first entries of the other two sets.

use shakmaty::{Color, Role, Square};

const OPPONENT_PIECE_BLOCK: usize = 6 * 64;
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
