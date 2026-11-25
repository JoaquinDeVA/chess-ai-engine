//! The active inputs of a position as the laboratory stores them in its caches: one entry per
//! piece, then the entries of the set, in the index space of `chess_lab/models/features.py` (which
//! has the 8 king-bucket entries after the 830 rows of `pst_ctx`, whatever families a network
//! reads). It is the one encoder of the laboratory; `features.py` stays as its slow reference in
//! the tests.

use shakmaty::{Chess, Position};

use super::ctx::{king_bucket_at, material_entries, pawn_entries};
use super::families::{KING_BUCKETS, MATERIAL_PER_SIDE, PAWN_FILES_PER_SIDE};
use super::pieces::{input_index, PIECE_INPUTS};
use super::threats::threats;

/// First index of the pawn-file entries and of the king-bucket ones in `pst_ctx`.
const PAWN_FILES_BASE: usize = PIECE_INPUTS + 2 * MATERIAL_PER_SIDE;
const KING_BASE: usize = PAWN_FILES_BASE + 2 * PAWN_FILES_PER_SIDE;

/// The input sets of the laboratory, by the name of its saved models.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Encoding {
    /// `pst_slotted`: the pieces alone.
    Slotted,
    /// `pst_threats`: the pieces and five dense threat entries.
    Threats,
    /// `pst_ctx`: the pieces, material, pawn files and king buckets.
    Ctx,
}

impl Encoding {
    /// The set called `name` in the saved models.
    pub fn from_name(name: &str) -> Option<Encoding> {
        Some(match name {
            "pst_slotted" => Encoding::Slotted,
            "pst_threats" => Encoding::Threats,
            "pst_ctx" => Encoding::Ctx,
            _ => return None,
        })
    }
}

/// Appends the active entries of `pos` (index and value) to `indices` and `values`, in the order of
/// `features.py`: the pieces from the highest square down (as `piece_map` lists them), then the entries of the set.
pub fn encode(encoding: Encoding, pos: &Chess, indices: &mut Vec<i64>, values: &mut Vec<f32>) {
    let mover = pos.turn();
    let board = pos.board();
    let mut push = |index: usize, value: f32| {
        indices.push(index as i64);
        values.push(value);
    };
    for square in board.occupied().into_iter().rev() {
        let piece = board.piece_at(square).expect("occupied");
        push(input_index(mover, piece.color, piece.role, square), 1.0);
    }
    match encoding {
        Encoding::Slotted => {}
        Encoding::Threats => {
            for (at, value) in threats(pos).into_iter().enumerate() {
                push(PIECE_INPUTS + at, value);
            }
        }
        Encoding::Ctx => {
            for (side, color) in [mover, !mover].into_iter().enumerate() {
                for entry in material_entries(board, color) {
                    push(PIECE_INPUTS + side * MATERIAL_PER_SIDE + entry, 1.0);
                }
                for entry in pawn_entries(board, color) {
                    push(PAWN_FILES_BASE + side * PAWN_FILES_PER_SIDE + entry, 1.0);
                }
            }
            for (side, color) in [mover, !mover].into_iter().enumerate() {
                let king = board.king_of(color).expect("a king per side");
                push(
                    KING_BASE + side * KING_BUCKETS + king_bucket_at(king, color),
                    1.0,
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::position_from_fen;

    fn encoded(encoding: Encoding, fen: &str) -> (Vec<i64>, Vec<f32>) {
        let (mut indices, mut values) = (Vec::new(), Vec::new());
        encode(
            encoding,
            &position_from_fen(fen).unwrap(),
            &mut indices,
            &mut values,
        );
        (indices, values)
    }

    const START: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

    #[test]
    fn the_start_position_has_a_piece_entry_per_piece() {
        let (indices, values) = encoded(Encoding::Slotted, START);
        assert_eq!(indices.len(), 32);
        assert!(values.iter().all(|&v| v == 1.0));
        // h8 rook of the rival comes first; a1 rook of the mover is the last
        assert_eq!(indices[0], 384 + 3 * 64 + 7);
        assert_eq!(indices[31], 3 * 64);
    }

    #[test]
    fn the_extra_entries_follow_the_pieces() {
        let (threats, values) = encoded(Encoding::Threats, START);
        assert_eq!(threats[32..], [768, 769, 770, 771, 772]);
        assert_eq!(values[32..], [0.0; 5]);
        let (ctx, _) = encoded(Encoding::Ctx, START);
        // 5 material and 8 pawn-file entries per side, then the two king buckets (at home)
        assert_eq!(ctx.len(), 32 + 2 * 5 + 2 * 8 + 2);
        assert_eq!(ctx[ctx.len() - 2..], [830 + 2, 830 + 4 + 2]);
    }

    #[test]
    fn the_names_of_the_sets() {
        assert_eq!(Encoding::from_name("pst_ctx"), Some(Encoding::Ctx));
        assert_eq!(Encoding::from_name("nothing"), None);
    }
}
