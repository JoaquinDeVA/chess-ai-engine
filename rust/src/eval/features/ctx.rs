//! The inputs of `pst_ctx` besides the pieces: the material of both sides, the pawns per file and
//! the bucket of each king, all one-hot. The entries are always the same; the families say which
//! of them a network has rows for, and with `king_buckets` the piece rows come once per bucket of
//! the viewer's own king. Not measured yet: `docs/RESULTS.md` has no SPRT for these networks.

use super::pieces::{input_index, PIECE_INPUTS};
use shakmaty::{Bitboard, Board, Chess, Color, File, Position, Role};

/// Entries of one side in the material block of `pst_ctx`: the counts of pawns (1 to 8), knights,
/// bishops, rooks (1 or 2 each) and queens (1).
const MATERIAL_PER_SIDE: usize = 15;
/// Entries of one side in the pawn-file block: one for a single pawn and one for two or more.
const PAWN_FILES_PER_SIDE: usize = 16;
/// Kinds of king bucket.
const KING_BUCKETS: usize = 4;
/// First index of the material block, the pawn-file block and the bucket block in the entries of
/// `pst_ctx` (the pieces come first).
const MATERIAL_BASE: usize = PIECE_INPUTS;
const PAWN_FILES_BASE: usize = MATERIAL_BASE + 2 * MATERIAL_PER_SIDE;
const KING_BASE: usize = PAWN_FILES_BASE + 2 * PAWN_FILES_PER_SIDE;
/// Roles that have a material entry, and the first entry of each in a side's material block.
pub(in crate::eval) const MATERIAL_ROLES: [Role; 5] = [
    Role::Pawn,
    Role::Knight,
    Role::Bishop,
    Role::Rook,
    Role::Queen,
];
const MATERIAL_FIRST: [usize; 5] = [0, 8, 10, 12, 14];
const MATERIAL_CAP: [usize; 5] = [8, 2, 2, 2, 1];

/// The families of `pst_ctx` a network reads besides the pieces; the model decides.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct Families {
    /// Piece counts of both sides.
    pub material: bool,
    /// Pawns per file of both sides.
    pub pawn_files: bool,
    /// Bucket of the viewer's own king: the piece rows come once per bucket.
    pub king_buckets: bool,
}

impl Families {
    /// The families called `names` in the saved models (`material`, `pawn_files`, `king_buckets`).
    pub fn from_names<S: AsRef<str>>(names: &[S]) -> Result<Families, String> {
        let mut families = Families::default();
        for name in names {
            match name.as_ref() {
                "material" => families.material = true,
                "pawn_files" => families.pawn_files = true,
                "king_buckets" => families.king_buckets = true,
                other => return Err(format!("unknown input family {other:?}")),
            }
        }
        Ok(families)
    }

    /// Rows of the piece block: 768 per king bucket.
    pub(in crate::eval) fn piece_rows(self) -> usize {
        PIECE_INPUTS * if self.king_buckets { KING_BUCKETS } else { 1 }
    }

    /// Rows of the first layer.
    pub fn rows(self) -> usize {
        self.piece_rows()
            + if self.material {
                2 * MATERIAL_PER_SIDE
            } else {
                0
            }
            + if self.pawn_files {
                2 * PAWN_FILES_PER_SIDE
            } else {
                0
            }
    }
}

/// Bucket of the king of `side`: from its own home rank, 0 castled short (f-h), 1 castled long
/// (a-c), 2 at home (d, e), 3 anywhere else.
pub(in crate::eval) fn king_bucket(board: &Board, side: Color) -> usize {
    let king = board.king_of(side).expect("a king per side");
    let oriented = if side == Color::Black {
        king.flip_vertical()
    } else {
        king
    };
    let square = u32::from(oriented) as usize;
    match square {
        5..=7 => 0,
        0..=2 => 1,
        3 | 4 => 2,
        _ => 3,
    }
}

/// Entry (in a side's material block) of `count` pieces of the role at position `at` of
/// [`MATERIAL_ROLES`], if there is any.
pub(in crate::eval) fn material_entry(at: usize, count: usize) -> Option<usize> {
    let count = count.min(MATERIAL_CAP[at]);
    (count > 0).then(|| MATERIAL_FIRST[at] + count - 1)
}

/// Entry (in a side's pawn-file block) of `count` pawns on `file`, if there is any.
pub(in crate::eval) fn pawn_entry(file: usize, count: usize) -> Option<usize> {
    (count > 0).then(|| 2 * file + usize::from(count > 1))
}

/// Pawns of `side` on `file`.
pub(in crate::eval) fn pawns_on(board: &Board, side: Color, file: usize) -> usize {
    (board.pawns() & board.by_color(side) & Bitboard::from(File::new(file as u32))).count()
}

/// Pieces of `role` of `side`.
pub(in crate::eval) fn count_of(board: &Board, side: Color, role: Role) -> usize {
    (board.by_role(role) & board.by_color(side)).count()
}

/// Entries of a side's material block that are active.
fn material_entries(board: &Board, side: Color) -> impl Iterator<Item = usize> + '_ {
    MATERIAL_ROLES
        .into_iter()
        .enumerate()
        .filter_map(move |(at, role)| material_entry(at, count_of(board, side, role)))
}

/// Entries of a side's pawn-file block that are active.
fn pawn_entries(board: &Board, side: Color) -> impl Iterator<Item = usize> + '_ {
    (0..8).filter_map(move |file| pawn_entry(file, pawns_on(board, side, file)))
}

/// The entries of `pst_ctx` after the pieces: for the mover and then its rival the material and
/// pawn-file entries, then the two king buckets.
pub(super) fn ctx_entries(pos: &Chess, indices: &mut Vec<usize>) {
    let board = pos.board();
    let sides = [pos.turn(), !pos.turn()];
    for (i, side) in sides.into_iter().enumerate() {
        indices.extend(
            material_entries(board, side).map(|e| MATERIAL_BASE + i * MATERIAL_PER_SIDE + e),
        );
        indices.extend(
            pawn_entries(board, side).map(|e| PAWN_FILES_BASE + i * PAWN_FILES_PER_SIDE + e),
        );
    }
    for (i, side) in sides.into_iter().enumerate() {
        indices.push(KING_BASE + i * KING_BUCKETS + king_bucket(board, side));
    }
}

/// Row of the first layer of the material entry `entry` of `side`, seen by `view`: the viewer's
/// own block first, then its rival's.
pub(in crate::eval) fn material_row(
    families: Families,
    view: Color,
    side: Color,
    entry: usize,
) -> usize {
    families.piece_rows() + if side == view { 0 } else { MATERIAL_PER_SIDE } + entry
}

/// Row of the pawn-file entry `entry` of `side`, seen by `view`.
pub(in crate::eval) fn pawn_row(
    families: Families,
    view: Color,
    side: Color,
    entry: usize,
) -> usize {
    let first = families.piece_rows()
        + if families.material {
            2 * MATERIAL_PER_SIDE
        } else {
            0
        };
    first + if side == view { 0 } else { PAWN_FILES_PER_SIDE } + entry
}

/// Rows of the first layer a `pst_ctx` network adds for the position as seen by `view`, as if
/// `view` were the side to move.
pub(in crate::eval) fn view_rows(
    families: Families,
    board: &Board,
    view: Color,
    rows: &mut Vec<usize>,
) {
    let offset = if families.king_buckets {
        king_bucket(board, view) * PIECE_INPUTS
    } else {
        0
    };
    for (square, piece) in board.clone() {
        rows.push(offset + input_index(view, piece.color, piece.role, square));
    }
    for side in [view, !view] {
        if families.material {
            rows.extend(
                material_entries(board, side).map(|e| material_row(families, view, side, e)),
            );
        }
        if families.pawn_files {
            rows.extend(pawn_entries(board, side).map(|e| pawn_row(families, view, side, e)));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::position_from_fen;
    use crate::eval::FeatureSet;

    #[test]
    fn the_ctx_entries_of_the_start_position() {
        let pos =
            position_from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1").unwrap();
        let (mut indices, mut values) = (Vec::new(), Vec::new());
        FeatureSet::from_name("pst_ctx")
            .unwrap()
            .active(&pos, &mut indices, &mut values);
        assert_eq!(indices.len(), 32 + 2 * (5 + 8) + 2);
        assert_eq!(values.len(), indices.len());
        // eight pawns, two of each minor piece and rook, one queen; the king at home
        assert_eq!(
            &indices[32..37],
            &[768 + 7, 768 + 9, 768 + 11, 768 + 13, 768 + 14]
        );
        assert_eq!(&indices[indices.len() - 2..], &[830 + 2, 830 + 4 + 2]);
    }

    #[test]
    fn the_family_names_and_sizes() {
        assert_eq!(Families::from_names::<&str>(&[]).unwrap().rows(), 768);
        let all = Families::from_names(&["material", "pawn_files", "king_buckets"]).unwrap();
        assert_eq!(all.rows(), 768 * 4 + 30 + 32);
        assert!(Families::from_names(&["nothing"]).is_err());
        assert!(FeatureSet::PstThreats.with_families(all).is_err());
        // the old name of the networks that read the pieces alone
        assert_eq!(
            FeatureSet::from_name("pst_slotted"),
            FeatureSet::from_name("pst_ctx")
        );
    }
}
