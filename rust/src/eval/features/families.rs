//! The families of inputs a `pst_ctx` network reads besides the pieces, and so the number of rows
//! of its first layer. The model decides: the file of a network lists its families.

use super::pieces::PIECE_INPUTS;

/// Entries of one side in the material block of `pst_ctx`: the counts of pawns (1 to 8), knights,
/// bishops, rooks (1 or 2 each) and queens (1).
pub(super) const MATERIAL_PER_SIDE: usize = 15;
/// Entries of one side in the pawn-file block: one for a single pawn and one for two or more.
pub(super) const PAWN_FILES_PER_SIDE: usize = 16;
/// Kinds of king bucket.
pub(super) const KING_BUCKETS: usize = 4;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_family_names_and_sizes() {
        assert_eq!(Families::from_names::<&str>(&[]).unwrap().rows(), 768);
        let all = Families::from_names(&["material", "pawn_files", "king_buckets"]).unwrap();
        assert_eq!(all.rows(), 768 * 4 + 30 + 32);
        assert!(Families::from_names(&["nothing"]).is_err());
    }
}
