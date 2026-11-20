//! The update of the accumulator of a `pst_ctx` network. Besides the pieces that change, a move
//! changes the material and pawn-file entries whose count changes, and the whole view of a side
//! whose own king changes bucket.

use shakmaty::{Chess, Color, Move, Piece, Position, Role, Square};

use super::{add_row, sign, NetState};
use crate::eval::features::{entry_edits, input_index, king_bucket_at, piece_offset, Families};

impl NetState<'_> {
    /// `apply` for a `pst_ctx` network. A view whose own king changes bucket is rebuilt from the
    /// board after the move (all its piece rows move); the others add the rows of the pieces that
    /// change, and of the material and pawn-file entries whose count changes.
    pub(super) fn apply_ctx(
        &mut self,
        families: Families,
        pos: &Chess,
        m: &Move,
        pieces: &[(Piece, Square, bool)],
    ) {
        let mover = pos.turn();
        let board = pos.board();
        self.push();
        let mut rebuilt = false;
        if families.king_buckets {
            let king = |put| pieces.iter().find(|p| p.0.role == Role::King && p.2 == put);
            if let (Some(&(_, from, _)), Some(&(_, to, _))) = (king(false), king(true)) {
                if king_bucket_at(from, mover) != king_bucket_at(to, mover) {
                    let mut after = pos.clone();
                    after.play_unchecked(*m);
                    self.refresh(mover, after.board());
                    rebuilt = true;
                }
            }
        }
        let edits = entry_edits(families, board, pieces);
        let net = self.net;
        for view in [Color::White, Color::Black] {
            if rebuilt && view == mover {
                continue;
            }
            let offset = piece_offset(families, board, view);
            for &(piece, square, put) in pieces {
                let row = offset + input_index(view, piece.color, piece.role, square);
                add_row(self.acc(view), net.first_row(row), sign(put));
            }
            for edit in &edits {
                add_row(
                    self.acc(view),
                    net.first_row(edit.row(families, view)),
                    edit.sign,
                );
            }
        }
    }
}
