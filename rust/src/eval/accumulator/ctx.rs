//! The update of the accumulator of a `pst_ctx` network. Besides the pieces that change, a move
//! changes the material and pawn-file entries whose count changes, and the whole view of a side
//! whose own king changes bucket.

use shakmaty::{Board, Chess, Color, Move, Piece, Position, Role, Square};

use super::{slot, NetState};
use crate::eval::features::{
    count_of, input_index, king_bucket, material_entry, material_row, pawn_entry, pawn_row,
    pawns_on, Families, MATERIAL_ROLES, PIECE_INPUTS,
};

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
                let bucket = |square: Square| {
                    let mut board = Board::empty();
                    board.set_piece_at(
                        square,
                        Piece {
                            color: mover,
                            role: Role::King,
                        },
                    );
                    king_bucket(&board, mover)
                };
                if bucket(from) != bucket(to) {
                    let mut after = pos.clone();
                    after.play_unchecked(*m);
                    self.refresh(mover, after.board());
                    rebuilt = true;
                }
            }
        }
        // count changes: [color][role - 1] for the material, [color][file] for the pawns
        let (mut by_role, mut by_file) = ([[0i32; 5]; 2], [[0i32; 8]; 2]);
        for &(piece, square, put) in pieces {
            if piece.role == Role::King {
                continue;
            }
            let delta = if put { 1 } else { -1 };
            by_role[slot(piece.color)][piece.role as usize - 1] += delta;
            if piece.role == Role::Pawn {
                by_file[slot(piece.color)][square.file() as usize] += delta;
            }
        }
        // (side, material or pawn file, entry, sign)
        let mut edits: Vec<(Color, bool, usize, f32)> = Vec::with_capacity(16);
        for side in [Color::White, Color::Black] {
            for (at, &delta) in by_role[slot(side)].iter().enumerate() {
                if delta != 0 && families.material {
                    let old = count_of(board, side, MATERIAL_ROLES[at]);
                    let new = (old as i32 + delta) as usize;
                    if material_entry(at, old) != material_entry(at, new) {
                        edits.extend(material_entry(at, old).map(|e| (side, true, e, -1.0)));
                        edits.extend(material_entry(at, new).map(|e| (side, true, e, 1.0)));
                    }
                }
            }
            for (file, &delta) in by_file[slot(side)].iter().enumerate() {
                if delta != 0 && families.pawn_files {
                    let old = pawns_on(board, side, file);
                    let new = (old as i32 + delta) as usize;
                    if pawn_entry(file, old) != pawn_entry(file, new) {
                        edits.extend(pawn_entry(file, old).map(|e| (side, false, e, -1.0)));
                        edits.extend(pawn_entry(file, new).map(|e| (side, false, e, 1.0)));
                    }
                }
            }
        }
        let net = self.net;
        for view in [Color::White, Color::Black] {
            if rebuilt && view == mover {
                continue;
            }
            let offset = if families.king_buckets {
                king_bucket(board, view) * PIECE_INPUTS
            } else {
                0
            };
            let mut rows: Vec<(usize, f32)> = Vec::with_capacity(24);
            for &(piece, square, put) in pieces {
                let row = offset + input_index(view, piece.color, piece.role, square);
                rows.push((row, if put { 1.0 } else { -1.0 }));
            }
            for &(side, material, entry, sign) in &edits {
                let row = if material {
                    material_row(families, view, side, entry)
                } else {
                    pawn_row(families, view, side, entry)
                };
                rows.push((row, sign));
            }
            let acc = self.acc(view);
            for (row, sign) in rows {
                for (a, w) in acc.iter_mut().zip(net.layers[0].row(row)) {
                    *a += sign * w;
                }
            }
        }
    }
}
