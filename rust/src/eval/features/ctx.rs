//! The entries of `pst_ctx` besides the pieces: the material of both sides, the pawns per file and
//! the bucket of each king, all one-hot, and the rows of the first layer they read. Which of them a
//! network has rows for is [`Families`](super::Families). Measured in play: `docs/RESULTS.md`
//! («Network matches»).

use super::families::{Families, MATERIAL_PER_SIDE, PAWN_FILES_PER_SIDE};
use super::pieces::{input_index, PIECE_INPUTS};
use shakmaty::{Bitboard, Board, Color, File, Piece, Role, Square};

/// Roles that have a material entry, and the first entry of each in a side's material block.
const MATERIAL_ROLES: [Role; 5] = [
    Role::Pawn,
    Role::Knight,
    Role::Bishop,
    Role::Rook,
    Role::Queen,
];
const MATERIAL_FIRST: [usize; 5] = [0, 8, 10, 12, 14];
const MATERIAL_CAP: [usize; 5] = [8, 2, 2, 2, 1];

/// Bucket of the king of `side` standing on `king`: from its own home rank, 0 castled short
/// (f-h), 1 castled long (a-c), 2 at home (d, e), 3 anywhere else.
pub(in crate::eval) fn king_bucket_at(king: Square, side: Color) -> usize {
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

/// First row of the piece block that `view` reads: the piece rows come once per bucket of its own
/// king when the family `king_buckets` is on.
pub(in crate::eval) fn piece_offset(families: Families, board: &Board, view: Color) -> usize {
    if families.king_buckets {
        king_bucket_at(board.king_of(view).expect("a king per side"), view) * PIECE_INPUTS
    } else {
        0
    }
}

/// Entry (in a side's material block) of `count` pieces of the role at position `at` of
/// [`MATERIAL_ROLES`], if there is any.
fn material_entry(at: usize, count: usize) -> Option<usize> {
    let count = count.min(MATERIAL_CAP[at]);
    (count > 0).then(|| MATERIAL_FIRST[at] + count - 1)
}

/// Entry (in a side's pawn-file block) of `count` pawns on `file`, if there is any.
fn pawn_entry(file: usize, count: usize) -> Option<usize> {
    (count > 0).then(|| 2 * file + usize::from(count > 1))
}

/// Pawns of `side` on `file`.
fn pawns_on(board: &Board, side: Color, file: usize) -> usize {
    (board.pawns() & board.by_color(side) & Bitboard::from(File::new(file as u32))).count()
}

/// Pieces of `role` of `side`.
fn count_of(board: &Board, side: Color, role: Role) -> usize {
    (board.by_role(role) & board.by_color(side)).count()
}

/// Entries of a side's material block that are active.
pub(super) fn material_entries(board: &Board, side: Color) -> impl Iterator<Item = usize> + '_ {
    MATERIAL_ROLES
        .into_iter()
        .enumerate()
        .filter_map(move |(at, role)| material_entry(at, count_of(board, side, role)))
}

/// Entries of a side's pawn-file block that are active.
pub(super) fn pawn_entries(board: &Board, side: Color) -> impl Iterator<Item = usize> + '_ {
    (0..8).filter_map(move |file| pawn_entry(file, pawns_on(board, side, file)))
}

/// Row of the first layer of the material entry `entry` of `side`, seen by `view`: the viewer's
/// own block first, then its rival's.
fn material_row(families: Families, view: Color, side: Color, entry: usize) -> usize {
    families.material_start() + if side == view { 0 } else { MATERIAL_PER_SIDE } + entry
}

/// Row of the pawn-file entry `entry` of `side`, seen by `view`.
fn pawn_row(families: Families, view: Color, side: Color, entry: usize) -> usize {
    families.pawn_start() + if side == view { 0 } else { PAWN_FILES_PER_SIDE } + entry
}

/// A change of an entry of `pst_ctx` that a move causes: the entry of `side` (of the material
/// block, or of the pawn-file one) that is on or off, `sign` being 1 or -1. Every move fills an
/// array of them, so they are kept small: the entry fits in a byte (with a `usize` the array cost
/// 2 % of the nodes per second of `uci bench`, which moves without entries).
#[derive(Clone, Copy)]
pub(in crate::eval) struct EntryEdit {
    side: Color,
    material: bool,
    entry: u8,
    pub(in crate::eval) sign: f32,
}

/// Most entry changes a move can cause: at most four pieces that are not kings change (a
/// castling moves a rook, a capture with promotion takes a pawn off, puts a piece on and takes the
/// captured one off), each changing the entry of its role and, a pawn, that of its file, and each
/// change turns one entry off and another on.
const MAX_EDITS: usize = 16;

impl EntryEdit {
    const NONE: EntryEdit = EntryEdit {
        side: Color::White,
        material: true,
        entry: 0,
        sign: 0.0,
    };

    /// The row of the first layer of the entry, seen by `view`.
    pub(in crate::eval) fn row(&self, families: Families, view: Color) -> usize {
        if self.material {
            material_row(families, view, self.side, usize::from(self.entry))
        } else {
            pawn_row(families, view, self.side, usize::from(self.entry))
        }
    }
}

/// The entries that turn on or off when `pieces` (piece, square and whether it is put, as
/// the accumulator lists them) change `board`, the position before the move.
/// Only the counts that change an entry matter, and only for the families the network has. The
/// edits are the first ones of the array, as many as the count says.
pub(in crate::eval) fn entry_edits(
    families: Families,
    board: &Board,
    pieces: &[(Piece, Square, bool)],
) -> ([EntryEdit; MAX_EDITS], usize) {
    let mut edits = [EntryEdit::NONE; MAX_EDITS];
    let mut count = 0;
    if !families.material && !families.pawn_files {
        return (edits, count);
    }
    let at = |side: Color| usize::from(side == Color::Black);
    // count changes: [color][role - 1] for the material, [color][file] for the pawns
    let (mut by_role, mut by_file) = ([[0i32; 5]; 2], [[0i32; 8]; 2]);
    for &(piece, square, put) in pieces {
        if piece.role == Role::King {
            continue;
        }
        let delta = if put { 1 } else { -1 };
        by_role[at(piece.color)][piece.role as usize - 1] += delta;
        if piece.role == Role::Pawn {
            by_file[at(piece.color)][square.file() as usize] += delta;
        }
    }
    let mut change = |side, material, old: Option<usize>, new: Option<usize>| {
        if old != new {
            for (entry, sign) in [(old, -1.0), (new, 1.0)] {
                if let Some(entry) = entry {
                    edits[count] = EntryEdit {
                        side,
                        material,
                        entry: entry as u8,
                        sign,
                    };
                    count += 1;
                }
            }
        }
    };
    for side in [Color::White, Color::Black] {
        for (role, &delta) in by_role[at(side)].iter().enumerate() {
            if delta != 0 && families.material {
                let old = count_of(board, side, MATERIAL_ROLES[role]);
                let new = (old as i32 + delta) as usize;
                change(
                    side,
                    true,
                    material_entry(role, old),
                    material_entry(role, new),
                );
            }
        }
        for (file, &delta) in by_file[at(side)].iter().enumerate() {
            if delta != 0 && families.pawn_files {
                let old = pawns_on(board, side, file);
                let new = (old as i32 + delta) as usize;
                change(side, false, pawn_entry(file, old), pawn_entry(file, new));
            }
        }
    }
    (edits, count)
}

/// Calls `add` with every row of the first layer that the position adds as seen by `view`, as if
/// `view` were the side to move: the pieces, then the entries of the families the network has.
pub(in crate::eval) fn view_rows(
    families: Families,
    board: &Board,
    view: Color,
    mut add: impl FnMut(usize),
) {
    let offset = piece_offset(families, board, view);
    for (square, piece) in board.clone() {
        add(offset + input_index(view, piece.color, piece.role, square));
    }
    for side in [view, !view] {
        if families.material {
            material_entries(board, side).for_each(|e| add(material_row(families, view, side, e)));
        }
        if families.pawn_files {
            pawn_entries(board, side).for_each(|e| add(pawn_row(families, view, side, e)));
        }
    }
}
