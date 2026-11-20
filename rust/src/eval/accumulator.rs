//! The accumulator of the network followed move by move during the search.

use shakmaty::{Board, Chess, Color, Move, Piece, Position, Role, Square};

use super::features::{
    count_of, input_index, king_bucket, material_entry, material_row, pawn_entry, pawn_row,
    pawns_on, view_rows, Families, FeatureSet, PIECE_INPUTS,
};
use super::net::Net;
use super::score;

fn slot(color: Color) -> usize {
    usize::from(color == Color::Black)
}

/// Plies the stack of accumulators holds before it has to grow: more than any search reaches.
const STACK_PLIES: usize = 128;

/// The accumulator of the first layer for both sides to move, updated move by move.
///
/// The inputs depend on who moves, so two accumulators are kept, one for each side as the mover; the
/// position is evaluated with the one of the side to move. Playing a move touches the two to four
/// pieces that change, instead of adding the rows of every piece again.
///
/// The accumulators of every ply of the line being searched are kept in a stack: a move copies the
/// top frame into the next one and changes the copy, and taking it back only steps down.
///
/// The dense inputs (the threats) are not pieces: they are computed and added at every evaluation.
///
/// The search calls `apply` for every move it plays and `undo` when it takes it back, and `score`
/// for the position it is in; the result equals [`Net::score`] up to the rounding of `f32` sums.
pub struct NetState<'a> {
    net: &'a Net,
    width: usize,
    /// The input families of a `pst_ctx` network, which `apply` follows with its own update.
    ctx: Option<Families>,
    /// One frame of `2 * width` sums per ply, the white mover's accumulator first.
    stack: Vec<f32>,
    /// Frame of the current position.
    top: usize,
    /// Buffers of an evaluation: the layers and the dense inputs.
    hidden: Vec<f32>,
    spare: Vec<f32>,
    dense_indices: Vec<usize>,
    dense_values: Vec<f64>,
}

impl<'a> NetState<'a> {
    /// The accumulators of `pos`, built from the board.
    pub fn new(net: &'a Net, pos: &Chess) -> NetState<'a> {
        let width = net.layers[0].fan_out;
        let ctx = match net.features {
            FeatureSet::PstCtx(families) => Some(families),
            _ => None,
        };
        let mut state = NetState {
            net,
            width,
            ctx,
            stack: vec![0.0; 2 * width * STACK_PLIES],
            top: 0,
            hidden: vec![0.0; net.max_width],
            spare: vec![0.0; net.max_width],
            dense_indices: Vec::with_capacity(8),
            dense_values: Vec::with_capacity(8),
        };
        for mover in [Color::White, Color::Black] {
            state.refresh(mover, pos.board());
        }
        state
    }

    /// The accumulator of `mover` in the current frame.
    fn acc(&mut self, mover: Color) -> &mut [f32] {
        let start = (2 * self.top + slot(mover)) * self.width;
        &mut self.stack[start..start + self.width]
    }

    /// The row of a piece in the accumulator of `mover`.
    fn row(&self, mover: Color, piece: Piece, square: Square) -> &'a [f32] {
        self.net.layers[0].row(input_index(mover, piece.color, piece.role, square))
    }

    /// Rebuild the accumulator of `mover` from the bias and every piece of `board`.
    fn refresh(&mut self, mover: Color, board: &Board) {
        let net = self.net;
        self.acc(mover).copy_from_slice(&net.layers[0].bias);
        if let Some(families) = self.ctx {
            let mut rows = Vec::with_capacity(64);
            view_rows(families, board, mover, &mut rows);
            for row in rows {
                for (a, w) in self.acc(mover).iter_mut().zip(net.layers[0].row(row)) {
                    *a += w;
                }
            }
            return;
        }
        for (square, piece) in board.clone() {
            let row = self.row(mover, piece, square);
            for (a, w) in self.acc(mover).iter_mut().zip(row) {
                *a += w;
            }
        }
    }

    /// Step up one frame, copying the current one into it.
    fn push(&mut self) {
        let frame = 2 * self.width;
        if (self.top + 2) * frame > self.stack.len() {
            self.stack.resize(self.stack.len() * 2, 0.0);
        }
        let start = self.top * frame;
        self.stack.copy_within(start..start + frame, start + frame);
        self.top += 1;
    }
}

/// The pieces a move takes off and puts on the board.
fn changes(mover: Color, m: &Move) -> ([(Piece, Square, bool); 4], usize) {
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

/// Adds to `lanes` the dot product of `acc` clipped to `[0, 1]` with `weights`, eight lanes at a
/// time (the length is a multiple of 8).
#[inline(always)]
fn clipped_dot(acc: &[f32], weights: &[f32], lanes: &mut [f32; 8]) {
    for (x, w) in acc
        .as_chunks::<8>()
        .0
        .iter()
        .zip(weights.as_chunks::<8>().0)
    {
        for lane in 0..8 {
            lanes[lane] += x[lane].clamp(0.0, 1.0) * w[lane];
        }
    }
}

impl NetState<'_> {
    /// The side to move in `pos` (the position before the move) plays `m`.
    pub fn apply(&mut self, pos: &Chess, m: &Move) {
        let mover = pos.turn();
        let (pieces, count) = changes(mover, m);
        if let Some(families) = self.ctx {
            return self.apply_ctx(families, pos, m, &pieces[..count]);
        }
        // step up one frame, writing the new accumulators straight from the old ones
        let frame = 2 * self.width;
        if (self.top + 2) * frame > self.stack.len() {
            self.stack.resize(self.stack.len() * 2, 0.0);
        }
        let width = self.width;
        let layer = &self.net.layers[0];
        {
            let (head, tail) = self.stack.split_at_mut((self.top + 1) * frame);
            let old = &head[self.top * frame..];
            let new = &mut tail[..frame];
            for side in [Color::White, Color::Black] {
                let at = slot(side) * width;
                let (o, n) = (&old[at..at + width], &mut new[at..at + width]);
                let mut rows = [o; 4];
                for (k, &(piece, square, _)) in pieces[..count].iter().enumerate() {
                    rows[k] =
                        &layer.row(input_index(side, piece.color, piece.role, square))[..width];
                }
                // the pieces alternate: take off, put on, take off, put on
                match count {
                    2 => {
                        for i in 0..width {
                            n[i] = o[i] - rows[0][i] + rows[1][i];
                        }
                    }
                    3 => {
                        for i in 0..width {
                            n[i] = o[i] - rows[0][i] + rows[1][i] - rows[2][i];
                        }
                    }
                    _ => {
                        for i in 0..width {
                            n[i] = o[i] - rows[0][i] + rows[1][i] - rows[2][i] + rows[3][i];
                        }
                    }
                }
            }
        }
        self.top += 1;
    }

    /// `apply` for a `pst_ctx` network. A view whose own king changes bucket is rebuilt from the
    /// board after the move (all its piece rows move); the others add the rows of the pieces that
    /// change, and of the material and pawn-file entries whose count changes.
    fn apply_ctx(
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
                    let old = count_of(board, side, super::features::MATERIAL_ROLES[at]);
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

    /// The side to move passes (a null move): the accumulators of both sides stay as they are.
    pub fn apply_null(&mut self) {
        self.push();
    }

    /// Take back the last move played with `apply` or `apply_null`.
    pub fn undo(&mut self) {
        self.top -= 1;
    }

    /// Network output before the logistic function, for the side to move in `pos`, the position
    /// the state is following.
    pub fn logit(&mut self, pos: &Chess) -> f32 {
        let net = self.net;
        let width = self.width;
        let start = (2 * self.top + slot(pos.turn())) * width;
        if net.layers.len() == 2 && width.is_multiple_of(8) {
            // one output layer straight over the accumulators: no intermediate buffer, except the
            // mover's own with the dense inputs added (they have no opponent view)
            let out = &net.layers[1];
            let mut lanes = [0.0f32; 8];
            if net.features.has_dense() {
                self.dense_indices.clear();
                self.dense_values.clear();
                net.features
                    .dense_inputs(pos, &mut self.dense_indices, &mut self.dense_values);
                let own = &mut self.hidden[..width];
                own.copy_from_slice(&self.stack[start..start + width]);
                for (&index, &value) in self.dense_indices.iter().zip(&self.dense_values) {
                    if value == 0.0 {
                        continue;
                    }
                    for (h, w) in own.iter_mut().zip(net.layers[0].row(index)) {
                        *h += w * value as f32;
                    }
                }
                clipped_dot(own, &out.weights[..width], &mut lanes);
            } else {
                clipped_dot(
                    &self.stack[start..start + width],
                    &out.weights[..width],
                    &mut lanes,
                );
            }
            let opponent = (2 * self.top + slot(!pos.turn())) * width;
            clipped_dot(
                &self.stack[opponent..opponent + width],
                &out.weights[width..2 * width],
                &mut lanes,
            );
            return out.bias[0] + lanes.iter().sum::<f32>();
        }
        self.hidden[..width].copy_from_slice(&self.stack[start..start + width]);
        // the opponent's accumulator follows the mover's: both views, one shared first layer
        let opponent = (2 * self.top + slot(!pos.turn())) * width;
        self.hidden[width..2 * width].copy_from_slice(&self.stack[opponent..opponent + width]);
        if net.features.has_dense() {
            self.dense_indices.clear();
            self.dense_values.clear();
            net.features
                .dense_inputs(pos, &mut self.dense_indices, &mut self.dense_values);
            for (&index, &value) in self.dense_indices.iter().zip(&self.dense_values) {
                for (h, w) in self.hidden.iter_mut().zip(net.layers[0].row(index)) {
                    *h += w * value as f32;
                }
            }
        }
        net.finish(&mut self.hidden, &mut self.spare)
    }

    /// Score of `player` in `pos`, the position the state is following.
    pub fn score(&mut self, pos: &Chess, player: Color) -> i32 {
        score(self.logit(pos), pos.turn(), player)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::{legal_moves, position_from_fen};
    use crate::eval::{FeatureSet, Layer};

    /// The incremental and the full sums add the same `f32` terms in another order.
    const TOLERANCE: f32 = 1e-4;

    fn layer(fan_in: usize, fan_out: usize, seed: u64) -> Layer {
        let mut state = seed;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) as f64 / (1u64 << 31) as f64) - 0.5
        };
        Layer {
            fan_in,
            fan_out,
            weights: (0..fan_in * fan_out).map(|_| next()).collect(),
            bias: (0..fan_out).map(|_| next() * 0.5).collect(),
        }
    }

    /// A network whose second layer reads both sides' accumulators (twice as wide as one).
    fn net(features: FeatureSet) -> Net {
        Net::new(
            features,
            vec![
                layer(features.size(), 16, 1),
                layer(32, 8, 2),
                layer(8, 1, 3),
            ],
        )
        .unwrap()
    }

    /// Plays random games, checking the incremental value against the full one after every move
    /// and again after taking every move back; returns the moves checked and the castlings.
    fn check_net(net: &Net) -> (usize, usize) {
        let mut seed = 12345u64;
        let mut pick = |n: usize| {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (seed >> 33) as usize % n
        };
        let starts = [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "r3k2r/pppq1ppp/2npbn2/2b1p3/2B1P3/2NP1N2/PPPQ1PPP/R3K2R w KQkq - 0 1",
            "r3k2r/1P4P1/8/3pP3/8/8/1p4p1/R3K2R w KQkq d6 0 1",
        ];
        let (mut checked, mut castles) = (0, 0);
        for start in starts {
            for _ in 0..40 {
                let mut pos = position_from_fen(start).unwrap();
                let mut state = NetState::new(net, &pos);
                let mut trail = Vec::new();
                for _ in 0..80 {
                    let moves = legal_moves(&pos);
                    if moves.is_empty() {
                        break;
                    }
                    let m = moves[pick(moves.len())];
                    castles += usize::from(matches!(m, Move::Castle { .. }));
                    state.apply(&pos, &m);
                    trail.push(pos.clone());
                    pos.play_unchecked(m);
                    assert!(
                        (state.logit(&pos) - net.logit(&pos)).abs() < TOLERANCE,
                        "{}",
                        crate::board::fen_of(&pos)
                    );
                    checked += 1;
                }
                while let Some(before) = trail.pop() {
                    state.undo();
                    pos = before;
                    assert!((state.logit(&pos) - net.logit(&pos)).abs() < TOLERANCE);
                }
            }
        }
        assert!(checked > 5000, "{checked}");
        assert!(castles > 0);
        (checked, castles)
    }

    #[test]
    fn incremental_equals_full_threats() {
        check_net(&net(FeatureSet::PstThreats));
    }

    /// Every combination of the families, alone and together.
    fn ctx_nets() -> Vec<Net> {
        let mut nets = Vec::new();
        for bits in 0..8u8 {
            let families = Families {
                material: bits & 1 != 0,
                pawn_files: bits & 2 != 0,
                king_buckets: bits & 4 != 0,
            };
            let features = FeatureSet::PstCtx(families);
            nets.push(net(features));
        }
        nets
    }

    #[test]
    fn incremental_equals_full_ctx() {
        for net in ctx_nets() {
            check_net(&net);
        }
    }
}
