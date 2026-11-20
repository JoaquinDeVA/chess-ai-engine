//! The accumulator of the network followed move by move during the search. `mod.rs` keeps the
//! stack of accumulators and the update for the pieces; [`changes`] says which pieces a move
//! moves, [`ctx`] has the extra update of `pst_ctx` and [`logit`] the output.

mod changes;
mod ctx;
mod logit;

pub use logit::EVAL_LIMIT;

use shakmaty::{Board, Chess, Color, Move, Position};

use super::features::{input_index, view_rows, Families, FeatureSet};
use super::net::Net;
use changes::changes;

fn slot(color: Color) -> usize {
    usize::from(color == Color::Black)
}

/// Adds `row` to `acc`, or takes it off when `sign` is -1.
fn add_row(acc: &mut [f32], row: &[f32], sign: f32) {
    for (a, w) in acc.iter_mut().zip(row) {
        *a += sign * w;
    }
}

/// 1 for a piece put on a square, -1 for a piece taken off it.
fn sign(put: bool) -> f32 {
    if put {
        1.0
    } else {
        -1.0
    }
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
/// for the position it is in; the result equals [`Net::logit`] scaled by `score` up to the rounding of `f32` sums.
pub struct NetState<'a> {
    net: &'a Net,
    width: usize,
    /// The input families of a `pst_ctx` network, which `apply` follows with its own update.
    families: Option<Families>,
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
        let width = net.width();
        let families = match net.features() {
            FeatureSet::PstCtx(families) => Some(families),
            _ => None,
        };
        let mut state = NetState {
            net,
            width,
            families,
            stack: vec![0.0; 2 * width * STACK_PLIES],
            top: 0,
            hidden: vec![0.0; net.buffer_len()],
            spare: vec![0.0; net.buffer_len()],
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

    /// Rebuild the accumulator of `mover` from the bias and every piece of `board`.
    fn refresh(&mut self, mover: Color, board: &Board) {
        let net = self.net;
        self.acc(mover).copy_from_slice(net.first_bias());
        if let Some(families) = self.families {
            let mut rows = Vec::with_capacity(64);
            view_rows(families, board, mover, &mut rows);
            for row in rows {
                add_row(self.acc(mover), net.first_row(row), 1.0);
            }
            return;
        }
        for (square, piece) in board.clone() {
            let row = input_index(mover, piece.color, piece.role, square);
            add_row(self.acc(mover), net.first_row(row), 1.0);
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

impl NetState<'_> {
    /// The side to move in `pos` (the position before the move) plays `m`.
    pub fn apply(&mut self, pos: &Chess, m: &Move) {
        let (pieces, count) = changes(pos.turn(), m);
        let pieces = &pieces[..count];
        if let Some(families) = self.families {
            return self.apply_ctx(families, pos, m, pieces);
        }
        self.push();
        let net = self.net;
        for side in [Color::White, Color::Black] {
            for &(piece, square, put) in pieces {
                let row = input_index(side, piece.color, piece.role, square);
                add_row(self.acc(side), net.first_row(row), sign(put));
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
            (((state >> 33) as f64 / (1u64 << 31) as f64) - 0.5) as f32
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
                        (state.logit(&pos) - NetState::new(net, &pos).logit(&pos)).abs()
                            < TOLERANCE,
                        "{}",
                        crate::board::fen_of(&pos)
                    );
                    checked += 1;
                }
                while let Some(before) = trail.pop() {
                    state.undo();
                    pos = before;
                    assert!(
                        (state.logit(&pos) - NetState::new(net, &pos).logit(&pos)).abs()
                            < TOLERANCE
                    );
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
