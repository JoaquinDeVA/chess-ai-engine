//! The accumulator of the network followed move by move during the search.

use shakmaty::{Board, Chess, Color, Move, Piece, Position, Role, Square};

use super::features::PIECE_INPUTS;
use super::net::Net;
use super::{for_player, sigmoid};

/// Index of a piece among the inputs of one king bucket when `mover` is the side to move: the
/// pieces of the mover come first, then the opponent's, each colour seen from its own home rank.
fn input_index(mover: Color, color: Color, role: Role, square: Square) -> usize {
    let mut index = (role as usize - 1) * 64;
    if color != mover {
        index += 384;
    }
    let square = if color == Color::Black {
        square.flip_vertical()
    } else {
        square
    };
    index + u32::from(square) as usize
}

fn slot(color: Color) -> usize {
    usize::from(color == Color::Black)
}

/// The accumulator of the first layer for both sides to move, updated move by move.
///
/// The inputs depend on who moves, so two accumulators are kept, one for each side as the mover; the
/// position is evaluated with the one of the side to move. Playing a move touches the two to four
/// pieces that change, instead of adding the rows of every piece again.
///
/// With king buckets each accumulator uses the rows of its mover's bucket: when that king moves to
/// another bucket, its accumulator is rebuilt from the board. The classical terms are not pieces:
/// they are computed and added at every evaluation.
///
/// The search calls `apply` for every move it plays and `undo` when it takes it back, and
/// `evaluate` for the position it is in; the result equals [`Net::evaluate`] up to rounding.
pub struct NetState<'a> {
    net: &'a Net,
    width: usize,
    acc: Vec<f64>,
    /// King bucket of each accumulator, by `slot` of its mover.
    buckets: [usize; 2],
    snapshots: Vec<f64>,
    bucket_snapshots: Vec<[usize; 2]>,
}

impl<'a> NetState<'a> {
    /// The accumulators of `pos`, built from the board.
    pub fn new(net: &'a Net, pos: &Chess) -> NetState<'a> {
        let width = net.layers[0].fan_out;
        let mut state = NetState {
            net,
            width,
            acc: vec![0.0; 2 * width],
            buckets: [0; 2],
            snapshots: Vec::with_capacity(2 * width * 64),
            bucket_snapshots: Vec::with_capacity(64),
        };
        for mover in [Color::White, Color::Black] {
            let king = pos.board().king_of(mover).expect("a king per side");
            state.buckets[slot(mover)] = net.features.king_bucket(mover, king);
            state.refresh(mover, pos.board());
        }
        state
    }

    /// The row of a piece in the accumulator of `mover`.
    fn row(&self, mover: Color, piece: Piece, square: Square) -> &'a [f64] {
        let first = &self.net.layers[0];
        let index = self.buckets[slot(mover)] * PIECE_INPUTS
            + input_index(mover, piece.color, piece.role, square);
        &first.weights[index * self.width..(index + 1) * self.width]
    }

    /// Rebuild the accumulator of `mover` from the bias and every piece of `board`.
    fn refresh(&mut self, mover: Color, board: &Board) {
        let start = slot(mover) * self.width;
        self.acc[start..start + self.width].copy_from_slice(&self.net.layers[0].bias);
        for (square, piece) in board.clone() {
            let row = self.row(mover, piece, square);
            for (a, w) in self.acc[start..start + self.width].iter_mut().zip(row) {
                *a += w;
            }
        }
    }

    /// Add (`sign` = 1) or remove (`sign` = -1) a piece from the accumulator of `mover`.
    fn toggle(&mut self, mover: Color, piece: Piece, square: Square, sign: f64) {
        let row = self.row(mover, piece, square);
        let start = slot(mover) * self.width;
        for (a, w) in self.acc[start..start + self.width].iter_mut().zip(row) {
            *a += sign * w;
        }
    }
}

/// The pieces a move takes off and puts on the board, and where the mover's king ends.
fn changes(mover: Color, m: &Move) -> ([(Piece, Square, f64); 4], usize, Option<Square>) {
    let own = |role| Piece { color: mover, role };
    let other = |role| Piece {
        color: !mover,
        role,
    };
    let none = (own(Role::Pawn), Square::A1, 0.0);
    match *m {
        Move::Normal {
            role,
            from,
            capture,
            to,
            promotion,
        } => {
            let taken = capture.map_or(none, |taken| (other(taken), to, -1.0));
            let king = (role == Role::King).then_some(to);
            (
                [
                    (own(role), from, -1.0),
                    (own(promotion.unwrap_or(role)), to, 1.0),
                    taken,
                    none,
                ],
                2 + usize::from(capture.is_some()),
                king,
            )
        }
        Move::EnPassant { from, to } => (
            [
                (own(Role::Pawn), from, -1.0),
                (own(Role::Pawn), to, 1.0),
                (
                    other(Role::Pawn),
                    Square::from_coords(to.file(), from.rank()),
                    -1.0,
                ),
                none,
            ],
            3,
            None,
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
                    (own(Role::King), king, -1.0),
                    (own(Role::King), king_to, 1.0),
                    (own(Role::Rook), rook, -1.0),
                    (
                        own(Role::Rook),
                        Square::from_coords(rook_file, rook.rank()),
                        1.0,
                    ),
                ],
                4,
                Some(king_to),
            )
        }
        Move::Put { .. } => unreachable!("there are no drops in chess"),
    }
}

impl NetState<'_> {
    /// The side to move in `pos` (the position before the move) plays `m`.
    pub fn apply(&mut self, pos: &Chess, m: &Move) {
        self.snapshots.extend_from_slice(&self.acc);
        self.bucket_snapshots.push(self.buckets);
        let mover = pos.turn();
        let (pieces, count, king) = changes(mover, m);
        // a king that enters another bucket rebuilds its own accumulator from the board
        let moved_bucket = king
            .map(|king| self.net.features.king_bucket(mover, king))
            .filter(|&bucket| bucket != self.buckets[slot(mover)]);
        for &(piece, square, sign) in &pieces[..count] {
            for side in [Color::White, Color::Black] {
                if side != mover || moved_bucket.is_none() {
                    self.toggle(side, piece, square, sign);
                }
            }
        }
        if let Some(bucket) = moved_bucket {
            self.buckets[slot(mover)] = bucket;
            let mut after = pos.clone();
            after.play_unchecked(*m);
            self.refresh(mover, after.board());
        }
    }

    /// The side to move passes (a null move).
    pub fn apply_null(&mut self) {
        // the accumulators of both sides are kept, so passing changes nothing
        self.snapshots.extend_from_slice(&self.acc);
        self.bucket_snapshots.push(self.buckets);
    }

    /// Take back the last move played with `apply` or `apply_null`.
    pub fn undo(&mut self) {
        let start = self.snapshots.len() - self.acc.len();
        self.acc.copy_from_slice(&self.snapshots[start..]);
        self.snapshots.truncate(start);
        self.buckets = self.bucket_snapshots.pop().expect("a move to undo");
    }

    /// Expected score of `player` in `pos`, the position the state is following.
    pub fn evaluate(&self, pos: &Chess, player: Color) -> f64 {
        let mover = pos.turn();
        let start = slot(mover) * self.width;
        let mut hidden = self.acc[start..start + self.width].to_vec();
        if self.net.features.has_terms() {
            let first = &self.net.layers[0];
            let (mut indices, mut values) = (Vec::with_capacity(32), Vec::with_capacity(32));
            self.net
                .features
                .term_inputs(pos, &mut indices, &mut values);
            for (index, value) in indices.iter().zip(&values) {
                let row = &first.weights[index * self.width..(index + 1) * self.width];
                for (h, w) in hidden.iter_mut().zip(row) {
                    *h += w * value;
                }
            }
        }
        for h in &mut hidden {
            *h = h.clamp(0.0, 1.0);
        }
        let score = sigmoid(self.net.finish(hidden));
        let white = if mover == Color::Black {
            1.0 - score
        } else {
            score
        };
        for_player(white, player)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::{ordered_moves, position_from_fen};
    use crate::eval::{FeatureSet, Layer};

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

    fn net(features: FeatureSet) -> Net {
        Net::new(
            features,
            vec![
                layer(features.size(), 16, 1),
                layer(16, 8, 2),
                layer(8, 1, 3),
            ],
        )
        .unwrap()
    }

    /// Plays random games, checking the incremental value against the full one after every move
    /// and again after taking every move back; returns the moves checked, the king moves that
    /// changed bucket and the castlings.
    fn check_incremental(features: FeatureSet) -> (usize, usize, usize) {
        let net = net(features);
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
        let (mut checked, mut bucket_changes, mut castles) = (0, 0, 0);
        for start in starts {
            for _ in 0..40 {
                let mut pos = position_from_fen(start).unwrap();
                let mut state = NetState::new(&net, &pos);
                let mut trail = Vec::new();
                for _ in 0..80 {
                    let moves = ordered_moves(&pos);
                    if moves.is_empty() {
                        break;
                    }
                    let m = moves[pick(moves.len())];
                    let mover = pos.turn();
                    let bucket =
                        |p: &Chess| features.king_bucket(mover, p.board().king_of(mover).unwrap());
                    let before = bucket(&pos);
                    castles += usize::from(matches!(m, Move::Castle { .. }));
                    state.apply(&pos, &m);
                    trail.push(pos.clone());
                    pos.play_unchecked(m);
                    bucket_changes += usize::from(bucket(&pos) != before);
                    for player in [Color::White, Color::Black] {
                        let full = net.evaluate(&pos, player);
                        assert!(
                            (state.evaluate(&pos, player) - full).abs() < 1e-12,
                            "{}",
                            crate::board::fen_of(&pos)
                        );
                    }
                    checked += 1;
                }
                while let Some(before) = trail.pop() {
                    state.undo();
                    pos = before;
                    for player in [Color::White, Color::Black] {
                        let full = net.evaluate(&pos, player);
                        assert!((state.evaluate(&pos, player) - full).abs() < 1e-12);
                    }
                }
            }
        }
        assert!(checked > 5000, "{checked}");
        assert!(castles > 0);
        (checked, bucket_changes, castles)
    }

    #[test]
    fn incremental_equals_full_slotted() {
        check_incremental(FeatureSet::Slotted);
    }

    #[test]
    fn incremental_equals_full_hybrid() {
        check_incremental(FeatureSet::Hybrid { file_groups: 0 });
    }

    #[test]
    fn incremental_equals_full_hybrid_k4() {
        let (_, bucket_changes, _) = check_incremental(FeatureSet::Hybrid { file_groups: 2 });
        assert!(bucket_changes > 100, "{bucket_changes}");
    }

    #[test]
    fn incremental_equals_full_hybrid_k8() {
        let (_, bucket_changes, _) = check_incremental(FeatureSet::Hybrid { file_groups: 4 });
        assert!(bucket_changes > 100, "{bucket_changes}");
    }

    #[test]
    fn incremental_equals_full_buckets_k4() {
        let (_, bucket_changes, _) = check_incremental(FeatureSet::Bucketed { file_groups: 2 });
        assert!(bucket_changes > 100, "{bucket_changes}");
    }

    /// A king that castles into another bucket rebuilds its accumulator, and undo restores the
    /// old one with its bucket.
    #[test]
    fn castling_across_buckets() {
        let features = FeatureSet::Hybrid { file_groups: 4 };
        let net = net(features);
        let pos = position_from_fen("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1").unwrap();
        for m in ordered_moves(&pos) {
            let Move::Castle { .. } = m else { continue };
            let mut state = NetState::new(&net, &pos);
            let mut after = pos.clone();
            after.play_unchecked(m);
            let king = |p: &Chess| {
                features.king_bucket(Color::White, p.board().king_of(Color::White).unwrap())
            };
            assert_ne!(king(&pos), king(&after));
            state.apply(&pos, &m);
            for player in [Color::White, Color::Black] {
                assert!(
                    (state.evaluate(&after, player) - net.evaluate(&after, player)).abs() < 1e-12
                );
            }
            state.undo();
            assert!(
                (state.evaluate(&pos, Color::White) - net.evaluate(&pos, Color::White)).abs()
                    < 1e-12
            );
        }
    }
}
