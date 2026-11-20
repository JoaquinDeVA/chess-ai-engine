//! The five dense threat inputs of `pst_threats`: the best capture of each side, the pieces en prise
//! of each side and whether the mover is in check. They are computed from the board at every
//! evaluation and only reach the mover's half of the accumulator. They are the inputs of the model
//! that plays, `net_pst_threats_w64_both`: `docs/RESULTS.md` has the SPRT that decided it
//! (`sp_threats_both_vs_both_opt`).

use crate::board::piece_value;
use shakmaty::attacks::{bishop_attacks, king_attacks, knight_attacks, rook_attacks};
use shakmaty::{Bitboard, Board, Chess, Color, File, Position};

/// Dense threat inputs: two best capture gains, two en-prise counts and the check flag.
pub(super) const THREAT_INPUTS: usize = 5;
const GAIN_SCALE: f64 = 9.0;
const EN_PRISE_SCALE: f64 = 4.0;

/// Every square that some piece of `side` attacks, own pieces included (so also the squares it
/// defends). Pawns by shifting the whole set, each other role by its attack table.
fn attacked_by(board: &Board, side: Color) -> Bitboard {
    let occupied = board.occupied();
    let mine = board.by_color(side);
    let pawns = (board.pawns() & mine).0;
    let (towards_h, towards_a) = if side == Color::White {
        (pawns << 9, pawns << 7)
    } else {
        (pawns >> 7, pawns >> 9)
    };
    let mut map = Bitboard(
        (towards_h & !Bitboard::from(File::A).0) | (towards_a & !Bitboard::from(File::H).0),
    );
    for from in board.knights() & mine {
        map |= knight_attacks(from);
    }
    for from in (board.bishops() | board.queens()) & mine {
        map |= bishop_attacks(from, occupied);
    }
    for from in (board.rooks() | board.queens()) & mine {
        map |= rook_attacks(from, occupied);
    }
    map | king_attacks(board.king_of(side).expect("a king per side"))
}

/// What `attacker` can win at once: the best gain over the pieces (kings aside) of the other side
/// that it attacks, and how many of them it wins something on. A piece gains its value when no
/// piece defends it, else its value minus that of the cheapest attacker when that is positive.
/// Attacks are those on the board as it stands: no x-rays, pins or en passant. `attacks` and
/// `defences` are the squares each side attacks, so only the pieces under attack are looked at.
fn en_prise(board: &Board, attacker: Color, attacks: Bitboard, defences: Bitboard) -> (i32, i32) {
    let occupied = board.occupied();
    let (mut best, mut count) = (0, 0);
    for square in board.by_color(!attacker) & !board.kings() & attacks {
        let target = piece_value(board.role_at(square).expect("occupied"));
        let gain = if defences.contains(square) {
            let mine = board.by_color(attacker);
            let cheapest = if target == 1
                || (shakmaty::attacks::pawn_attacks(!attacker, square) & board.pawns() & mine).any()
            {
                1
            } else if ((shakmaty::attacks::knight_attacks(square) & board.knights())
                | (shakmaty::attacks::bishop_attacks(square, occupied) & board.bishops()))
                & mine
                != Bitboard(0)
            {
                3
            } else if (shakmaty::attacks::rook_attacks(square, occupied) & board.rooks() & mine)
                .any()
            {
                5
            } else if ((shakmaty::attacks::bishop_attacks(square, occupied)
                | shakmaty::attacks::rook_attacks(square, occupied))
                & board.queens()
                & mine)
                .any()
            {
                9
            } else {
                100
            };
            target - cheapest
        } else {
            target
        };
        if gain > 0 {
            best = best.max(gain);
            count += 1;
        }
    }
    (best, count)
}

/// The threat inputs from `base` on: the mover's best capture, the opponent's, the pieces each side
/// has en prise to the other, and whether the mover is in check.
pub(super) fn threats(pos: &Chess, base: usize, indices: &mut Vec<usize>, values: &mut Vec<f64>) {
    let mover = pos.turn();
    let board = pos.board();
    let mover_map = attacked_by(board, mover);
    let opponent_map = attacked_by(board, !mover);
    let (mover_gain, mover_count) = en_prise(board, mover, mover_map, opponent_map);
    let (opponent_gain, opponent_count) = en_prise(board, !mover, opponent_map, mover_map);
    indices.extend(base..base + THREAT_INPUTS);
    values.extend([
        f64::from(mover_gain) / GAIN_SCALE,
        f64::from(opponent_gain) / GAIN_SCALE,
        f64::from(mover_count) / EN_PRISE_SCALE,
        f64::from(opponent_count) / EN_PRISE_SCALE,
        if pos.is_check() { 1.0 } else { 0.0 },
    ]);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::position_from_fen;

    /// The five threat inputs of a position, in order.
    fn threat_values(fen: &str) -> Vec<f64> {
        let pos = position_from_fen(fen).unwrap();
        let (mut indices, mut values) = (Vec::new(), Vec::new());
        threats(&pos, 0, &mut indices, &mut values);
        assert_eq!(indices, (0..THREAT_INPUTS).collect::<Vec<_>>());
        values
    }

    fn assert_close(actual: &[f64], wanted: &[f64]) {
        for (a, w) in actual.iter().zip(wanted) {
            assert!((a - w).abs() < 1e-12, "{actual:?} != {wanted:?}");
        }
    }

    #[test]
    fn nothing_is_en_prise_at_the_start() {
        let start = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
        assert_close(&threat_values(start), &[0.0; 5]);
    }

    #[test]
    fn an_undefended_piece_gains_its_whole_value() {
        // the pawn on e4 takes the queen; the queen takes the pawn
        let values = threat_values("4k3/8/8/3q4/4P3/8/8/4K3 w - - 0 1");
        assert_close(&values, &[1.0, 1.0 / 9.0, 0.25, 0.25, 0.0]);
    }

    #[test]
    fn a_defended_piece_gains_the_difference_with_the_cheapest_attacker() {
        // the pawn takes a rook that a pawn defends: 5 - 1
        let values = threat_values("4k3/8/4p3/3r4/4P3/8/8/4K3 w - - 0 1");
        assert_close(&values, &[4.0 / 9.0, 0.0, 0.25, 0.0, 0.0]);
    }

    #[test]
    fn a_check_sets_the_flag_and_the_king_can_take_what_is_undefended() {
        let values = threat_values("4k3/8/8/8/8/8/4r3/4K3 w - - 0 1");
        assert_close(&values, &[5.0 / 9.0, 0.0, 0.25, 0.0, 1.0]);
    }
}
