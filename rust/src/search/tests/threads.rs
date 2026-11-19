//! Lazy SMP: several threads search one position.

use crate::board::uci_string;
use crate::search::*;

use super::helpers::*;

#[test]
fn several_threads_return_a_legal_move_and_do_not_lose_nodes() {
    let positions = [
        START,
        "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
        "r3k2r/pp3ppp/2n1bn2/q1pp4/3P4/2N1PN2/PP2BPPP/R2QK2R w KQkq - 0 1",
        "8/2k5/8/3Pp3/8/8/8/4K3 w - e6 0 1",
    ];
    for fen in positions {
        let options = Options::default();
        let mut memory = Memory::new(4, 1);
        let mut game = state(fen);
        let (best, stats) = iterative_deepening(
            &mut game,
            &material(),
            &mut memory,
            &options,
            4,
            1e9,
            &control(1e9, &no_time),
        )
        .unwrap();
        let legal = crate::board::legal_moves(game.pos());
        assert!(
            legal
                .iter()
                .any(|m| Some(uci_string(m)) == best.as_ref().map(uci_string)),
            "{fen}"
        );
        assert!(stats.nodes_visited > 0);
    }
}
