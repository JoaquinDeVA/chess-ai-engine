//! The rules of the core: move generation by perft, and the end of the game.

use _core::board::{ordered_moves, position_from_fen, GameState};
use shakmaty::{Chess, Color, Position};

const START: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
const KIWIPETE: &str = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1";

fn perft(pos: &Chess, depth: u32) -> u64 {
    if depth == 0 {
        return 1;
    }
    let moves = ordered_moves(pos);
    if depth == 1 {
        return moves.len() as u64;
    }
    moves
        .into_iter()
        .map(|m| perft(&pos.clone().play(m).unwrap(), depth - 1))
        .sum()
}

#[test]
fn perft_counts_match_the_published_ones() {
    let start = position_from_fen(START).unwrap();
    assert_eq!(perft(&start, 4), 197_281);
    let kiwipete = position_from_fen(KIWIPETE).unwrap();
    assert_eq!(perft(&kiwipete, 3), 97_862);
}

fn outcome(fen: &str, moves: &[&str]) -> Option<Option<Color>> {
    let moves: Vec<String> = moves.iter().map(|m| m.to_string()).collect();
    let state = GameState::from_moves(fen, &moves).unwrap();
    state.outcome(&ordered_moves(state.pos()))
}

#[test]
fn the_end_of_the_game_is_recognised() {
    let cases = [
        (
            "R5k1/5ppp/8/8/8/8/5PPP/6K1 b - - 0 1",
            Some(Some(Color::White)),
        ), // mate
        ("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1", Some(None)), // stalemate
        ("7k/8/6K1/8/8/8/8/N7 w - - 0 1", Some(None)),  // knight only
        ("7k/8/6K1/8/8/8/8/R7 w - - 100 80", Some(None)), // fifty moves
        ("7k/8/6K1/8/8/8/8/R7 w - - 99 80", None),
        (START, None),
    ];
    for (fen, expected) in cases {
        assert_eq!(outcome(fen, &[]), expected, "{fen}");
    }
}

#[test]
fn a_threefold_repetition_is_a_draw() {
    let shuffle = ["g1f3", "g8f6", "f3g1", "f6g8"];
    let moves: Vec<&str> = shuffle.iter().chain(shuffle.iter()).copied().collect();
    assert_eq!(outcome(START, &moves), Some(None));
    assert_eq!(outcome(START, &shuffle), None);
}
