import json

import chess

from knowledge.opening_book import DEFAULT_BOOK_PATH, OpeningBook, position_key


def test_key_ignores_move_counters():
    a = chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    b = chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 12 40")
    assert position_key(a) == position_key(b)


def test_lookup_returns_the_book_move_even_after_a_transposition():
    book = OpeningBook({"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1": "e2e4"})
    transposed = chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 6 20")
    assert book.lookup(transposed) == chess.Move.from_uci("e2e4")


def test_unknown_position_and_illegal_entries_return_none():
    book = OpeningBook({"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1": "e2e5"})
    assert book.lookup(chess.Board()) is None  # the stored move is illegal
    assert OpeningBook({}).lookup(chess.Board()) is None


def test_first_entry_wins_for_duplicated_positions():
    book = OpeningBook(
        {
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1": "e2e4",
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 4 3": "d2d4",
        }
    )
    assert len(book) == 1
    assert book.lookup(chess.Board()) == chess.Move.from_uci("e2e4")


def test_load_from_file(tmp_path):
    path = tmp_path / "book.json"
    path.write_text(json.dumps({chess.Board().fen(): "g1f3"}))
    assert OpeningBook.load(path).lookup(chess.Board()) == chess.Move.from_uci("g1f3")


def test_shipped_book_is_legal_and_covers_the_starting_position():
    book = OpeningBook.load(DEFAULT_BOOK_PATH)
    assert len(book) > 2000
    assert book.lookup(chess.Board()) is not None
    with open(DEFAULT_BOOK_PATH, encoding="utf-8") as handle:
        for fen, uci in json.load(handle).items():
            assert chess.Move.from_uci(uci) in chess.Board(fen).legal_moves
