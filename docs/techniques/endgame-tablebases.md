# Endgame tablebases

> **What it is:** Playing positions of five pieces or fewer from the Syzygy tables instead of searching them.
>
> **Read it when:** You want to know how endgames are played or where the tables come from.
>
> **Next:** [The search in Rust](rust-core.md)

**Problem.** A search with a static evaluation plays simple endgames badly. King and queen against
king is a win that takes about ten moves of method; at a depth of four or five plies the horizon
is nowhere near the mate, so the evaluation wanders and the 50-move rule can arrive first.
With five pieces or fewer the exact answer for every position is known and can be stored.

**What it does.** When the position has at most five pieces (kings included) and no castling
rights, the bot does not search: it plays the move given by Syzygy tablebases. The tables say, for
every position, whether the side to move wins, draws or loses and how many moves are left before the
next capture or pawn move (the distance to zeroing, DTZ). The bot keeps the best result and then
mates at once if it can; among winning moves it takes the one with the smallest DTZ, and when it is
losing it takes the one with the largest, to resist longest.

**How it is implemented.** `EndgameTablebase` in `rust/src/knowledge/tablebase.rs` (`covers`,
`best_move` and the move ranking), on top of the `shakmaty-syzygy` crate, with
`src/chess_engine/knowledge/endgame_tablebase.py` as its Python face. The 3-4-5
piece tables (290 files, 939 MB) live in `data/knowledge/endgame`; they are not versioned.
`SearchBot` takes a `tablebase` argument and consults it at the root of `get_move`,
recording `stop_reason = "tablebase"` in the search statistics; `tablebase=None`
turns it off (the window has a switch for it). It is on by default when the tables are present; the tournaments and measurement scripts
build their players without it (`make_player`), so that they compare evaluations and search, not tables.

**Measured effect.** Measured with `python (removed) scripts/bench/compare_endgame_tablebase.py --positions 60 --seconds 0.5`
(report: `data/reports/ENDGAME_TABLEBASE.md`): 60 random won positions with three to five pieces, the
defender always plays the tablebase move, the winner has 0.5 s per move and 100 moves to mate. The search
with the tables converts 52 of 60 (mean 14.2 moves to mate); the same search without them converts 8 of
60 (20.6 moves).

**Status.** Kept as an option; since 2026-10-21 (commit `2fc0f86`). It is used only at the root:
the search never probes the tables inside the tree, and positions with six or more pieces are
searched as usual.

**The endgame tables** are the Syzygy tablebases for three, four and five pieces (290 files, 939 MB)
in `data/knowledge/endgame`. They are not versioned and cannot be rebuilt by this project: they
come from `tablebase.lichess.ovh` (or any Syzygy 3-4-5 set). The search bots use them when the folder exists
(the `tablebase` option of the bots, or the setting of the window, switches them off).
