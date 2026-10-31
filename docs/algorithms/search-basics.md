# The classic search techniques

> **What it is:** The techniques of the reference search: the preference for faster wins, iterative deepening, the transposition table, killer moves, MVV-LVA ordering, quiescence, make/unmake, terminal detection, the endgame tables and the opening book.
>
> **Read it when:** You want to know what the search does before any selective pruning.
>
> **Next:** [The search in Rust and threads](rust-and-threads.md)

## Preference for faster wins

**Problem.** A forced mate in 2 and a forced mate in 9 are both worth `1`, so the engine can see a
win and keep postponing it. The same happens in reverse for losses.

**What it does.** A terminal value is nudged by `0.003` per ply from the root: a quicker win is worth
slightly more than a slower one, a slower loss more than a quicker one. Draws are untouched.

**How it is implemented.** `DEPTH_PREFERENCE_EPSILON = 0.003` and `Search::terminal_utility` in
`rust/src/search/node.rs`: `1 - ply * epsilon` for a win, `ply * epsilon` for a loss, with `ply`
computed from `depth_remaining`. The quiescence search uses it too.

**Measured effect.** Not measured in this repository.

**Status.** Kept; since the first commit (2026-10-16, imported from the coursework version).

## Iterative deepening with a clock

**Problem.** A game is played against a clock, not against a depth. A fixed depth wastes time in easy
positions and flags in sharp ones, and a search interrupted at a random moment has no answer.

**What it does.** The engine searches depth 1, 2, 3... and keeps the move of the deepest *finished*
iteration. The cheap shallow iterations also fill the transposition table and killer moves that
speed up the deeper ones (the transposition table and killer moves, below). The clock logic:

* **Soft budget** (`TimeManager.turn_budget`): 15 % of the time left, between 1.5 s and 40 s and never
  more than half the clock (10 minutes left: the 40 s cap).
* **Time estimate.** From the fourth iteration on, a rule predicts the duration of the next
  iteration from the finished ones (`rust/src/search/estimators.rs`; the default is
  `geometric_growth`: the last duration times the geometric mean of the growth between iterations,
  kept between 2 and 8). If the prediction is more than 15 % over the remaining budget, the next
  depth is not started.
* **Hard limit** (`TimeManager.hard_limit`): `1.5 × budget − 0.3 s` (59.7 s for a 40 s budget). The
  running iteration is then interrupted, flagged as truncated and *discarded*, unless it is the very
  first (some move is always needed).
* **Minimum depth 1** (`min_depth`, `DEFAULT_MIN_DEPTH`): the first iteration is always completed, so
  there is always an answer. Until 2026-10-31 the floor was 3 iterations, the fix described next for
  the original estimator; with `geometric_growth` it makes no difference (see Status).

**The defect and its cause.** In the per-move logs of a 10 min + 5 s tournament (partial, 1,051
moves of three models), 126 moves (12.0 %) finished below depth 3, every one of them in under 2 s
although the budget was 35-40 s. The original rule predicted the next iteration as
`last² / previous`, the square of the last duration over the one before. With a transposition table
warmed by the previous moves the depth-1 iteration can last microseconds, so the ratio of the second
iteration to the first reaches thousands (42,166 at most in the recorded games) and its square
predicts hours: the search stopped at depth 2. Replaying recorded iteration times (below), the
original rule ends 15 of 85 moves under depth 3 with a 40 s budget (19 with 10 s), and every bounded
rule ends none. An earlier replay with a cold table, or with a table warmed by a deeper search of the
same position, did not reproduce it; the recorded series, taken from consecutive moves of a game, did.

**The estimators compared.** `python src/chess_lab/scripts/bench/record_iteration_times.py` replays the
positions of finished games with the stopping rules disabled and records the duration of every
iteration (85 moves of three games, two of `linear_material` and one of `net4_pst`);
`python src/chess_lab/scripts/bench/compare_time_estimators.py` replays the loop on those series with each rule.
With a 40 s budget (the cap used with 10 minutes on the clock):

| rule | mean depth | budget used | stopped though the next iteration fitted | iteration cut by the hard limit | move over budget +15 % |
|---|---|---|---|---|---|
| `ratio_squared` (original) | 4.55 | 61 % | 21 % | 9 % | 11 % |
| `geometric_growth` (default) | 4.72 | 68 % | 8 % | 8 % | 13 % |
| `clamped_growth` | 4.72 | 79 % | 9 % | 16 % | 14 % |
| `node_growth` | 4.73 | 76 % | 7 % | 14 % | 13 % |
| `fixed_factor` (×4) | 4.80 | 85 % | 1 % | 16 % | 14 % |

With a 10 s budget the original rule reaches 3.96 and `geometric_growth` 4.00, with 14 % against 8 %
of premature stops. All the bounded rules are about 0.2 plies deeper than the original at 20 and 40 s;
the differences among them are within the noise of 85 moves. `geometric_growth` was chosen because
it uses the whole history, has the lowest overrun at 10 s (2 %) and the fewest wasted
iterations at 40 s. The replay assumes each iteration lasts what it lasted in the recording; the
change was not tested in games, where a difference of 0.2 plies (a few Elo) is far below what a
hundred games can resolve.

**How it is implemented.** `iterative_deepening` in `rust/src/search/deepening.rs` (loop,
`OVERSHOOT_FACTOR = 1.15`, discard of truncated iterations; `completed_depth` in the statistics).
`TimeManager` in `src/chess_engine/search/time_management.py` holds the numbers;
`IterativeDeepeningBot.search` in `src/chess_engine/bots/_search_bots.py` computes the soft budget and
the hard limit and passes them to the core. `elapsed` runs from the start of the move, not of each
iteration.

**Measured effect.** Before the fix, 12.0 % of the moves of the partial tournament above finished
below depth 3 (126 of 1,051). With the floor of three iterations, 0 of the 2,900 moves of the first 34
games at 10 min + 5 s did; at 1 min + 1 s, 0 to 0.1 % for the learned models and 5.7 % for `h1`, the
slowest evaluation, whose depth-3 iteration is sometimes cut by the hard limit when the clock is low
(`python src/chess_lab/scripts/bench/depth_in_game_logs.py data/tournament/reference`). The tournaments in this repository were played with the original
estimator and the floor; the estimator above was chosen afterwards and was not replayed in games. The logs are local files.

**Status.** Iterative deepening and `TimeManager`: kept, since the first commit (2026-10-16).
Minimum depth: 3 from 2026-10-21 (commit `0cdf404`), 1 from 2026-10-31 and no default depth cap (the bots used to stop at depth 3; now only the clock does). Measured with `geometric_growth` on 60 random positions (seeded; ad-hoc script, not versioned): with a floor of 1 or 3 the completed depths are identical (all at least 6 with `linear_pst_terms` at a 1.8 s budget, at least 4 with `h1` at 1 s). Repeating the iterations 4 to 7 pays: on a middlegame position the iterations to depth 7 take 730,876 nodes in total against 1,083,355 for searching depth 7 directly (`_core.Searcher.search_fixed`), because each iteration seeds the move ordering of the next. Estimator `geometric_growth`: default since 2026-10-21 (commit `d3e848c`); the original rule stays available as `ratio_squared`.

## Transposition table

**Problem.** The same position can be reached through different move orders, and iterative deepening
repeats the positions of the previous iteration. A plain search analyses them again every time.

**What it does.** Each searched position is stored with its value, best move, depth and a flag
(`EXACT`, `LOWERBOUND`, `UPPERBOUND`) computed against the `(alpha, beta)` window the node was entered
with, since a cut-off value is only a bound. If the position returns with no more depth remaining than
stored, an exact entry is returned at once and a bound tightens the window (returning if it closes it).

**How it is implemented.** `Table` and `Entry` in `rust/src/search/tt.rs`, probed at the start of
`Search::node` and written at its end. A hash map keyed by the position key (`GameState::key()` in
`rust/src/board.rs`), capped at 500,000 entries (`DEFAULT_MAX_TT_ENTRIES`), oldest dropped first;
nothing is stored once the clock has truncated a search. With several threads the table is a
lock-free one shared by all of them ([several threads](rust-and-threads.md#several-threads-lazy-smp)). It is shared by the iterations of a move and by the moves of a game, and cleared in
`on_game_start` (values are from the point of view of the colour played). Quiescence does not use it.

**Measured effect.** `python src/chess_lab/scripts/bench/compare_search_features.py --player linear_material
--positions 60 --depth 4` (60 positions; the table is empty at the start of every position):

| variant | nodes/move | s/move | same as baseline | Stockfish |
|---|---|---|---|---|
| baseline | 15,988 | 1.365 | 1.000 | 0.317 |
| no transposition table | 17,184 | 1.474 | 1.000 | 0.317 |

At depth 4 the table saves 7.0 % of the nodes (17,184 → 15,988) and 7.4 % of the time, and changes
no move. At depth 6 the sample says the opposite (`--positions 20 --depth 6 --variants
"no transposition table,no killer moves"`):

| variant | nodes/move | s/move | same as baseline | Stockfish |
|---|---|---|---|---|
| baseline | 224,896 | 16.562 | 1.000 | 0.450 |
| no transposition table | 185,089 | 14.619 | 0.950 | 0.400 |

On those 20 positions the baseline visited *more* nodes (+21.5 %) and took more time (+13.3 %)
than the search without the table. The sample is small and the repository does not explain this
result; do not read it as a general rule.

**Status.** Kept; since the first commit (2026-10-16); size cap and no-store-after-truncation since `1ee37d6`.

## Killer moves

**Problem.** Alpha-beta prunes best when the best move is tried first, and a good move in one branch
is often good in a sibling branch.

**What it does.** A move that causes a cut-off is remembered as a "killer" for its depth and tried
before the other moves in later nodes at that depth.

**How it is implemented.** `Killers::record` and the first step of `order` in
`rust/src/search/tt.rs`, called from `Search::node`. Up to three killers are kept per depth
(`MAX_KILLERS_PER_DEPTH`), shared across iterations and moves, cleared at the start of each game. A
killer is identified by its squares and promotion, not by what it captures, as `chess.Move` does.

**Measured effect.** Depth 4, 60 positions: without killers 18,303 nodes per move against 15,988
(killers save 12.6 % of the nodes and 7.9 % of the time, 1.482 s against 1.365 s; `same as
baseline` 0.983). At depth 6 on 20 positions the baseline visited 224,896 nodes per move and the
search without killers 175,520 (baseline +28.1 % nodes; 16.562 s against 13.408 s): killers did *not*
reduce the nodes there. Same caveat as in the transposition table section: small sample, no explanation here.

The whole decision of a move, around that search: the opening book answers first while the game follows
it, then the endgame tables (five pieces or fewer), and only otherwise the search above runs.

![Decision path of one move: opening book, then endgame tables, then iterative-deepening search; the first that applies chooses the move](../img/path-of-a-move.svg)

**Status.** Kept; since the first commit (2026-10-16).

## Move ordering by MVV-LVA

**Problem.** The order in which moves are tried decides how much alpha-beta can skip, and python-chess
lists them in arbitrary order.

**What it does.** Captures go first, best Most Valuable Victim / Least Valuable Attacker first (a pawn
taking a queen before a queen taking a pawn; en passant counts as a pawn capture). Quiet moves keep
their order, so killers stay in front of them. The full order a node uses is: the move stored in the
transposition table (`tt_move`, if that technique is on), the captures by MVV-LVA, the killers, and the
rest of the quiet moves by their history score (`history`, if on).

![Order in which moves are tried: table move, captures by MVV-LVA, killers, then history; an early cut-off skips the later moves](../img/move-ordering.svg)

**How it is implemented.** `mvv_lva_score` and `order` in `rust/src/search/tt.rs` (score
`10 × victim − attacker`; a stable sort, so equal captures keep the order python-chess generates
them in, `generation_order` in `rust/src/board.rs`). `fallback_move` in
`src/chess_engine/bots/_search_bots.py` picks the best capture when a search returns nothing.

**Measured effect.** Depth 4, 60 positions: without ordering 64,003 nodes per move instead of 15,988
(ordering saves 75.0 % of the nodes) and 5.408 s instead of 1.365 s (74.8 % less time); 11.7 % of the
moves differ (`same as baseline` 0.883) and Stockfish agreement drops from 0.317 to 0.283. By far the
most effective technique measured. With table, killers, ordering and quiescence all off: 47,445 nodes
(fewer than without ordering alone, because switching quiescence off removes many nodes itself).

**Status.** Kept; since the first commit (2026-10-16).

## Quiescence search

**Problem.** A fixed-depth search ends in the middle of an exchange: it sees "queen takes pawn", misses
the recapture and scores a free pawn (the horizon effect).

**What it does.** At depth 0 the position is not scored directly. The side to move may keep the static
evaluation ("stand-pat": nobody is forced to capture) or follow a *noisy* move (capture or promotion)
for a few more plies, with alpha-beta pruning, and the best option is returned.

**How it is implemented.** `Search::quiescence` in `rust/src/search/node.rs`, called from
`Search::node` at `depth_remaining <= 0`, and `is_noisy` in `rust/src/board.rs` (legal captures and
promotions). It uses neither the table nor killers; its nodes count in `nodes_visited`. The bots
(`IterativeDeepeningBot`) use `DEFAULT_QUIESCENCE_PLIES = 4`; `quiescence_plies=0` switches it off, which
is also the default of the search function itself.

**Measured effect.**

Measured with `python src/chess_lab/scripts/bench/compare_quiescence.py --players h1,linear_pst_terms` on 1000
test positions (seed 42); agreement is the share of positions where the chosen move is Stockfish's
move (report: `data/reports/QUIESCENCE.md`).

| player | depth | quiescence | agreement | nodes/move | s/move |
|---|---|---|---|---|---|
| Heuristic (`h1`) | 3 | off | 0.308 | 1,648 | 0.392 |
| Heuristic (`h1`) | 3 | 4 plies | 0.355 | 5,980 | 1.454 |
| Position-terms logistic model (`linear_pst_terms`) | 3 | off | 0.291 | 1,601 | 0.209 |
| Position-terms logistic model (`linear_pst_terms`) | 3 | 4 plies | 0.387 | 3,974 | 0.571 |

Paired difference at depth 3 (quiescence minus none): +0.047 for `h1` and +0.096 for
`linear_pst_terms`, both with a standard error of 0.015 (three standard errors or more). The price is
2.5 to 3.6 times the nodes and about 2.7 to 3.7 times the time per move at the same depth.

**Status.** Kept and on by default in the search bots; added on 2026-10-17 (commit `40571a4`).

## Make/unmake on one state, with fast transposition keys

**Problem.** The first version copied the whole game state at every node and built the position key
from a FEN string, which costs time at every node.

**What it does.** The search plays a move in place on one `GameState`, searches the child and
takes the move back; the state keeps an undo stack (position, history). The transposition key is
a hash of the position plus the two move counters, which identifies the same positions as the FEN
without building the string.

**How it is implemented.** `GameState::apply`, `undo` and `key` in `rust/src/board.rs`; the
`apply` / `undo` pairs in `Search::node` and `Search::quiescence`.

**Measured effect.** Commit `ea86b20` reports, on the fixed benchmark at depth 3 of the Python
engine, identical moves and node counts and a total time from 10.8 s to 3.7 s (2.9 times faster),
*together with* the change of the terminal-detection section.

**Status.** Kept; since 2026-10-16 (commit `ea86b20`), now in Rust.

## Fast terminal detection with the same draw rules

**Problem.** "Is the game over here?" is asked at every node, and python-chess's
`board.outcome(claim_draw=True)` replays the game to claim a threefold repetition.

**What it does.** Same answer as `board.outcome(claim_draw=True)`, but the repetition is read from the
keys of the positions since the last irreversible move, which the state extends in constant time per
move. Checkmate, stalemate and insufficient material come from the rules of the position; the
fifty-move claim needs 100 halfmoves and a legal move; the threefold claim holds when the position
occurred three times or a legal move reaches one that occurred twice.

**How it is implemented.** `GameState::outcome` and `can_claim_threefold` in `rust/src/board.rs`;
the history is rebuilt on every call from the root FEN and the moves played. Tests:
`rust/tests/rules.rs` (end of game by mate, stalemate, insufficient material, fifty moves and repetition).

**Measured effect.** Commit `ba3d5cb` reports, at depth 3 on the fixed benchmark of the Python
engine, the same moves and nodes as the baseline and a search 1.89 times faster (the 10.8 s to 3.7 s
of the make/unmake section is the combined effect of both changes).

**Status.** Kept; since 2026-10-16 (commit `ba3d5cb`), now in Rust.

## Endgame tablebases

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
`IterativeDeepeningBot` takes a `tablebase` argument and consults it at the root of `get_move`,
recording `stop_reason = "tablebase"` in the search statistics; `tablebase=None`
turns it off (the window has a switch for it). It is on by default when the tables are present; the tournaments and measurement scripts
build their players without it (`make_player`), so that they compare evaluations and search, not tables.

**Measured effect.** Measured with `python src/chess_lab/scripts/bench/compare_endgame_tablebase.py --positions 60 --seconds 0.5`
(report: `data/reports/ENDGAME_TABLEBASE.md`): 60 random won positions with three to five pieces, the
defender always plays the tablebase move, the winner has 0.5 s per move and 100 moves to mate. The search
with the tables converts 52 of 60 (mean 14.2 moves to mate); the same search without them converts 8 of
60 (20.6 moves).

**Status.** Kept as an option; since 2026-10-21 (commit `2fc0f86`). It is used only at the root:
the search never probes the tables inside the tree, and positions with six or more pieces are
searched as usual.

## Around the search

**The opening book.** Every search bot (`IterativeDeepeningBot`; on by default, `--no-opening-book` switches it off) plays
book moves while the game follows the book, then searches. `IterativeModelBookBot`
(`src/chess_engine/bots/iterative_model_book_bot.py`) is the learned-model bot of the registry. The book
(`OpeningBook` in `rust/src/knowledge/book.rs`, with `src/chess_engine/knowledge/opening_book.py` as its
Python face) is keyed by the first four FEN fields, so
transpositions hit the same entry. Once the game leaves the book it is never consulted again;
`on_game_start` re-enables it and clears the table, killers and statistics.

```mermaid
flowchart TD
    S["get_move(board)"] --> Q{"still in the opening book?"}
    Q -- yes --> W{"position in the book?"}
    W -- yes --> M["play the book move"]
    W -- no --> I["leave the book for the rest of the game"] --> ID
    Q -- no --> ID["iterative deepening with the<br/>learned evaluation"]
    ID --> F{"found a legal move?"}
    F -- yes --> OUT["play it"]
    F -- no --> FB["fallback: best capture (MVV-LVA), else first legal move"]
```

See also [RESULTS.md](../RESULTS.md) and [ARCHITECTURE.md](../ARCHITECTURE.md).
