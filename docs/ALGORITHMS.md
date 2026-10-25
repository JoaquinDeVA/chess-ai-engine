# Search algorithms

> **What it is:** How the engine chooses a move: one main algorithm (alpha-beta search) and, one section each, the modifications made on it.
>
> **Read it when:** You want to know why the engine thinks the way it does, what each technique costs or saves, or where it lives in the code.
>
> **Next:** [Data and training](DATA_AND_TRAINING.md)

A chess engine tries moves, imagines the replies, and picks the move leading to the best position it
can force. With about 30 moves per position the lines explode, so everything below is about looking at
fewer or better positions within the clock. After the main algorithm, each modification gets a section
with five parts. Code: `rust/src/search/`, `rust/src/board.rs`, `src/chess_engine/bots/`; evaluations: [MODELS.md](MODELS.md).

The search was first written in Python and moved to Rust (section 10). The node counts, the pruning
counts and the moves quoted in this page were measured with the Python engine; the Rust core
reproduces them exactly (the golden files in `tests/golden/` check it on every run), so they hold
for both. Only the time per node changed, and section 10 gives the speed.

## The main algorithm: alpha-beta search

**Minimax.** The engine's side (MAX) picks the child with the highest value, the opponent (MIN) the
lowest; leaf scores bubble up to the root.

**Alpha-beta pruning.** While exploring, the search keeps a window `(alpha, beta)`: `alpha` is the
best value MAX is already sure to get elsewhere, `beta` the best value MIN is already sure to
concede. When `alpha >= beta` the rest of a node's children cannot change the decision, so they are
skipped (a cut-off). The result is exactly the minimax result with fewer nodes.

![Alpha-beta pruning](https://upload.wikimedia.org/wikipedia/commons/9/91/AB_pruning.svg)

*Alpha-beta pruning: branches that cannot change the decision are cut. Image by Jez9999, CC BY-SA 3.0,
via Wikimedia Commons.*

**Heuristic evaluation and a depth cut-off.** Games are too deep to search to the end, so the search
stops at a fixed depth and asks an evaluation function for a number. This is `Search::node` in
`rust/src/search/mod.rs` (one function for MAX and MIN nodes; `alpha_beta` runs it for a depth and
returns the value, the best move and the counters). Every modification below was added to it.

**Values.** Every value is the expected score for the player the search runs for: `1` win, `0.5` draw,
`0` loss. Evaluations (heuristics and learned models) are clipped to `0.01 .. 0.99`, so an estimate never
outranks a real result; terminal positions use exact `1`, `0.5`, `0`.

How the search visits one node (`Search::node`):

```mermaid
flowchart TD
    N["visit(node)"] --> T{"in transposition table<br/>with enough depth?"}
    T -- "exact, or its bound closes the window" --> R1["return stored result"]
    T -- no --> X{"terminal?"}
    X -- yes --> R2["return 1 - 0.003 · ply (win)<br/>0.5 (draw)<br/>0.003 · ply (loss)"]
    X -- no --> D{"depth_remaining == 0?"}
    D -- yes --> Q["quiescence search at the horizon<br/>(or evaluate(node) if switched off)"]
    D -- no --> C{"hard time limit passed<br/>(or stop flag raised)?"}
    C -- "yes" --> R4["evaluate(node), mark search as truncated"]
    C -- no --> O["order moves:<br/>killer moves first, then captures by MVV-LVA"]
    O --> L["for each move: make it, search the child with (alpha, beta), unmake it"]
    L --> P{"alpha ≥ beta?"}
    P -- yes --> K["remember the move as a killer, stop"]
    P -- no --> L
    K --> S["store the result in the table<br/>(not if truncated)"]
    L --> S
```

The measured tables below come from `python scripts/bench/compare_search_features.py`: a fixed sample
of test positions searched with each technique switched off, one at a time, with `linear_material`
as evaluation. Its report files are local (not versioned), so the numbers are copied here.

## 1. Preference for faster wins

**Problem.** A forced mate in 2 and a forced mate in 9 are both worth `1`, so the engine can see a
win and keep postponing it. The same happens in reverse for losses.

**What it does.** A terminal value is nudged by `0.003` per ply from the root: a quicker win is worth
slightly more than a slower one, a slower loss more than a quicker one. Draws are untouched.

**How it is implemented.** `DEPTH_PREFERENCE_EPSILON = 0.003` and `Search::terminal_utility` in
`rust/src/search/mod.rs`: `1 - ply * epsilon` for a win, `ply * epsilon` for a loss, with `ply`
computed from `depth_remaining`. The quiescence search uses it too.

**Measured effect.** Not measured in this repository.

**Status.** Kept; since the first commit (2026-10-16, imported from the coursework version).

## 2. Iterative deepening with a clock

**Problem.** A game is played against a clock, not against a depth. A fixed depth wastes time in easy
positions and flags in sharp ones, and a search interrupted at a random moment has no answer.

**What it does.** The engine searches depth 1, 2, 3... and keeps the move of the deepest *finished*
iteration. The cheap shallow iterations also fill the transposition table and killer moves that
speed up the deeper ones (sections 3 and 4). The clock logic:

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
* **Minimum depth 3** (`min_depth`, `DEFAULT_MIN_DEPTH`): iterations 1 to 3 are always completed.
  Only the hard limit can interrupt them. This is the fix described next.

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

**The estimators compared.** `python scripts/bench/record_iteration_times.py` replays the
positions of finished games with the stopping rules disabled and records the duration of every
iteration (85 moves of three games, two of `linear_material` and one of `net4_pst`);
`python scripts/bench/compare_time_estimators.py` replays the loop on those series with each rule.
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

**How it is implemented.** `iterative_deepening` in `rust/src/search/mod.rs` (loop,
`OVERSHOOT_FACTOR = 1.15`, discard of truncated iterations; `completed_depth` in the statistics).
`TimeManager` in `src/chess_engine/search/time_management.py` holds the numbers;
`IterativeDeepeningBot.search` in `src/chess_engine/bots/_search_bots.py` computes the soft budget and
the hard limit and passes them to the core. `elapsed` runs from the start of the move, not of each
iteration.

**Measured effect.** Before the fix, 12.0 % of the moves of the partial tournament above finished
below depth 3 (126 of 1,051). With the floor of three iterations, 0 of the 2,900 moves of the first 34
games at 10 min + 5 s did; at 1 min + 1 s, 0 to 0.1 % for the learned models and 5.7 % for `h1`, the
slowest evaluation, whose depth-3 iteration is sometimes cut by the hard limit when the clock is low
(`python scripts/bench/depth_in_game_logs.py data/tournament/reference`). The tournaments in this repository were played with the original
estimator and the floor; the estimator above was chosen afterwards and was not replayed in games. The logs are local files.

**Status.** Iterative deepening and `TimeManager`: kept, since the first commit (2026-10-16).
Minimum depth 3: kept as a guard, since 2026-10-21 (commit `0cdf404`). Estimator `geometric_growth`: default since 2026-10-21 (commit `d3e848c`); the original rule stays available as `ratio_squared`.

## 3. Transposition table

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
lock-free one shared by all of them (section 11). It is shared by the iterations of a move and by the moves of a game, and cleared in
`on_game_start` (values are from the point of view of the colour played). Quiescence does not use it.

**Measured effect.** `python scripts/bench/compare_search_features.py --player linear_material
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

## 4. Killer moves

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
reduce the nodes there. Same caveat as in section 3: small sample, no explanation here.

**Status.** Kept; since the first commit (2026-10-16).

## 5. Move ordering by MVV-LVA

**Problem.** The order in which moves are tried decides how much alpha-beta can skip, and python-chess
lists them in arbitrary order.

**What it does.** Captures go first, best Most Valuable Victim / Least Valuable Attacker first (a pawn
taking a queen before a queen taking a pawn; en passant counts as a pawn capture). Quiet moves keep
their order, so killers stay in front of them.

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

## 6. Quiescence search

**Problem.** A fixed-depth search ends in the middle of an exchange: it sees "queen takes pawn", misses
the recapture and scores a free pawn (the horizon effect).

**What it does.** At depth 0 the position is not scored directly. The side to move may keep the static
evaluation ("stand-pat": nobody is forced to capture) or follow a *noisy* move (capture or promotion)
for a few more plies, with alpha-beta pruning, and the best option is returned.

**How it is implemented.** `Search::quiescence` in `rust/src/search/mod.rs`, called from
`Search::node` at `depth_remaining <= 0`, and `is_noisy` in `rust/src/board.rs` (legal captures and
promotions). It uses neither the table nor killers; its nodes count in `nodes_visited`. The bots
(`IterativeDeepeningBot`) use `DEFAULT_QUIESCENCE_PLIES = 4`; `quiescence_plies=0` switches it off, which
is also the default of the search function itself.

**Measured effect.**

Measured with `python scripts/bench/compare_quiescence.py --players h1,linear_pst_terms` on 1000
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

## 7. Make/unmake on one state, with fast transposition keys

**Problem.** The first version copied the whole game state at every node and built the position key
from a FEN string, which costs time at every node.

**What it does.** The search plays a move in place on one `GameState`, searches the child and
takes the move back; the state keeps an undo stack (position, history). The transposition key is
a hash of the position plus the two move counters, which identifies the same positions as the FEN
without building the string.

**How it is implemented.** `GameState::apply`, `undo` and `key` in `rust/src/board.rs`; the
`apply` / `undo` pairs in `Search::node` and `Search::quiescence`. (The Python version was
`ChessGameState.apply_move`, `undo_move` and `key`, which survive only as a thin wrapper of a board
for the evaluation functions.)

**Measured effect.** Commit `ea86b20` reports, on the fixed benchmark at depth 3 of the Python
engine, identical moves and node counts and a total time from 10.8 s to 3.7 s (2.9 times faster),
*together with* the change of section 8.

**Status.** Kept; since 2026-10-16 (commit `ea86b20`), now in Rust.

## 8. Fast terminal detection with the same draw rules

**Problem.** "Is the game over here?" is asked at every node, and python-chess's
`board.outcome(claim_draw=True)` replays the game to claim a threefold repetition.

**What it does.** Same answer as `board.outcome(claim_draw=True)`, but the repetition is read from the
keys of the positions since the last irreversible move, which the state extends in constant time per
move. Checkmate, stalemate and insufficient material come from the rules of the position; the
fifty-move claim needs 100 halfmoves and a legal move; the threefold claim holds when the position
occurred three times or a legal move reaches one that occurred twice.

**How it is implemented.** `GameState::outcome` and `can_claim_threefold` in `rust/src/board.rs`;
the history is rebuilt on every call from the root FEN and the moves played. Tests:
`rust/tests/golden_board.rs`, which checks 2,066 positions and histories against the Python engine.

**Measured effect.** Commit `ba3d5cb` reports, at depth 3 on the fixed benchmark of the Python
engine, the same moves and nodes as the baseline and a search 1.89 times faster (the 10.8 s to 3.7 s
of section 7 is the combined effect of both changes).

**Status.** Kept; since 2026-10-16 (commit `ba3d5cb`), now in Rust.

## 9. Endgame tablebases

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
recording `stop_reason = "tablebase"` in the search statistics; `python main.py --no-endgame-db`
turns it off. It is on by default when the tables are present; the tournaments and measurement scripts
build their players without it (`make_player`), so that they compare evaluations and search, not tables.

**Measured effect.** Measured with `python scripts/bench/compare_endgame_tablebase.py --positions 60 --seconds 0.5`
(report: `data/reports/ENDGAME_TABLEBASE.md`): 60 random won positions with three to five pieces, the
defender always plays the tablebase move, the winner has 0.5 s per move and 100 moves to mate. The search
with the tables converts 52 of 60 (mean 14.2 moves to mate); the same search without them converts 8 of
60 (20.6 moves).

**Status.** Kept as an option; since 2026-10-21 (commit `2fc0f86`). It is used only at the root:
the search never probes the tables inside the tree, and positions with six or more pieces are
searched as usual.

## 10. The search in Rust

**Problem.** In Python the search visited 3,000 to 12,000 nodes per second, so in the 3 to 10
seconds a move gets it reached depth 4 or 5, far from the depths where a chess engine plays well.
The time went into the interpreter, the board of python-chess and the evaluations, not into the
algorithm.

**What it does.** The same search, in Rust: the rules of the game, the evaluations the search calls
(`h1`, the four logistic models and the sparse network), the alpha-beta with all the techniques
above, the opening book and the endgame tables. Python keeps the game loop, the bots' interface, the
training and the comparison tools, and calls the core once per move with the position, the
evaluation and the time limits.

**How it is implemented.** The crate in `rust/` is built as the Python module `chess_engine._core`
(PyO3 and `maturin`) on top of `shakmaty` for the rules. Nothing changed in the algorithm: the Rust
core reproduces the Python engine node for node. `tests/golden/` holds what the Python engine
produced (2,000 positions with their histories, the features and values of every evaluation,
fixed-depth searches of `h1` and the five models with value, move, nodes and prunings, book
lookups and tablebase moves) and `rust/tests/golden_*.rs` and `tests/test_core_golden.py` compare
the core with it. The legal moves are listed in the order python-chess generates them
(`generation_order`), so that moves of equal value are chosen as before. Replaying the 21,203 moves
of the 200 recorded games of the depth-3 tournament, the core plays 21,192 of them (99.95 %) with
the same number of nodes; the 11 others differ in the last decimal of a value.

**Measured effect.** Measured with `python scripts/bench/measure_search_speed.py --positions 200 --seconds 1,3,10 --depths 4`
(200 test positions, seed 42, a fresh table per position, five processes of one thread each, books
and tables off), once with the Python engine (commit `148b86d`) and once with the core. Time to search
every position to depth 4, in seconds per move:

| player | Python | Rust | times faster |
|---|---|---|---|
| Heuristic (`h1`) | 7.995 | 0.226 | 35.4 |
| Position-terms logistic model (`linear_pst_terms`) | 3.429 | 0.139 | 24.7 |
| Piece-square logistic model (`linear_pst`) | 2.279 | 0.110 | 20.6 |
| Material logistic model (`linear_material`) | 1.335 | 0.076 | 17.6 |
| King-bucket logistic model (`linear_king_bucket`) | 1.990 | 0.120 | 16.6 |
| Four-layer piece-square network (`net4_pst`) | 2.266 | 0.286 | 7.9 |

Mean depth completed in a time budget per move (the clock decides; the estimator stops an iteration
that will not fit, so a move uses 50 to 80 % of the budget):

| player | 1 s: Python → Rust | 3 s: Python → Rust | 10 s: Python → Rust |
|---|---|---|---|
| `h1` | 2.58 → 5.29 | 3.25 → 6.04 | 4.12 → 6.93 |
| `linear_material` | 3.78 → 6.04 | 4.49 → 6.82 | 5.37 → 7.68 |
| `linear_pst` | 3.35 → 5.60 | 4.05 → 6.34 | 4.94 → 7.18 |
| `linear_pst_terms` | 3.19 → 5.43 | 3.85 → 6.26 | 4.62 → 7.13 |
| `linear_king_bucket` | 3.50 → 5.63 | 4.23 → 6.36 | 5.04 → 7.51 |
| `net4_pst` | 3.35 → 4.80 | 4.04 → 5.61 | 4.85 → 6.82 |

In 10 seconds the core searches 2.0 to 2.8 plies deeper. A 200-game round robin at a fixed depth of 3
took 56 s (it took hours in Python). The network gains least because it still adds the 768 inputs
to its first layer at every leaf; an accumulator updated move by move, as NNUE does, would remove
that and has not been written.

**Status.** Kept; since 2026-10-25.

## 11. Several threads (Lazy SMP)

**Problem.** A computer has several cores and the search used one.

**What it does.** With `threads = N` the search runs on N threads. They all search the same position
and share one transposition table, without locks; the main thread runs the iterative deepening as
usual and the others, from other depths and with the moves of the root in a different order, fill the
table with what they find. When the main thread finishes it raises a flag and the others stop. The
move is the main thread's. Dividing the root moves among the threads was not chosen: the first move
takes most of the work (the others are cut off by its value), so the threads would wait for each
other and, ignoring what the others find, they would prune less.

**How it is implemented.** `iterative_deepening` and `helper` in `rust/src/search/mod.rs`; the
table is `SharedTable` in `rust/src/search/shared.rs`, with three atomic words per slot checked by
XOR (a slot torn by two writers is ignored). `threads` and `hash_mb` are options of
`IterativeDeepeningBot` and `--threads`, `--hash` of `main.py`. The tournaments use one thread, so
that games are reproducible and the clock is fair among processes. With more than one thread the
search is not deterministic and the evaluation must run in the core.

**Measured effect.** Measured with `python scripts/bench/measure_search_speed.py --positions 60 --seconds 3 --depths 6
--players linear_pst_terms --threads 1,2,4,6 --workers 1` (the Position-terms logistic model, 60 test
positions, nothing else running on the six-core machine). The nodes per second count the nodes of all
threads:

| threads | nodes/s | seconds to depth 6 | mean depth in 3 s |
|---|---|---|---|
| 1 | 352,906 | 1.273 | 6.88 |
| 2 | 673,207 | 1.185 | 6.93 |
| 4 | 1,212,354 | 1.090 | 6.97 |
| 6 | 1,628,464 | 1.075 | 6.95 |

With a budget of 10 s (30 positions) the mean depth goes from 7.70 with one thread to 7.93 with six.
The threads do add work (4.6 times the nodes per second with six), but most of it repeats what the
main thread already does, so the search to a fixed depth is only 15 % faster with six threads and
the depth gained in a given time is a fraction of a ply. The searches here are short (seconds, depth 6
to 8) and the table has little to share yet. A first version in which the helpers started at depths
1 to 3 gave 1.15 s to depth 6 with six threads; making them search one or two plies deeper than the
main thread is, as here, improved it to 1.075 s. Gains in longer searches were not measured.

**Status.** Kept as an option, off by default; since 2026-10-25.

## Tried and removed: tactical move ordering with Prolog

The repository once had a `TacticalAlphaBetaBot` that asked a Prolog knowledge base which legal moves
create a tactical pattern (forks, pins, skewers) and searched those first, hoping for earlier
cut-offs. Commit `62c2904` measured about 2 ms per node (every node loaded its position into Prolog),
several seconds per move at depth 3 against about one second without it, and sometimes more nodes.
Commit `ca35218` (2026-10-18) removed it because no bot or script used it. In one sentence: the idea
was never shown to pay for its cost, and it was dropped rather than disproved.

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

See also [RESULTS.md](RESULTS.md) and [ARCHITECTURE.md](ARCHITECTURE.md).
