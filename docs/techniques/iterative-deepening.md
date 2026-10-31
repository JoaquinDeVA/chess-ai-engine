# Iterative deepening with a clock

> **What it is:** Searching depth 1, 2, 3... and keeping the deepest finished answer, with the rules that decide when to stop.
>
> **Read it when:** You want to know how the engine spends its time on a move.
>
> **Next:** [Transposition table](transposition-table.md)

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

**The estimators compared.** `python (removed) scripts/bench/record_iteration_times.py` replays the
positions of finished games with the stopping rules disabled and records the duration of every
iteration (85 moves of three games, two of `linear_material` and one of `net4_pst`);
`python (removed) scripts/bench/compare_time_estimators.py` replays the loop on those series with each rule.
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
`TimeManager` in `(removed) chess_engine/bots/time_manager.py` holds the numbers;
`SearchBot.get_move` in `(removed) chess_engine/bots/search_bot.py` computes the soft budget and
the hard limit and passes them to the core. `elapsed` runs from the start of the move, not of each
iteration.

**Measured effect.** Before the fix, 12.0 % of the moves of the partial tournament above finished
below depth 3 (126 of 1,051). With the floor of three iterations, 0 of the 2,900 moves of the first 34
games at 10 min + 5 s did; at 1 min + 1 s, 0 to 0.1 % for the learned models and 5.7 % for `h1`, the
slowest evaluation, whose depth-3 iteration is sometimes cut by the hard limit when the clock is low
(`python (removed) scripts/bench/depth_in_game_logs.py data/tournament/reference`). The tournaments in this repository were played with the original
estimator and the floor; the estimator above was chosen afterwards and was not replayed in games. The logs are local files.

**Status.** Iterative deepening and `TimeManager`: kept, since the first commit (2026-10-16).
Minimum depth: 3 from 2026-10-21 (commit `0cdf404`), 1 from 2026-10-31 and no default depth cap (the bots used to stop at depth 3; now only the clock does). Measured with `geometric_growth` on 60 random positions (seeded; ad-hoc script, not versioned): with a floor of 1 or 3 the completed depths are identical (all at least 6 with `linear_pst_terms` at a 1.8 s budget, at least 4 with `h1` at 1 s). Repeating the iterations 4 to 7 pays: on a middlegame position the iterations to depth 7 take 730,876 nodes in total against 1,083,355 for searching depth 7 directly (one alpha-beta of depth 7 through `search::alpha_beta`; ad-hoc, not versioned), because each iteration seeds the move ordering of the next. Estimator `geometric_growth`: default since 2026-10-21 (commit `d3e848c`); the original rule stays available as `ratio_squared`.
