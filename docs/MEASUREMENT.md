# How a change is measured

> **What it is:** The instrument that decides whether a change to the engine is kept: engine matches with fastchess, paired openings, the SPRT test and the A/A check of the instrument itself.
>
> **Read it when:** You are about to measure a technique or a model, or want to know what a score, an Elo interval or an LLR in these pages means.
>
> **Next:** [Openings](OPENINGS.md)

## Why games

Nodes and time per move show what a change costs, not whether the engine plays better. A change can search fewer
nodes, reach more depth and still lose: late move reductions (removed, [catalogue](techniques/README.md#how-the-switches-were-judged)) searched half a ply deeper and
scored 49 % against the engine without them. So a change is kept on the evidence of games against the current
engine, and the node and depth counts only explain the result.

## The instrument

- **Two copies of the same binary.** fastchess plays the UCI binary against itself, engine A with the change
  switched on through a UCI option (`option.Techniques=...`) and engine B without it. Everything else is equal:
  machine, time control, `Hash`, threads.
- **Paired openings.** Every game starts from a position of the [opening suite](OPENINGS.md) and every position is
  played twice with the colours swapped, so a lopsided opening favours both engines equally. A pair scores 0, ½,
  1, 1½ or 2 points for A.
- **A time control or a node count.** At a time control (`tc=10+0.1`) the result includes speed: a change that
  costs time per node must earn it back. At a fixed node count (`nodes=20000`) speed plays no part and the search
  alone is compared; the result is reproducible on any machine.
- **Another machine.** Matches run on an otherwise idle eight-core desktop, so that the clock of one game is not
  shared with other work; `src/chess_lab/scripts/worker/match.sh` starts them there and passes its options to
  fastchess.

## Reading a result

fastchess reports, every few pairs and at the end, a block like this one, from the A/A match `aa_s12` below
(`data/matches/aa_s12/match.log`):

```text
Elo: 9.04 +/- 25.04, nElo: 11.01 +/- 30.45
Games: 500, Wins: 227, Losses: 214, Draws: 59, Points: 256.5 (51.30 %)
Ptnml(0-2): [37, 16, 129, 33, 35], WL/DD Ratio: 24.80
LLR: 0.28 (9.5%) (-2.94, 2.94) [-5.00, 0.00]
```

- **Score and Elo.** A scored 51.3 %; the Elo difference that predicts that score is +9, with a 95 % interval of
  ±25. An interval that contains 0 says nothing about which engine is stronger.
- **Pentanomial.** Of the 250 pairs, 37 gave A 0 points, 16 gave ½, 129 gave 1, 33 gave 1½ and 35 gave 2. The
  error is computed over pairs, not games, because the two games of an opening are not independent.
- **LLR.** The progress of the SPRT test, next section.

## The SPRT

A fixed number of games is either too few to separate a small difference or more than needed for a clear one.
The sequential probability ratio test (SPRT) plays until the evidence is strong enough for one of two hypotheses:

- **H0:** A is `elo0` Elo stronger than B (in these matches `elo0 = −5`: A is 5 Elo weaker, a regression).
- **H1:** A is `elo1` Elo stronger than B (`elo1 = 0`: no regression).

After every pair it updates the log-likelihood ratio (LLR): how much more likely the results are under H1 than
under H0. With error rates alpha = beta = 0.05 the bounds are `ln(0.05 / 0.95) = −2.94` and
`ln(0.95 / 0.05) = +2.94`. The match stops when the LLR crosses one: below −2.94 H0 is accepted (the change
regresses), above +2.94 H1 is accepted (it does not). A cap on rounds stops a match that crosses neither, which
means the difference is smaller than the test can see in that many games.

**Worked example**: the test of late move reductions (removed, [catalogue](techniques/README.md#how-the-switches-were-judged)) at 20,000 nodes per move, `[−5, 0]`
(`data/matches/lmr_nodes/match.log`). A scored a little under 49 % from the start, and the LLR fell steadily:

| games | 200 | 2,200 | 4,200 | 6,200 | 8,200 | 10,200 | 10,244 |
|---|---|---|---|---|---|---|---|
| LLR | −0.19 | −0.67 | −1.16 | −1.65 | −2.13 | −2.62 | −2.94 |

At 10,244 games it crossed −2.94 and the match stopped: H0 accepted, the change regresses (final score 48.76 %,
−8.6 ± 6.1 Elo), with a 5 % risk of being wrong. Had A held 50 %, the LLR would have climbed towards +2.94.

How many games a test needs grows with the square of the precision: the A/A matches below have an interval of
about ±25 Elo after 500 games, so ±5 Elo takes roughly 25 times as many, about 12,000 games. That is why the
`[−5, 0]` tests are capped at 6,000 rounds (12,000 games).

## The A/A check

Before trusting the instrument, the same engine plays itself (A and B identical) under the conditions of a real
test. The result must be about 50 % with an interval around 0, and the SPRT must not accept H0: a false regression
would mean the openings, the pairing or the machine bias the result.

Three A/A matches at 10 s + 0.1 s, `Hash` 16, SPRT `[−5, 0]`, default techniques, 3 games at a time on the worker
(`match.sh start aa_s12 -each tc=10+0.1 proto=uci option.Hash=16 -sprt elo0=-5 elo1=0 alpha=0.05 beta=0.05 model=normalized -rounds 250 -games 2 -concurrency 3 -ratinginterval 20 -srand 12`;
`aa_s11` and `aa_s13` with seeds 11 and 13; logs in `data/matches/<name>/match.log`):

| match | games | score | Elo (95 %) | LLR | verdict |
|---|---|---|---|---|---|
| `aa_s11` | 400 counted | 50.5 % | +3.5 ± 26.8 | +0.11 | no bound crossed |
| `aa_s12` | 500 | 51.3 % | +9.0 ± 25.0 | +0.28 | no bound crossed |
| `aa_s13` | 500 | 50.7 % | +4.9 ± 23.9 | +0.18 | no bound crossed |

All three intervals contain 0 and none came near −2.94: the instrument shows no bias at this size. At a fixed node
count an A/A match says nothing: the engine is deterministic, so both games of a pair are the same game with the
colours exchanged and every pair scores exactly 1–1 (`data/matches/aa_probe*/match.log`: Elo 0.00 ± 0.00 over 16
and 32 games). A fixed-node test has variance only because A and B differ.

## Which time control

A test needs about 12,000 games to see ±5 Elo (above), so the cost of a test is the length of one game times
12,000, divided by the games played at once. Three levels follow from that:

| Level | Control | Use | Status |
|---|---|---|---|
| Screening | `tc=10+0.1`, SPRT `[−5, 0]` | every candidate; one that regresses here is dropped | used for the A/A matches above |
| Confirmation | `tc=60+0.6`, SPRT `[0, 5]` | only a candidate that did not regress | not run yet |
| Reference | `tc=600+5` | the final engine against other engines, never one test per technique | not run yet (a game lasts tens of minutes: 12,000 games would take hundreds of hours even eight at a time, not measured) |

**Fixed nodes favour a change that saves nodes.** With the same node budget such a change searches deeper, so the
test is the best case for it. Late move reductions lost anyway (48.76 % at 20,000 nodes,
`data/matches/lmr_nodes/match.log`), and a change that loses at equal nodes does not win at equal time unless it
also makes each node cheaper. A win at fixed nodes proves nothing about time: it has to be confirmed at a time
control. Fixed nodes are therefore the first filter for a change that should not lose, never the last.

## How to measure a change

1. Build the binary (`cargo build --release --bin uci` in `rust/`) and the opening suite ([Openings](OPENINGS.md)).
2. Check that the binary accepts the technique list. A misspelt name stops nothing: the binary answers
   `info string unknown technique ...` and keeps the default set, fastchess does not show that line, and the match
   silently becomes an A/A:

   ```bash
   printf 'setoption name Techniques value incremental,tt_move,qs_tt\nquit\n' | rust/target/release/uci   # must print nothing
   ```

3. Play an SPRT with the change in A only, for example:

   ```bash
   fastchess -engine cmd=rust/target/release/uci name=A option.Techniques=incremental,tt_move,history,pvs,aspiration,null_move,qs_tt \
       -engine cmd=rust/target/release/uci name=B option.Techniques=incremental,tt_move,history,pvs,aspiration,null_move \
       -each proto=uci nodes=20000 option.Hash=16 -openings file=data/openings/openings.epd format=epd order=random \
       -sprt elo0=-5 elo1=0 alpha=0.05 beta=0.05 model=normalized -rounds 6000 -games 2 -concurrency 8 -srand 21
   ```

4. Write the final block of fastchess, the command and the verdict in the technique's page; if it changes the
   default, also in [Results](RESULTS.md) and the [changelog](../CHANGELOG.md).
