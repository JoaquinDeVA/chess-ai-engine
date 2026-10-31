# Tactical move ordering with Prolog

> **What it is:** Asking a Prolog knowledge base for tactical moves to search first; tried and removed.
>
> **Read it when:** You wonder what happened to the Prolog bot.
>
> **Next:** [Data](../DATA.md)

The repository once had a `TacticalAlphaBetaBot` that asked a Prolog knowledge base which legal moves
create a tactical pattern (forks, pins, skewers) and searched those first, hoping for earlier
cut-offs. Commit `62c2904` measured about 2 ms per node (every node loaded its position into Prolog),
several seconds per move at depth 3 against about one second without it, and sometimes more nodes.
Commit `ca35218` (2026-10-18) removed it because no bot or script used it. In one sentence: the idea
was never shown to pay for its cost, and it was dropped rather than disproved.
