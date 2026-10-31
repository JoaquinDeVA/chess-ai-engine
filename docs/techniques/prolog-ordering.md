# Tactical move ordering with Prolog

> **What it is:** Asking a Prolog knowledge base for tactical moves to search first; tried and removed.
>
> **Read it when:** You wonder what happened to the Prolog bot.
>
> **Next:** [Data](../DATA.md)

## Problem

Good move ordering makes alpha-beta cut sooner, and tactical moves (forks, pins, skewers) are often the best.
MVV-LVA only sees captures.

## Idea

Describe tactical patterns as rules in a Prolog knowledge base, ask it which legal moves create one, and search
those first.

## How it works

At every node the position was loaded into Prolog as facts, the rules were queried for moves that create a
pattern, and those moves went to the front of the list.

**Worked example.** A knight move that attacks king and rook at once matches the fork rule and is searched first,
before quiet moves that MVV-LVA would leave in generation order.

## Implementation

`TacticalAlphaBetaBot`, removed in `ca35218` (2026-10-18); in the git history.

## Measured effect

Commit `62c2904` measured about 2 ms per node to load the position into Prolog: several seconds per move at depth
3 against about one second without it, and sometimes more nodes, not fewer.

## Status

Removed on 2026-10-18 (`ca35218`): no bot or script used it. The idea was never shown to pay for its cost; it was
dropped, not disproved.
