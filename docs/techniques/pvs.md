# Principal variation search (`pvs`)

> **What it is:** Searching every move after the first with a zero-width window, and again in full only if it beats the first.
>
> **Read it when:** You want to know how the search proves a move is not better cheaply.
>
> **Next:** [Aspiration windows](aspiration.md)

**Problem.** After the first move, the rest are only checked to see that none is better; a full window wastes work.

**What it does.** Searches the first move with the full window and the others with a zero-width window around alpha;
only a move that beats alpha is searched again with the full window.

![PVS searches later moves with a zero-width window and repeats with the full window if one fails high; aspiration starts with a narrow window and widens it when the result falls outside](../img/pvs-aspiration.svg)

**How it is implemented.** `child_value` in `rust/src/search/node.rs`; the scout window uses
`SCOUT_EPSILON = 1e-9` because values are continuous probabilities.

**Measured effect.** Nodes 672,800 -> 487,244; same move 100 %.

**Status.** On by default.
