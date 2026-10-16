# Architecture

The engine is split in layers; each one only depends on the ones below it. All packages live in
`src/chess_engine/`.

```mermaid
flowchart TB
    CLI["main.py<br/>command line"] --> GM["engine/<br/>GameManager, results, PGN"]
    GM --> REG["bots/bot_registry.py<br/>discovers bots"]
    REG --> BOTS["bots/<br/>playable bots"]
    BOTS --> BASES["bots/_search_bots.py<br/>IterativeDeepeningBot"]
    BASES --> SEARCH["search/<br/>algorithms, transposition table,<br/>killer moves, time management"]
    BOTS --> EVAL["evaluation/<br/>features and heuristics"]
    BOTS --> MODELS["models/<br/>learned evaluators"]
    BOTS --> KNOW["knowledge/<br/>opening book · Prolog patterns"]
    SEARCH --> GAME["search/game_problems.py<br/>GameState · GameProblem"]
    TRAIN["training/<br/>train · evaluate · report"] --> MODELS
```

| Package | Responsibility |
|---|---|
| `search/` | Game abstractions (`GameState`, `GameProblem`), the chess implementation on top of python-chess, minimax / alpha-beta / advanced alpha-beta / iterative deepening, search helpers, time management |
| `bots/` | The playable bots and the registry that finds them |
| `evaluation/` | Hand-crafted features (material, mobility, king safety...) and the heuristics `h1`..`h4` |
| `models/` | Learned position evaluators behind one interface (`EvaluationModel`) and a registry |
| `training/` | Commands that train one model and benchmark it against Stockfish |
| `knowledge/` | Opening book (Python) and tactical-pattern knowledge base (Prolog) |
| `engine/` | Game loop with clocks, tournaments, results and PGN export |

## Main classes

```mermaid
classDiagram
    class GameState {
        <<abstract>>
        get_legal_moves()
        make_move(move)
        is_over()
        key()
    }
    class GameProblem {
        <<abstract>>
        actions(state)
        result(state, move)
        utility(state, player)
    }
    GameState <|-- ChessGameState
    GameProblem <|-- ChessProblem

    class ChessBot {
        <<abstract>>
        name
        get_move(board, time_limit)
        on_game_start()
        on_game_end()
    }
    class IterativeDeepeningBot {
        <<abstract>>
        time_manager
        transposition_table
        evaluate(state, player)*
    }
    ChessBot <|-- IterativeDeepeningBot
    ChessBot <|-- HumanPlayer
    IterativeDeepeningBot <|-- IterativeHeuristicBot
    IterativeDeepeningBot <|-- IterativeModelBookBot

    class EvaluationModel {
        <<abstract>>
        extractor
        predict_cp(board)
        predict_target(inputs)
        fit()
        save()
        load()
    }
    EvaluationModel <|-- SklearnModel
    EvaluationModel <|-- KerasModel
    IterativeModelBookBot o-- EvaluationModel
```

`ChessGameState` wraps a `chess.Board` and its `key()` is the FEN, which is what the transposition
table uses. A bot's whole job is to implement `evaluate()` (and optionally `order_moves()`); the
base classes provide the search, the time budget and the bookkeeping.

## One move

```mermaid
sequenceDiagram
    participant G as GameManager
    participant B as Bot (IterativeDeepeningBot)
    participant S as Iterative deepening
    participant A as Alpha-beta
    participant E as evaluate()
    G->>B: get_move(board, time left)
    B->>B: budget = TimeManager.turn_budget(time left)
    B->>S: search(state, budget)
    loop depth 1 .. max_depth, while time allows
        S->>A: advanced_heuristic_alphabeta_search(depth)
        A->>E: evaluate(leaf, player)
        E-->>A: expected score in [0.01, 0.99]
        A-->>S: best move + statistics
    end
    S-->>B: move of the deepest finished iteration
    B-->>G: legal move
```

## Bot discovery

`BotRegistry` imports every public module in `bots/` and registers each `ChessBot` subclass defined
there under its class name. Modules starting with `_` hold shared base classes and are skipped, so
adding a bot is adding one file (see [ADDING_A_BOT.md](ADDING_A_BOT.md)). `get_bot(name, **options)`
forwards constructor arguments, which is how `--model` and `--depth` reach the bots.

## Extension points

| To add... | Implement | Guide |
|---|---|---|
| a bot | `evaluate()` in a subclass of `IterativeDeepeningBot` | [ADDING_A_BOT.md](ADDING_A_BOT.md) |
| an evaluation model | an `EvaluationModel` and register it | [ADDING_A_MODEL.md](ADDING_A_MODEL.md) |
| another game | `GameState` and `GameProblem`; the algorithms are game-agnostic | [SEARCH_ALGORITHMS.md](SEARCH_ALGORITHMS.md) |
