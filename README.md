# Chess AI Engine

Motor de ajedrez en Python basado en búsqueda adversaria: minimax, poda alfa-beta y profundización
iterativa con control de reloj. Evalúa posiciones con heurísticas a mano o con modelos entrenados para
imitar a Stockfish, y trae gestor de partidas, torneos y un marco para añadir bots y modelos nuevos.

[![CI](https://github.com/JoaquinDeVA/chess-ai-engine/actions/workflows/ci.yml/badge.svg)](../../actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Code style](https://img.shields.io/badge/code%20style-ruff-261230)

## Instalación

Python 3.11 o superior.

```bash
git clone https://github.com/JoaquinDeVA/chess-ai-engine.git
cd chess-ai-engine
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"              # motor y tests
pip install -e ".[ml,prolog,data]"   # opcional: modelos entrenados, patrones Prolog, datos
```

Opcional: SWI-Prolog para el detector de patrones tácticos y Stockfish para etiquetar datos
(`apt install swi-prolog stockfish`).

## Uso

```bash
# tú (blancas) contra un bot, jugadas en notación UCI (por ejemplo e2e4)
python main.py --player1 human --player2 IterativeHeuristicBot

# bot contra bot, eligiendo el modelo del bot con ML y la profundidad
python main.py --player1 IterativeModelBookBot --model cnn_resnet --player2 IterativeHeuristicBot --depth 3

# torneo todos contra todos entre los bots (guarda resultados y PGN)
python main.py --mode tournament
```

`python main.py --list-bots` muestra los bots disponibles; `--position` admite un FEN y `--export`
guarda la partida en PGN.

## Dónde leer

| Quiero... | Página |
|---|---|
| Ver todo el índice | [docs/README.md](docs/README.md) |
| Entender las capas y el recorrido de una jugada | [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) |
| Conocer los algoritmos de búsqueda | [docs/SEARCH_ALGORITHMS.md](docs/SEARCH_ALGORITHMS.md) |
| Ver los patrones tácticos en Prolog | [docs/TACTICAL_PATTERNS.md](docs/TACTICAL_PATTERNS.md) |
| Añadir un bot | [docs/ADDING_A_BOT.md](docs/ADDING_A_BOT.md) |
| Añadir un modelo de evaluación | [docs/ADDING_A_MODEL.md](docs/ADDING_A_MODEL.md) |
| Ver el historial de modelos | [docs/training/MODELS_HISTORY.md](docs/training/MODELS_HISTORY.md) |
| Ver lo que falta por hacer | [docs/ROADMAP.md](docs/ROADMAP.md) |

## Tests

```bash
pytest                       # los tests que necesitan Prolog o Stockfish se saltan si faltan
pytest -m "not slow"
ruff check . && ruff format --check .
```

## Autoría y agradecimientos

Joaquín De Vicente Abad. El marco base (gestor de partidas, interfaz de bots, registro, clases abstractas
de juego) y la batería inicial de tests vienen de la asignatura *Inteligencia Artificial e Ingeniería del
Conocimiento* (Universidad CEU San Pablo); las clases abstractas derivan del código de
[AIMA](https://github.com/aimacode/aima-python) (MIT). La búsqueda, los bots, los modelos y las
herramientas son trabajo original. Usa [python-chess](https://python-chess.readthedocs.io),
[Stockfish](https://stockfishchess.org), [SWI-Prolog](https://www.swi-prolog.org) con
[PySwip](https://github.com/yuce/pyswip) y la [base de evaluaciones de Lichess](https://database.lichess.org) (CC0).

## Licencia

[MIT](LICENSE)
