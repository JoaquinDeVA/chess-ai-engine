# Chess AI Engine

> **Qué es:** Punto de entrada: qué es el motor, cómo instalarlo y los tres comandos para usarlo.
>
> **Cuándo leerla:** La primera vez que abres el repositorio.
>
> **Siguiente paso:** [Índice de la documentación](docs/README.md)

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
pip install -e ".[dev]"          # motor y tests
pip install -e ".[ml,data]"      # opcional: modelos entrenados y datos
```

Opcional: Stockfish para etiquetar datos (`apt install stockfish`).

## Uso

```bash
# tú (blancas) contra un bot, jugadas en notación UCI (por ejemplo e2e4)
python main.py --player1 human --player2 IterativeHeuristicBot

# bot contra bot, eligiendo el modelo del bot con ML y la profundidad
python main.py --player1 IterativeModelBookBot --model linear_pst_terms --player2 IterativeHeuristicBot --depth 3

# torneo todos contra todos entre los bots (guarda resultados y PGN)
python main.py --mode tournament
```

`python main.py --list-bots` muestra los bots disponibles; `--position` admite un FEN y `--export`
guarda la partida en PGN.

## Documentación

Todo el resto (arquitectura, búsqueda, cómo añadir un modelo, resultados) está en
[docs/README.md](docs/README.md).

## Tests

```bash
pytest
pytest -m "not slow"
ruff check . && ruff format --check .
```

## Autoría y agradecimientos

Joaquín De Vicente Abad. El marco base (gestor de partidas, interfaz de bots, registro, clases abstractas
de juego) y la batería inicial de tests vienen de la asignatura *Inteligencia Artificial e Ingeniería del
Conocimiento* (Universidad CEU San Pablo); las clases abstractas derivan del código de
[AIMA](https://github.com/aimacode/aima-python) (MIT). La búsqueda, los bots, los modelos y las
herramientas son trabajo original. Usa [python-chess](https://python-chess.readthedocs.io),
[Stockfish](https://stockfishchess.org) y la [base de evaluaciones de Lichess](https://database.lichess.org) (CC0).

## Licencia

[MIT](LICENSE)
