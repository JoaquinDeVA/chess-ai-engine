# Documentación

> **Qué es:** Índice único de la documentación del motor.
>
> **Cuándo leerla:** Cuando buscas una página concreta o no sabes por dónde empezar.
>
> **Siguiente paso:** [ARCHITECTURE.md](ARCHITECTURE.md)

## Empieza aquí

| Página | Para qué | Cuándo leerla |
|---|---|---|
| [README raíz](../README.md) | Instalación y los comandos para jugar, enfrentar bots y lanzar torneos | La primera vez que abres el repo |

## Usar el motor

| Página | Para qué | Cuándo leerla |
|---|---|---|
| [README raíz](../README.md) | `main.py`: partida contra un bot, bot contra bot, torneo, `--list-bots` | Para jugar o lanzar un torneo |
| [RESULTS.md](RESULTS.md) | Qué modelo juega mejor y con qué comandos se midió | Para elegir el modelo de un bot |

## Entender el diseño

| Página | Para qué | Cuándo leerla |
|---|---|---|
| [ARCHITECTURE.md](ARCHITECTURE.md) | Paquetes, dependencias entre capas, vida de una jugada y cómo añadir un bot | Antes de tocar el código |
| [SEARCH_ALGORITHMS.md](SEARCH_ALGORITHMS.md) | Minimax, alfa-beta, profundización iterativa y quiescencia | Para entender cómo se elige una jugada |

## Modelos y resultados

| Página | Para qué | Cuándo leerla |
|---|---|---|
| [ADDING_A_MODEL.md](ADDING_A_MODEL.md) | Crear, entrenar y medir un modelo de evaluación | Al añadir un modelo |
| [RESULTS.md](RESULTS.md) | Datos, escala de esfuerzo, juez estático, quiescencia, torneo y discrepancia abierta | Para comparar modelos o repetir una medición |

Las decisiones y el glosario de trabajo viven en `vault/`, que es local y no se versiona.
