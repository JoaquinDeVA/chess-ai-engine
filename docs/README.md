# Documentación

Índice único de la documentación del motor. Si es tu primera vez, empieza por la primera sección.

## Empieza aquí

- Instalar y jugar una partida: [README raíz](../README.md).
- Entender cómo encajan las piezas: [ARCHITECTURE.md](ARCHITECTURE.md).
- Saber qué falta por hacer: [ROADMAP.md](ROADMAP.md).

## Usar el motor

| Página | Para qué | Cuándo leerla |
|---|---|---|
| [README raíz](../README.md) | Instalación y comandos para jugar, enfrentar bots y lanzar torneos | La primera vez que abres el repo |

## Entender el diseño

| Página | Para qué | Cuándo leerla |
|---|---|---|
| [ARCHITECTURE.md](ARCHITECTURE.md) | Paquetes, dependencias entre capas y vida de una jugada | Antes de tocar el código |
| [SEARCH_ALGORITHMS.md](SEARCH_ALGORITHMS.md) | Minimax, alfa-beta y profundización iterativa | Para entender cómo se elige una jugada |
| [TACTICAL_PATTERNS.md](TACTICAL_PATTERNS.md) | Detector de clavadas, horquillas y ensartadas en Prolog | Si vas a usar conocimiento táctico |

## Modelos de evaluación

| Página | Para qué | Cuándo leerla |
|---|---|---|
| [ADDING_A_MODEL.md](ADDING_A_MODEL.md) | Crear, entrenar y medir un modelo | Al añadir un modelo |
| [training/MODELS_HISTORY.md](training/MODELS_HISTORY.md) | Tabla de modelos con precisión y coste | Para comparar modelos |
| [training/SELECTION_PROTOCOL.md](training/SELECTION_PROTOCOL.md) | Cómo se entrenan, comparan y eligen los candidatos | Antes de lanzar una comparación |
| [training/DATA.md](training/DATA.md) | Origen y formato de los datos de Lichess | Antes de preparar datos |

## Extender el proyecto

| Página | Para qué | Cuándo leerla |
|---|---|---|
| [ADDING_A_BOT.md](ADDING_A_BOT.md) | Añadir un bot en un solo fichero | Al crear un jugador nuevo |
| [ADDING_A_MODEL.md](ADDING_A_MODEL.md) | Añadir un modelo de evaluación | Al crear un evaluador nuevo |

## Decisiones y roadmap

| Página | Para qué | Cuándo leerla |
|---|---|---|
| [ROADMAP.md](ROADMAP.md) | Ideas pendientes por orden de utilidad | Antes de proponer una mejora |
| [../vault/README.md](../vault/README.md) | Decisiones, glosario y trampas del proyecto | Antes de cambiar algo no obvio |
