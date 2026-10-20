"""Tournament configuration presets (blitz) and their loader."""

# Import from the shared data classes
from chess_engine.engine.data_classes import TournamentConfig

# Preset configurations
BLITZ_CONFIG = {
    "total_game_time": 180.0,
    "time_increment": 2.0,
    "rounds": 1,
    "scoring": {"win": 1.0, "draw": 0.5, "loss": 0.0},
}


def load_config() -> TournamentConfig:
    """
    Load tournament configuration.

    :return: TournamentConfig object with default tournament settings
    :rtype: TournamentConfig
    """
    return TournamentConfig(**BLITZ_CONFIG)
