"""Name -> model class registry."""

from pathlib import Path
from typing import Callable, Dict, List, Optional, Type

from .base import EvaluationModel

_REGISTRY: Dict[str, Type[EvaluationModel]] = {}


def register_model(name: str) -> Callable[[Type[EvaluationModel]], Type[EvaluationModel]]:
    """Class decorator that makes a model available under ``name``.

    :param name: Unique identifier, also used for artifact and card file names.
    :return: The decorator.
    :raises ValueError: If ``name`` is already registered to a different class.
    """

    def decorator(cls: Type[EvaluationModel]) -> Type[EvaluationModel]:
        existing = _REGISTRY.get(name)
        if existing is not None and existing is not cls:
            raise ValueError(f"model name {name!r} is already registered to {existing.__name__}")
        cls.name = name
        _REGISTRY[name] = cls
        return cls

    return decorator


def available_models() -> List[str]:
    """Names of all registered models.

    :return: Sorted list of names.
    """
    return sorted(_REGISTRY)


def get_model_class(name: str) -> Type[EvaluationModel]:
    """Look a model class up by name.

    :param name: Registered model name.
    :return: The model class.
    :raises KeyError: With the list of valid names if ``name`` is unknown.
    """
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(f"unknown model {name!r}; available: {', '.join(available_models())}") from None


def load_model(name: str, path: Optional[Path] = None) -> EvaluationModel:
    """Load the saved model registered under ``name``.

    :param name: Registered model name.
    :param path: Saved file; defaults to ``models/artifacts/<name>.<ext>``.
    :return: The loaded model.
    :raises KeyError: If ``name`` is unknown.
    :raises FileNotFoundError: If the model file does not exist (train it first).
    """
    return get_model_class(name).load(path)
