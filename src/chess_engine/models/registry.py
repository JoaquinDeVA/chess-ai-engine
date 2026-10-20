"""Name -> model class registry."""

from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Type

from .base import ARTIFACTS_DIR, EvaluationModel

_REGISTRY: Dict[str, Type[EvaluationModel]] = {}

#: Models that are not retrained nor played: only their accuracy and cost are measured.
ACCURACY_RECORDS = frozenset({"mlp_features", "cnn_resnet"})


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
        raise KeyError(
            f"unknown model {name!r}; available: {', '.join(available_models())}"
        ) from None


def split_spec(spec: str) -> Tuple[str, Optional[str]]:
    """Split a model specification ``name`` or ``name@tag``.

    A tag names a second artifact of the same model (e.g. trained on undersampled mates).

    :param spec: Registered model name, optionally followed by ``@`` and a tag.
    :return: ``(name, tag)``; the tag is None for an untagged name.
    :raises ValueError: If the tag after ``@`` or the name before it is empty.
    """
    name, separator, tag = spec.partition("@")
    if not separator:
        return name, None
    if not name or not tag:
        raise ValueError(f"model spec must look like name@tag, got {spec!r}")
    return name, tag


def artifact_path(spec: str) -> Path:
    """Where the artifact of a model specification is saved.

    :param spec: Model name, or ``name@tag``.
    :return: ``default_path()`` of the class for a name; ``models/artifacts/<spec><suffix>``
        for a tagged spec.
    :raises KeyError: If the name is unknown.
    """
    name, tag = split_spec(spec)
    cls = get_model_class(name)
    if tag is None:
        return cls.default_path()
    return ARTIFACTS_DIR / f"{spec}{cls.artifact_suffix}"


def load_model(spec: str, path: Optional[Path] = None) -> EvaluationModel:
    """Load the saved model named by ``spec``.

    :param spec: Registered model name, or ``name@tag`` for a tagged artifact.
    :param path: Saved file; defaults to :func:`artifact_path`.
    :return: The loaded model.
    :raises KeyError: If the name is unknown.
    :raises FileNotFoundError: If the model file does not exist (train it first).
    """
    name, _ = split_spec(spec)
    return get_model_class(name).load(path or artifact_path(spec))
