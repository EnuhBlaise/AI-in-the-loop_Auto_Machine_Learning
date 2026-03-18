"""Model factory with registry-based architecture selection."""

from __future__ import annotations

import torch.nn as nn

from src.models.mlp import MLPClassifier
from src.models.cnn1d import CNN1DClassifier
from src.models.transformer import TabTransformer
from src.models.ensemble import EnsembleModel


# ---------------------------------------------------------------------------
# Registry: maps architecture name -> (model class, config sub-key)
# ---------------------------------------------------------------------------
MODEL_REGISTRY: dict[str, type[nn.Module]] = {
    "mlp": MLPClassifier,
    "cnn1d": CNN1DClassifier,
    "transformer": TabTransformer,
}


class ModelFactory:
    """Creates model instances from a hierarchical configuration dict.

    The factory reads ``config["model"]["architecture"]`` to decide which
    architecture to instantiate.  For ``"ensemble"``, it builds every
    sub-model listed in ``config["model"]["ensemble"]["models"]`` and wraps
    them in an :class:`EnsembleModel`.

    Examples
    --------
    >>> model = ModelFactory.create(config, num_features=8, num_classes=10)
    """

    @staticmethod
    def create(config: dict, num_features: int, num_classes: int) -> nn.Module:
        """Build a model from *config*.

        Parameters
        ----------
        config : dict
            Full experiment configuration.  Must contain a ``"model"`` key
            with at least an ``"architecture"`` field.
        num_features : int
            Dimensionality of the input feature vector.
        num_classes : int
            Number of target classes.

        Returns
        -------
        nn.Module
            A model whose ``forward`` method maps
            ``(batch, num_features) -> (batch, num_classes)``.

        Raises
        ------
        ValueError
            If the requested architecture is not in :data:`MODEL_REGISTRY`.
        """
        model_cfg: dict = config["model"]
        architecture: str = model_cfg["architecture"]

        if architecture == "ensemble":
            return ModelFactory._create_ensemble(model_cfg, num_features, num_classes)

        if architecture not in MODEL_REGISTRY:
            raise ValueError(
                f"Unknown architecture '{architecture}'. "
                f"Available: {list(MODEL_REGISTRY) + ['ensemble']}"
            )

        model_cls = MODEL_REGISTRY[architecture]
        arch_config = model_cfg.get(architecture, {})
        return model_cls(num_features, num_classes, arch_config)

    @staticmethod
    def _create_ensemble(
        model_cfg: dict, num_features: int, num_classes: int
    ) -> EnsembleModel:
        """Build an ensemble from the sub-model list in *model_cfg*."""
        ensemble_cfg: dict = model_cfg.get("ensemble", {})
        sub_model_names: list[str] = ensemble_cfg.get("models", [])

        if not sub_model_names:
            raise ValueError(
                "Ensemble configuration must specify a non-empty 'models' list."
            )

        sub_models: list[nn.Module] = []
        for name in sub_model_names:
            if name not in MODEL_REGISTRY:
                raise ValueError(
                    f"Unknown sub-model '{name}' in ensemble. "
                    f"Available: {list(MODEL_REGISTRY)}"
                )
            cls = MODEL_REGISTRY[name]
            arch_config = model_cfg.get(name, {})
            sub_models.append(cls(num_features, num_classes, arch_config))

        return EnsembleModel(sub_models)
