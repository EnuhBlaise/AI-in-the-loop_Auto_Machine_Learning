"""
Training callbacks: early stopping and model checkpointing.
"""

import copy
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class EarlyStopping:
    """Stop training when a monitored metric stops improving.

    Parameters
    ----------
    patience : int
        Number of evaluations with no improvement before stopping.
    metric : str
        Name of the metric to monitor (for logging only).
    mode : str
        ``"max"`` if higher is better, ``"min"`` if lower is better.
    min_delta : float
        Minimum change to qualify as an improvement.

    Examples
    --------
    >>> es = EarlyStopping(patience=10, mode="max")
    >>> should_stop = es.step(0.85)
    """

    def __init__(
        self,
        patience: int = 30,
        metric: str = "val_auroc",
        mode: str = "max",
        min_delta: float = 0.0,
    ) -> None:
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got '{mode}'")

        self.patience = patience
        self.metric = metric
        self.mode = mode
        self.min_delta = min_delta

        self._best_score: Optional[float] = None
        self._counter: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def step(self, metric_value: float) -> bool:
        """Record a new metric value and return whether training should stop.

        Parameters
        ----------
        metric_value : float
            Current value of the monitored metric.

        Returns
        -------
        bool
            ``True`` if training should stop (patience exhausted).
        """
        if self._best_score is None:
            self._best_score = metric_value
            return False

        if self._is_improvement(metric_value):
            self._best_score = metric_value
            self._counter = 0
        else:
            self._counter += 1
            if self._counter >= self.patience:
                logger.info(
                    "Early stopping triggered: no improvement in %s for %d evaluations.",
                    self.metric,
                    self.patience,
                )
                return True
        return False

    @property
    def best_score(self) -> Optional[float]:
        """Best metric value observed so far."""
        return self._best_score

    @property
    def counter(self) -> int:
        """Number of evaluations since last improvement."""
        return self._counter

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "EarlyStopping":
        """Create an ``EarlyStopping`` instance from the ``training.early_stopping``
        section of the experiment config.

        Parameters
        ----------
        config : dict
            Must contain keys ``patience``, ``metric``, ``mode``.
        """
        return cls(
            patience=config.get("patience", 30),
            metric=config.get("metric", "val_auroc"),
            mode=config.get("mode", "max"),
            min_delta=config.get("min_delta", 0.0),
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _is_improvement(self, value: float) -> bool:
        if self.mode == "max":
            return value > self._best_score + self.min_delta  # type: ignore[operator]
        return value < self._best_score - self.min_delta  # type: ignore[operator]


class ModelCheckpoint:
    """Save the best model state dict when a monitored metric improves.

    Parameters
    ----------
    save_path : str or Path or None
        File path to persist the checkpoint. If ``None``, the best state dict
        is kept in memory only (no disk I/O).
    metric : str
        Name of the metric to monitor (for logging).
    mode : str
        ``"max"`` if higher is better, ``"min"`` if lower is better.
    min_delta : float
        Minimum change to qualify as an improvement.
    """

    def __init__(
        self,
        save_path: Optional[str] = None,
        metric: str = "val_auroc",
        mode: str = "max",
        min_delta: float = 0.0,
    ) -> None:
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got '{mode}'")

        self.save_path = Path(save_path) if save_path else None
        self.metric = metric
        self.mode = mode
        self.min_delta = min_delta

        self._best_score: Optional[float] = None
        self._best_state: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def step(self, metric_value: float, model: nn.Module) -> None:
        """Check if the metric improved and, if so, save the model.

        Parameters
        ----------
        metric_value : float
            Current value of the monitored metric.
        model : nn.Module
            Model whose ``state_dict`` should be saved on improvement.
        """
        if self._best_score is None or self._is_improvement(metric_value):
            self._best_score = metric_value
            self._best_state = copy.deepcopy(model.state_dict())

            if self.save_path is not None:
                self.save_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(self._best_state, self.save_path)
                logger.debug(
                    "ModelCheckpoint: saved new best %s=%.6f to %s",
                    self.metric,
                    metric_value,
                    self.save_path,
                )
            else:
                logger.debug(
                    "ModelCheckpoint: new best %s=%.6f (in-memory).",
                    self.metric,
                    metric_value,
                )

    @property
    def best_score(self) -> Optional[float]:
        """Best metric value observed so far."""
        return self._best_score

    @property
    def best_state(self) -> Optional[Dict[str, Any]]:
        """State dict of the best model observed so far."""
        return self._best_state

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config: Dict[str, Any], save_path: Optional[str] = None) -> "ModelCheckpoint":
        """Create a ``ModelCheckpoint`` from the ``training.early_stopping``
        config section (shares the same metric/mode settings).

        Parameters
        ----------
        config : dict
            Must contain keys ``metric`` and ``mode``.
        save_path : str or None
            Where to save the checkpoint on disk.
        """
        return cls(
            save_path=save_path,
            metric=config.get("metric", "val_auroc"),
            mode=config.get("mode", "max"),
            min_delta=config.get("min_delta", 0.0),
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _is_improvement(self, value: float) -> bool:
        if self.mode == "max":
            return value > self._best_score + self.min_delta  # type: ignore[operator]
        return value < self._best_score - self.min_delta  # type: ignore[operator]
