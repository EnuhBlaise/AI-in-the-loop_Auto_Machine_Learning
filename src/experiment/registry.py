"""Lightweight experiment registry for fast duplicate detection."""

import threading
from typing import Any


# Keys used to determine config similarity.
_SIMILARITY_KEYS = ("architecture", "hidden_dims", "channels", "lr", "batch_size")


def _get_nested(d: dict, dotted_key: str, default=None):
    """Get a value from a nested dict using a dot-separated key."""
    keys = dotted_key.split(".")
    for k in keys:
        if not isinstance(d, dict):
            return default
        d = d.get(k, default)
    return d


# Dot-separated paths into the nested config for fingerprinting.
_SIMILARITY_PATHS = (
    "model.architecture",
    "model.mlp.hidden_dims",
    "model.cnn1d.channels",
    "model.transformer.d_model",
    "training.lr",
    "training.batch_size",
)


def _config_fingerprint(config: dict) -> tuple:
    """Produce a hashable fingerprint from the similarity-relevant fields.

    Two configs are considered duplicates when they share the same
    architecture, hidden_dims/channels, learning rate, and batch size.
    """
    parts: list[Any] = []
    for path in _SIMILARITY_PATHS:
        value = _get_nested(config, path)
        # Convert mutable types (lists, dicts) to something hashable.
        if isinstance(value, list):
            value = tuple(value)
        elif isinstance(value, dict):
            value = tuple(sorted(value.items()))
        parts.append((path, value))
    return tuple(parts)


class ExperimentRegistry:
    """In-memory index for rapid experiment look-ups and duplicate checking.

    Thread-safe: all mutations are guarded by an internal lock.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # Set of fingerprints for O(1) duplicate lookups.
        self._fingerprints: set[tuple] = set()
        # Full index keyed by experiment id.
        self._index: dict[str, dict] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, experiment_id: str, config: dict) -> None:
        """Add an experiment to the in-memory index.

        Args:
            experiment_id: Unique experiment identifier (e.g. "exp_001").
            config: The experiment configuration dictionary.
        """
        fp = _config_fingerprint(config)
        with self._lock:
            self._fingerprints.add(fp)
            self._index[experiment_id] = {"id": experiment_id, "config": config}

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def has_been_tried(self, config: dict) -> bool:
        """Check whether a similar configuration has already been run.

        Similarity is determined by matching: architecture, hidden_dims,
        channels, lr, and batch_size.

        Args:
            config: Candidate configuration to test.

        Returns:
            True if a matching config exists in the registry.
        """
        fp = _config_fingerprint(config)
        with self._lock:
            return fp in self._fingerprints

    def get(self, experiment_id: str) -> dict | None:
        """Retrieve a registered experiment entry by id."""
        with self._lock:
            return self._index.get(experiment_id)

    def all_experiments(self) -> list[dict]:
        """Return a list of all registered experiments (id + config)."""
        with self._lock:
            return list(self._index.values())

    @property
    def count(self) -> int:
        """Number of registered experiments."""
        with self._lock:
            return len(self._index)

    # ------------------------------------------------------------------
    # Bulk loading (e.g. restoring state from disk)
    # ------------------------------------------------------------------

    def load_from_history(self, history: list[dict]) -> None:
        """Populate the registry from a list of experiment summaries.

        Useful for restoring the in-memory index after a restart by
        passing the output of ``ExperimentTracker.get_history()``.

        Args:
            history: List of dicts, each with at least "id" and "config".
        """
        with self._lock:
            for entry in history:
                exp_id = entry.get("id", "")
                config = entry.get("config", {})
                fp = _config_fingerprint(config)
                self._fingerprints.add(fp)
                self._index[exp_id] = {"id": exp_id, "config": config}
