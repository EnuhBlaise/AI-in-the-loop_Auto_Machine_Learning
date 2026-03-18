"""Experiment tracking system for Bio-ML autonomous research platform."""

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml


class ExperimentTracker:
    """Tracks experiments, persists configs/metrics/models, and provides querying."""

    def __init__(self, base_dir: str = "experiments") -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _next_experiment_id(self) -> str:
        """Return the next sequential experiment id (e.g. 'exp_001')."""
        existing = sorted(self.base_dir.glob("exp_*"))
        if not existing:
            return "exp_001"
        last = existing[-1].name  # e.g. "exp_012"
        last_num = int(last.split("_")[1])
        return f"exp_{last_num + 1:03d}"

    def _exp_dir(self, experiment_id: str) -> Path:
        return self.base_dir / experiment_id

    # ------------------------------------------------------------------
    # Core lifecycle
    # ------------------------------------------------------------------

    def start_experiment(self, config: dict) -> str:
        """Create a new experiment directory and persist its config.

        Args:
            config: Hyperparameter / architecture configuration dict.

        Returns:
            experiment_id: String like "exp_001".
        """
        with self._lock:
            experiment_id = self._next_experiment_id()
            exp_dir = self._exp_dir(experiment_id)
            exp_dir.mkdir(parents=True, exist_ok=True)

        config_path = exp_dir / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        return experiment_id

    def log_metrics(self, experiment_id: str, metrics: dict) -> None:
        """Save metrics to the experiment directory.

        Args:
            experiment_id: Target experiment (e.g. "exp_001").
            metrics: Dictionary of metric name -> value.
        """
        exp_dir = self._exp_dir(experiment_id)
        if not exp_dir.exists():
            raise FileNotFoundError(f"Experiment directory not found: {exp_dir}")

        metrics_path = exp_dir / "metrics.json"

        with self._lock:
            # Append-safe: merge with any existing metrics file.
            existing: dict[str, Any] = {}
            if metrics_path.exists():
                existing = json.loads(metrics_path.read_text())
            existing.update(metrics)
            metrics_path.write_text(json.dumps(existing, indent=2))

    def log_model(self, experiment_id: str, model_state_dict: dict) -> None:
        """Persist the best model checkpoint.

        Args:
            experiment_id: Target experiment.
            model_state_dict: PyTorch state dict (from model.state_dict()).
        """
        exp_dir = self._exp_dir(experiment_id)
        if not exp_dir.exists():
            raise FileNotFoundError(f"Experiment directory not found: {exp_dir}")

        model_path = exp_dir / "best_model.pt"
        torch.save(model_state_dict, model_path)

    def finish_experiment(
        self,
        experiment_id: str,
        metrics: dict,
        duration: float,
    ) -> None:
        """Finalise an experiment by writing a summary file.

        Args:
            experiment_id: Target experiment.
            metrics: Final metrics dict.
            duration: Wall-clock training time in seconds.
        """
        exp_dir = self._exp_dir(experiment_id)
        if not exp_dir.exists():
            raise FileNotFoundError(f"Experiment directory not found: {exp_dir}")

        # Load the config that was saved at start time.
        config_path = exp_dir / "config.yaml"
        config: dict = {}
        if config_path.exists():
            with open(config_path) as f:
                config = yaml.safe_load(f) or {}

        summary = {
            "id": experiment_id,
            "config": config,
            "metrics": metrics,
            "duration": duration,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        summary_path = exp_dir / "summary.json"
        with self._lock:
            summary_path.write_text(json.dumps(summary, indent=2))

        # Also persist the final metrics file for consistency.
        self.log_metrics(experiment_id, metrics)

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def get_history(self) -> list[dict]:
        """Return all experiment summaries sorted by experiment id.

        Each entry contains: id, config, metrics, duration, timestamp.
        """
        summaries: list[dict] = []
        for summary_path in sorted(self.base_dir.glob("exp_*/summary.json")):
            try:
                summary = json.loads(summary_path.read_text())
                summaries.append(summary)
            except (json.JSONDecodeError, OSError):
                continue
        return summaries

    def get_leaderboard(
        self, metric: str = "val_auroc", top_n: int = 10
    ) -> list[dict]:
        """Return experiments sorted by *metric* in descending order.

        Args:
            metric: The metric key to rank by.
            top_n: Maximum number of entries to return.
        """
        history = self.get_history()
        # Filter to experiments that actually have the requested metric.
        with_metric = [
            entry for entry in history if metric in entry.get("metrics", {})
        ]
        ranked = sorted(
            with_metric,
            key=lambda e: e["metrics"][metric],
            reverse=True,
        )
        return ranked[:top_n]

    def get_best_experiment(self, metric: str = "val_auroc") -> dict:
        """Return the single best experiment summary by *metric*.

        Returns an empty dict if no experiments have the metric.
        """
        leaderboard = self.get_leaderboard(metric=metric, top_n=1)
        if not leaderboard:
            return {}
        return leaderboard[0]

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def save_report(self, report: str) -> None:
        """Write a final report to ``base_dir/final_report.md``."""
        report_path = self.base_dir / "final_report.md"
        report_path.write_text(report)

    def format_history_for_llm(self) -> str:
        """Return a concise, human-readable summary of all experiments.

        Designed to be injected into an LLM prompt so the model can reason
        about past experiments and propose improvements.
        """
        history = self.get_history()
        if not history:
            return "No experiments have been run yet."

        lines: list[str] = ["=== Experiment History ===", ""]
        for entry in history:
            exp_id = entry.get("id", "unknown")
            config = entry.get("config", {})
            metrics = entry.get("metrics", {})
            duration = entry.get("duration", 0.0)

            # Extract key config values from nested structure.
            model_cfg = config.get("model", {})
            train_cfg = config.get("training", {})
            data_cfg = config.get("data", {})
            arch = model_cfg.get("architecture", "?")

            config_parts: list[str] = [f"arch={arch}"]

            # Architecture-specific params
            arch_cfg = model_cfg.get(arch, {})
            if "hidden_dims" in arch_cfg:
                config_parts.append(f"hidden_dims={arch_cfg['hidden_dims']}")
            if "channels" in arch_cfg:
                config_parts.append(f"channels={arch_cfg['channels']}")
            if "d_model" in arch_cfg:
                config_parts.append(f"d_model={arch_cfg['d_model']}")
            if "dropout" in arch_cfg:
                config_parts.append(f"dropout={arch_cfg['dropout']}")

            # Training params
            for key in ("lr", "batch_size", "optimizer", "scheduler", "weight_decay", "label_smoothing"):
                if key in train_cfg:
                    config_parts.append(f"{key}={train_cfg[key]}")

            # Data params
            if "scaler" in data_cfg:
                config_parts.append(f"scaler={data_cfg['scaler']}")
            aug = data_cfg.get("augmentation", {})
            if aug.get("mixup"):
                config_parts.append(f"mixup_alpha={aug.get('mixup_alpha', 0.2)}")

            # Format metrics.
            metric_parts = [
                f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                for k, v in metrics.items()
            ]

            lines.append(f"[{exp_id}] duration={duration:.1f}s")
            lines.append(f"  Config: {', '.join(config_parts)}")
            lines.append(f"  Metrics: {', '.join(metric_parts) or 'N/A'}")
            lines.append("")

        return "\n".join(lines)
