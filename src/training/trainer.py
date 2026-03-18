"""
Training loop for Bio-ML classification models.

Handles optimiser/scheduler construction, mixup augmentation, gradient clipping,
periodic evaluation, early stopping, model checkpointing, and time-budget enforcement.
"""

import logging
import time
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.data.augmentation import mixup_data
from src.training.callbacks import EarlyStopping, ModelCheckpoint
from src.training.evaluator import Evaluator

logger = logging.getLogger(__name__)

# Optional rich progress bar -- degrade gracefully if not installed.
try:
    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        TextColumn,
        TimeRemainingColumn,
    )

    _RICH_AVAILABLE = True
except ImportError:
    _RICH_AVAILABLE = False


class Trainer:
    """Encapsulates a full training run for a PyTorch classification model.

    Parameters
    ----------
    model : nn.Module
        The network to train (already moved to *device* is fine, but the trainer
        will call ``.to(device)`` for safety).
    config : dict
        Full experiment config dict.  The trainer reads ``training``, ``evaluation``,
        ``experiment``, and ``data.augmentation`` sections.
    device : torch.device or str
        Target device (``"cpu"``, ``"cuda"``, ``"mps"``).
    """

    def __init__(
        self,
        model: nn.Module,
        config: Dict[str, Any],
        device: torch.device | str = "cpu",
    ) -> None:
        self.model = model
        self.config = config
        self.device = torch.device(device)

        self._train_cfg = config.get("training", {})
        self._eval_cfg = config.get("evaluation", {})
        self._exp_cfg = config.get("experiment", {})
        self._aug_cfg = config.get("data", {}).get("augmentation", {})

        self._best_model_state: Optional[Dict[str, Any]] = None
        self._evaluator = Evaluator()
        self._history: Dict[str, list] = {
            "epoch": [],
            "train_loss": [],
            "val_auroc": [],
            "val_f1": [],
            "val_accuracy": [],
            "lr": [],
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(
        self,
        train_loader: DataLoader,
        X_val: np.ndarray,
        y_val: np.ndarray,
        num_classes: int,
        class_weights: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """Run the full training loop.

        Parameters
        ----------
        train_loader : DataLoader
            Training data loader (batches of ``(X, y)`` tensors).
        X_val : np.ndarray
            Validation features as a NumPy array.
        y_val : np.ndarray
            Validation labels as a NumPy integer array.
        num_classes : int
            Number of target classes.
        class_weights : torch.Tensor or None
            Optional per-class weights for the loss function.

        Returns
        -------
        dict
            Best validation metrics observed during training.
        """
        self.model = self.model.to(self.device)

        epochs: int = self._train_cfg.get("epochs", 200)
        eval_interval: int = self._eval_cfg.get("eval_interval", 5)
        time_budget: float = self._exp_cfg.get("time_budget", 300)
        primary_metric: str = self._eval_cfg.get("primary_metric", "val_auroc")

        # Build training components
        optimizer = self._build_optimizer()
        scheduler = self._build_scheduler(optimizer, epochs, len(train_loader))
        loss_fn = self._build_loss(class_weights)

        # Callbacks
        es_cfg = self._train_cfg.get("early_stopping", {})
        early_stopping_enabled = es_cfg.get("enabled", True)
        early_stopper = EarlyStopping.from_config(es_cfg) if early_stopping_enabled else None
        checkpointer = ModelCheckpoint.from_config(es_cfg)

        # Mixup settings
        use_mixup = self._aug_cfg.get("mixup", False)
        mixup_alpha = self._aug_cfg.get("mixup_alpha", 0.2)

        # Gradient clipping
        grad_clip = self._train_cfg.get("gradient_clip", 0.0)

        # Mixed precision
        use_amp = self._train_cfg.get("mixed_precision", False) and self.device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda") if use_amp else None

        best_metrics: Dict[str, float] = {}
        start_time = time.monotonic()

        logger.info(
            "Starting training: epochs=%d, optimizer=%s, lr=%.6f, device=%s",
            epochs,
            self._train_cfg.get("optimizer", "adamw"),
            self._train_cfg.get("lr", 1e-3),
            self.device,
        )

        progress_ctx = self._make_progress_bar(epochs)

        with progress_ctx as progress:
            task_id = progress.add_task("Training", total=epochs) if progress is not None else None

            for epoch in range(1, epochs + 1):
                # --- Time budget check ---
                elapsed = time.monotonic() - start_time
                if elapsed >= time_budget:
                    logger.info(
                        "Time budget exhausted (%.1fs / %.1fs). Stopping at epoch %d.",
                        elapsed,
                        time_budget,
                        epoch,
                    )
                    break

                # --- Train one epoch ---
                train_loss = self._train_epoch(
                    train_loader,
                    optimizer,
                    loss_fn,
                    num_classes,
                    use_mixup,
                    mixup_alpha,
                    grad_clip,
                    use_amp,
                    scaler,
                )

                # --- Step scheduler (except ReduceLROnPlateau, handled after eval) ---
                scheduler_name = self._train_cfg.get("scheduler", "cosine").lower()
                if scheduler is not None and scheduler_name != "plateau":
                    scheduler.step()

                # --- Periodic validation ---
                if epoch % eval_interval == 0 or epoch == epochs:
                    metrics = self._validate(X_val, y_val, num_classes)
                    metric_value = metrics.get(primary_metric, 0.0)

                    # Record history
                    current_lr = optimizer.param_groups[0]["lr"]
                    self._history["epoch"].append(epoch)
                    self._history["train_loss"].append(train_loss)
                    self._history["val_auroc"].append(metrics.get("val_auroc", 0.0))
                    self._history["val_f1"].append(metrics.get("val_f1", 0.0))
                    self._history["val_accuracy"].append(metrics.get("val_accuracy", 0.0))
                    self._history["lr"].append(current_lr)

                    logger.info(
                        "Epoch %d/%d  train_loss=%.4f  %s",
                        epoch,
                        epochs,
                        train_loss,
                        "  ".join(f"{k}={v:.4f}" for k, v in metrics.items()),
                    )

                    # ReduceLROnPlateau steps on metric
                    if scheduler is not None and scheduler_name == "plateau":
                        scheduler.step(metric_value)

                    # Checkpoint
                    checkpointer.step(metric_value, self.model)

                    # Early stopping
                    if early_stopper is not None and early_stopper.step(metric_value):
                        break

                    # Track best
                    if not best_metrics or self._is_better(
                        metric_value,
                        best_metrics.get(primary_metric, None),
                        es_cfg.get("mode", "max"),
                    ):
                        best_metrics = metrics.copy()

                if progress is not None:
                    progress.update(task_id, advance=1)

        # Store best model state
        self._best_model_state = checkpointer.best_state

        elapsed_total = time.monotonic() - start_time
        logger.info(
            "Training complete in %.1fs.  Best: %s",
            elapsed_total,
            "  ".join(f"{k}={v:.4f}" for k, v in best_metrics.items()),
        )

        return best_metrics

    @property
    def best_model_state(self) -> Optional[Dict[str, Any]]:
        """State dict of the best-performing model (by primary metric)."""
        return self._best_model_state

    @property
    def history(self) -> Dict[str, list]:
        """Per-epoch training history (epoch, train_loss, val metrics, lr)."""
        return self._history

    # ------------------------------------------------------------------
    # Training helpers
    # ------------------------------------------------------------------

    def _train_epoch(
        self,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
        num_classes: int,
        use_mixup: bool,
        mixup_alpha: float,
        grad_clip: float,
        use_amp: bool,
        scaler: Optional[torch.amp.GradScaler],
    ) -> float:
        """Train for one epoch, returning average loss."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for X_batch, y_batch in loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            if use_mixup:
                # Convert labels to one-hot for mixup
                y_onehot = F.one_hot(y_batch, num_classes).float()
                X_batch, y_mixed = mixup_data(X_batch, y_onehot, alpha=mixup_alpha)

            optimizer.zero_grad()

            amp_device = "cuda" if self.device.type == "cuda" else "cpu"
            with torch.autocast(device_type=amp_device, enabled=use_amp):
                logits = self.model(X_batch)

                if use_mixup:
                    # Soft cross-entropy for mixed labels
                    log_probs = F.log_softmax(logits, dim=1)
                    loss = -(y_mixed * log_probs).sum(dim=1).mean()
                else:
                    loss = loss_fn(logits, y_batch)

            if scaler is not None:
                scaler.scale(loss).backward()
                if grad_clip > 0:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if grad_clip > 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def _validate(
        self,
        X_val: np.ndarray,
        y_val: np.ndarray,
        num_classes: int,
    ) -> Dict[str, float]:
        """Run validation and return metrics dict."""
        self.model.eval()

        X_tensor = torch.tensor(X_val, dtype=torch.float32, device=self.device)
        logits = self.model(X_tensor)
        proba = F.softmax(logits, dim=1).cpu().numpy()
        preds = np.argmax(proba, axis=1)

        return self._evaluator.evaluate(y_val, proba, preds, split="val")

    # ------------------------------------------------------------------
    # Component builders
    # ------------------------------------------------------------------

    def _build_optimizer(self) -> torch.optim.Optimizer:
        name = self._train_cfg.get("optimizer", "adamw").lower()
        lr = self._train_cfg.get("lr", 1e-3)
        weight_decay = self._train_cfg.get("weight_decay", 1e-4)

        params = self.model.parameters()

        if name == "adam":
            return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
        elif name == "adamw":
            return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
        elif name == "sgd":
            momentum = self._train_cfg.get("momentum", 0.9)
            return torch.optim.SGD(
                params, lr=lr, momentum=momentum, weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer: {name}")

    def _build_scheduler(
        self,
        optimizer: torch.optim.Optimizer,
        epochs: int,
        steps_per_epoch: int,
    ) -> Optional[torch.optim.lr_scheduler.LRScheduler]:
        name = self._train_cfg.get("scheduler", "cosine").lower()
        params = self._train_cfg.get("scheduler_params", {})

        if name == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=params.get("T_max", epochs),
                eta_min=params.get("eta_min", 1e-6),
            )
        elif name == "step":
            return torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=params.get("step_size", 50),
                gamma=params.get("gamma", 0.1),
            )
        elif name == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=params.get("mode", "max"),
                factor=params.get("factor", 0.1),
                patience=params.get("patience", 10),
            )
        elif name == "onecycle":
            return torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=params.get("max_lr", self._train_cfg.get("lr", 1e-3) * 10),
                epochs=epochs,
                steps_per_epoch=steps_per_epoch,
            )
        elif name == "none":
            return None
        else:
            raise ValueError(f"Unsupported scheduler: {name}")

    def _build_loss(self, class_weights: Optional[torch.Tensor] = None) -> nn.Module:
        label_smoothing = self._train_cfg.get("label_smoothing", 0.0)
        use_class_weights = self._train_cfg.get("class_weighting", False)

        weight = None
        if use_class_weights and class_weights is not None:
            weight = class_weights.to(self.device)

        return nn.CrossEntropyLoss(weight=weight, label_smoothing=label_smoothing)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _is_better(
        current: float, best: Optional[float], mode: str
    ) -> bool:
        if best is None:
            return True
        if mode == "max":
            return current > best
        return current < best

    @staticmethod
    def _make_progress_bar(total_epochs: int):
        """Return a rich Progress context manager, or a no-op if rich is unavailable."""
        if _RICH_AVAILABLE:
            return Progress(
                TextColumn("[bold blue]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TimeRemainingColumn(),
            )
        return _NoOpProgress()


class _NoOpProgress:
    """Minimal stand-in when rich is not installed."""

    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False
