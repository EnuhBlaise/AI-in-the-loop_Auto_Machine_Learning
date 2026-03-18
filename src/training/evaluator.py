"""
Evaluation utilities for Bio-ML classification tasks.

Computes AUC-ROC (macro, one-vs-rest), macro F1, and accuracy on a given split.
"""

import logging
from typing import Dict

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

logger = logging.getLogger(__name__)


class Evaluator:
    """Stateless evaluator that computes classification metrics.

    Usage
    -----
    >>> evaluator = Evaluator()
    >>> metrics = evaluator.evaluate(y_true, y_pred_proba, y_pred, split="val")
    """

    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        y_pred: np.ndarray,
        split: str = "val",
    ) -> Dict[str, float]:
        """Compute classification metrics for a single data split.

        Parameters
        ----------
        y_true : np.ndarray
            Ground-truth integer labels of shape ``(n_samples,)``.
        y_pred_proba : np.ndarray
            Predicted probability matrix of shape ``(n_samples, num_classes)``.
        y_pred : np.ndarray
            Predicted integer labels of shape ``(n_samples,)``.
        split : str
            Name prefix for the returned metric keys (e.g. ``"val"``, ``"test"``).

        Returns
        -------
        dict
            ``{"{split}_auroc": float, "{split}_f1": float, "{split}_accuracy": float}``
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        y_pred_proba = np.asarray(y_pred_proba)

        # --- Accuracy ---
        acc = accuracy_score(y_true, y_pred)

        # --- Macro F1 ---
        f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

        # --- AUC-ROC (macro, one-vs-rest) ---
        auroc = self._safe_auroc(y_true, y_pred_proba)

        metrics = {
            f"{split}_auroc": round(float(auroc), 6),
            f"{split}_f1": round(float(f1), 6),
            f"{split}_accuracy": round(float(acc), 6),
        }
        return metrics

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_auroc(y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        """Compute AUC-ROC with graceful handling of edge cases.

        Edge cases handled:
        - Only one class present in ``y_true`` (AUC undefined).
        - Probability matrix has fewer columns than unique labels.
        - Constant predictions.
        """
        unique_classes = np.unique(y_true)

        if len(unique_classes) < 2:
            logger.warning(
                "AUC-ROC is undefined when only one class is present in y_true. "
                "Returning 0.0."
            )
            return 0.0

        # Binary case: use column 1 probabilities if available
        if len(unique_classes) == 2 and y_pred_proba.ndim == 2:
            if y_pred_proba.shape[1] == 2:
                try:
                    return float(roc_auc_score(y_true, y_pred_proba[:, 1]))
                except ValueError:
                    return 0.0
            elif y_pred_proba.shape[1] == 1:
                try:
                    return float(roc_auc_score(y_true, y_pred_proba[:, 0]))
                except ValueError:
                    return 0.0

        # Multiclass case
        try:
            return float(
                roc_auc_score(
                    y_true,
                    y_pred_proba,
                    multi_class="ovr",
                    average="macro",
                )
            )
        except ValueError as exc:
            logger.warning("AUC-ROC computation failed (%s). Returning 0.0.", exc)
            return 0.0
