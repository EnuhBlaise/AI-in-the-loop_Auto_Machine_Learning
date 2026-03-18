"""Ensemble model that combines predictions from multiple sub-models."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EnsembleModel(nn.Module):
    """Wraps multiple models and combines their predictions via soft voting.

    Soft voting averages the softmax probability distributions produced by
    each sub-model and returns the corresponding logits (log-probabilities).

    Parameters
    ----------
    models : list[nn.Module]
        Pre-constructed sub-models.  Each must accept ``(batch, num_features)``
        and return logits of shape ``(batch, num_classes)``.
    """

    def __init__(self, models: list[nn.Module]) -> None:
        super().__init__()
        if not models:
            raise ValueError("EnsembleModel requires at least one sub-model.")
        self.models = nn.ModuleList(models)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass – soft vote over sub-model predictions.

        Parameters
        ----------
        x : Tensor of shape ``(batch, num_features)``

        Returns
        -------
        Tensor of shape ``(batch, num_classes)`` – averaged logits.
        """
        probs = [F.softmax(model(x), dim=-1) for model in self.models]
        avg_probs = torch.stack(probs, dim=0).mean(dim=0)
        # Return log-probabilities so downstream code can treat them as logits
        # (compatible with CrossEntropyLoss when wrapped with log_softmax offset).
        # We return raw averaged logits via log to keep the interface consistent.
        return torch.log(avg_probs + 1e-8)
