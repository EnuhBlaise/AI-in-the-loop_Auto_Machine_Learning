"""Transformer-based classifier for tabular data (TabTransformer variant)."""

import math

import torch
import torch.nn as nn


class TabTransformer(nn.Module):
    """Projects each scalar feature into a ``d_model``-dimensional embedding,
    prepends a learnable [CLS] token, and processes the sequence through a
    standard ``TransformerEncoder``.  The [CLS] representation is fed to a
    linear classification head.

    Parameters
    ----------
    num_features : int
        Number of input features.
    num_classes : int
        Number of output classes.
    config : dict
        The ``model.transformer`` section of the experiment configuration.
        Expected keys (all optional):

        - ``d_model``         : int   – embedding dimension          (default 64)
        - ``n_heads``         : int   – number of attention heads    (default 4)
        - ``n_layers``        : int   – number of encoder layers     (default 2)
        - ``dim_feedforward`` : int   – FFN hidden dimension         (default 128)
        - ``dropout``         : float – dropout probability          (default 0.2)
    """

    def __init__(self, num_features: int, num_classes: int, config: dict) -> None:
        super().__init__()

        d_model: int = config.get("d_model", 64)
        n_heads: int = config.get("n_heads", 4)
        n_layers: int = config.get("n_layers", 2)
        dim_feedforward: int = config.get("dim_feedforward", 128)
        dropout: float = config.get("dropout", 0.2)

        self.d_model = d_model
        self.num_features = num_features

        # Per-feature linear embeddings: each scalar -> d_model vector
        self.feature_embeddings = nn.Linear(1, d_model)

        # Learnable [CLS] token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        # Positional encoding (learnable, length = num_features + 1 for CLS)
        self.pos_embedding = nn.Parameter(
            torch.randn(1, num_features + 1, d_model) * 0.02
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
        )

        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier-uniform initialisation for linear layers."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : Tensor of shape ``(batch, num_features)``

        Returns
        -------
        Tensor of shape ``(batch, num_classes)`` – raw logits.
        """
        batch_size = x.size(0)

        # (batch, num_features) -> (batch, num_features, 1) -> (batch, num_features, d_model)
        x = x.unsqueeze(-1)
        x = self.feature_embeddings(x)

        # Prepend [CLS] token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (batch, num_features + 1, d_model)

        # Add positional encoding
        x = x + self.pos_embedding

        # Transformer encoder
        x = self.encoder(x)

        # Extract [CLS] representation
        cls_out = self.norm(x[:, 0, :])  # (batch, d_model)
        return self.head(cls_out)
