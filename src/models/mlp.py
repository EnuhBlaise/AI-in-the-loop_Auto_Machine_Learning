"""Multi-Layer Perceptron classifier with optional residual connections and batch normalization."""

import torch
import torch.nn as nn


_ACTIVATIONS = {
    "gelu": nn.GELU,
    "relu": nn.ReLU,
    "silu": nn.SiLU,
}


class MLPClassifier(nn.Module):
    """Fully-connected classifier with configurable depth, width, and regularisation.

    Parameters
    ----------
    num_features : int
        Number of input features.
    num_classes : int
        Number of output classes.
    config : dict
        The ``model.mlp`` section of the experiment configuration.  Expected
        keys (all optional, sensible defaults are provided):

        - ``hidden_dims``  : list[int]  – widths of hidden layers (default [128, 64, 32])
        - ``dropout``      : float      – dropout probability            (default 0.3)
        - ``activation``   : str        – one of gelu / relu / silu      (default "gelu")
        - ``use_residual`` : bool       – add skip connections            (default True)
        - ``use_batchnorm``: bool       – add BatchNorm1d after linear    (default True)
    """

    def __init__(self, num_features: int, num_classes: int, config: dict) -> None:
        super().__init__()

        hidden_dims: list[int] = config.get("hidden_dims", [128, 64, 32])
        dropout: float = config.get("dropout", 0.3)
        activation_name: str = config.get("activation", "gelu")
        self.use_residual: bool = config.get("use_residual", True)
        use_batchnorm: bool = config.get("use_batchnorm", True)

        activation_cls = _ACTIVATIONS.get(activation_name)
        if activation_cls is None:
            raise ValueError(
                f"Unsupported activation '{activation_name}'. "
                f"Choose from {list(_ACTIVATIONS)}"
            )

        # Build hidden blocks
        self.blocks = nn.ModuleList()
        self.residual_projections = nn.ModuleList()

        in_dim = num_features
        for out_dim in hidden_dims:
            layers: list[nn.Module] = [nn.Linear(in_dim, out_dim)]
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(out_dim))
            layers.append(activation_cls())
            layers.append(nn.Dropout(dropout))
            self.blocks.append(nn.Sequential(*layers))

            # Projection for residual when dimensions differ
            if self.use_residual:
                if in_dim != out_dim:
                    self.residual_projections.append(nn.Linear(in_dim, out_dim))
                else:
                    self.residual_projections.append(nn.Identity())

            in_dim = out_dim

        # Classification head
        self.head = nn.Linear(in_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : Tensor of shape ``(batch, num_features)``

        Returns
        -------
        Tensor of shape ``(batch, num_classes)`` – raw logits.
        """
        for idx, block in enumerate(self.blocks):
            identity = x
            x = block(x)
            if self.use_residual:
                x = x + self.residual_projections[idx](identity)
        return self.head(x)
