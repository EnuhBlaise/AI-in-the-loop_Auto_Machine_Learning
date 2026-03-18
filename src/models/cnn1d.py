"""1-D Convolutional classifier for tabular data."""

import torch
import torch.nn as nn


class CNN1DClassifier(nn.Module):
    """Treats each input feature as a spatial position in a single-channel 1-D signal.

    The input ``(batch, num_features)`` is reshaped to ``(batch, 1, num_features)``
    and passed through a stack of ``Conv1d -> BatchNorm1d -> Activation -> Dropout``
    blocks, followed by adaptive average pooling and a linear classification head.

    Parameters
    ----------
    num_features : int
        Number of input features.
    num_classes : int
        Number of output classes.
    config : dict
        The ``model.cnn1d`` section of the experiment configuration.  Expected
        keys (all optional):

        - ``channels``    : list[int] – output channels per conv layer (default [64, 128, 64])
        - ``kernel_size`` : int       – convolution kernel size        (default 3)
        - ``dropout``     : float     – dropout probability            (default 0.3)
        - ``pool``        : str       – pooling strategy               (default "adaptive_avg")
    """

    def __init__(self, num_features: int, num_classes: int, config: dict) -> None:
        super().__init__()

        channels: list[int] = config.get("channels", [64, 128, 64])
        kernel_size: int = config.get("kernel_size", 3)
        dropout: float = config.get("dropout", 0.3)
        padding: int = kernel_size // 2  # "same" padding

        # Convolutional blocks
        conv_layers: list[nn.Module] = []
        in_channels = 1
        for out_channels in channels:
            conv_layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding),
                nn.BatchNorm1d(out_channels),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            in_channels = out_channels

        self.conv_blocks = nn.Sequential(*conv_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(in_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : Tensor of shape ``(batch, num_features)``

        Returns
        -------
        Tensor of shape ``(batch, num_classes)`` – raw logits.
        """
        # (batch, features) -> (batch, 1, features)
        x = x.unsqueeze(1)
        x = self.conv_blocks(x)       # (batch, C_out, L)
        x = self.pool(x).squeeze(-1)  # (batch, C_out)
        return self.head(x)
