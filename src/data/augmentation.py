"""
Data augmentation utilities for tabular Bio-ML training.
"""

import torch


def mixup_data(
    x: torch.Tensor,
    y_onehot: torch.Tensor,
    alpha: float = 0.2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply mixup augmentation to a batch of inputs and one-hot labels.

    Linearly interpolates between randomly shuffled pairs within the batch
    using a mixing coefficient drawn from a Beta distribution.

    Parameters
    ----------
    x : torch.Tensor
        Input feature tensor of shape ``(batch_size, num_features)``.
    y_onehot : torch.Tensor
        One-hot encoded label tensor of shape ``(batch_size, num_classes)``.
    alpha : float, optional
        Concentration parameter for the symmetric Beta distribution.
        Larger values produce more uniform mixing.  Default ``0.2``.

    Returns
    -------
    x_mixed : torch.Tensor
        Mixed input tensor, same shape as *x*.
    y_mixed : torch.Tensor
        Mixed one-hot label tensor, same shape as *y_onehot*.
    """
    if alpha > 0.0:
        lam = torch.distributions.Beta(alpha, alpha).sample()
    else:
        lam = torch.tensor(1.0)

    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)

    x_mixed = lam * x + (1.0 - lam) * x[index]
    y_mixed = lam * y_onehot + (1.0 - lam) * y_onehot[index]

    return x_mixed, y_mixed
