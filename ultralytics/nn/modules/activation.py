# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Activation modules."""

import torch
import torch.nn as nn


class AGLU(nn.Module):
    """
    Unified activation function module from https://github.com/kostas1515/AGLU.

    This class implements a parameterized activation function with learnable parameters lambda and kappa.

    Attributes:
        act (nn.Softplus): Softplus activation function with negative beta.
        lambd (nn.Parameter): Learnable lambda parameter initialized with uniform distribution.
        kappa (nn.Parameter): Learnable kappa parameter initialized with uniform distribution.
    """

    def __init__(self, device=None, dtype=None) -> None:
        """Initialize the Unified activation function with learnable parameters."""
        super().__init__()
        self.act = nn.Softplus(beta=-1.0)
        self.lambd = nn.Parameter(nn.init.uniform_(torch.empty(1, device=device, dtype=dtype)))  # lambda parameter
        self.kappa = nn.Parameter(nn.init.uniform_(torch.empty(1, device=device, dtype=dtype)))  # kappa parameter

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the forward pass of the Unified activation function."""
        lam = torch.clamp(self.lambd, min=0.0001)  # Clamp lambda to avoid division by zero
        inv_lam = lam.reciprocal()  # More efficient than (1 / lam)
        log_lam = lam.log()  # More efficient than torch.log(lam) per call

        # Fused arithmetic operations leveraging PyTorch's broadcasting and in-place computations
        kappa_x = self.kappa * x
        splus = self.act(kappa_x.sub_(log_lam))  # in-place subtraction (log_lam is scalar)
        exp_input = inv_lam * splus
        return exp_input.exp()
