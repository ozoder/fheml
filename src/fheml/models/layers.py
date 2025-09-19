"""FHE-compatible neural network layers."""

import logging

import tenseal as ts
import torch

from ..core.operations import FHEOperations

logger = logging.getLogger(__name__)


class FHELinearLayer:
    """FHE-compatible linear layer with Xavier initialization."""

    def __init__(self, in_features: int, out_features: int) -> None:
        """
        Initialize FHE linear layer.

        Args:
            in_features: Number of input features
            out_features: Number of output features
        """
        self.in_features = in_features
        self.out_features = out_features

        # Xavier/Glorot initialization for better convergence
        std = (2.0 / (in_features + out_features)) ** 0.5
        self.weight = torch.randn(out_features, in_features) * std
        self.bias = torch.zeros(out_features)

    def forward_encrypted(self, x: ts.CKKSTensor, operations: FHEOperations) -> ts.CKKSTensor:
        """
        Forward pass with encrypted input.

        Args:
            x: Encrypted input tensor
            operations: FHE operations handler

        Returns:
            Encrypted output tensor
        """
        # Check input scale health
        operations.check_scale_health(x, f"linear_layer_input_{self.out_features}")

        bias_plain = self.bias.numpy().tolist()

        # For vector x matrix multiplication, we need to transpose
        # x is shape (input_dim,) and weight is (out_features, in_features)
        # We want: x @ weight.T = (input_dim,) @ (in_features, out_features)
        weight_t_plain = self.weight.t().numpy().tolist()

        result = x.dot(weight_t_plain)

        # Apply safe rescaling
        result = operations.safe_rescale(result)
        operations.check_scale_health(result, f"linear_layer_after_dot_{self.out_features}")

        result = result + bias_plain

        # Final scale check
        operations.check_scale_health(result, f"linear_layer_output_{self.out_features}")

        return result

    def forward_plain(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with plain input.

        Args:
            x: Plain input tensor

        Returns:
            Plain output tensor
        """
        return torch.mm(x, self.weight.t()) + self.bias

    def update_weights(self, weight: torch.Tensor, bias: torch.Tensor) -> None:
        """
        Update layer weights and bias.

        Args:
            weight: New weight tensor
            bias: New bias tensor
        """
        self.weight = weight
        self.bias = bias

    def get_parameters(self) -> list[torch.Tensor]:
        """Get layer parameters."""
        return [self.weight, self.bias]

    def set_parameters(self, weight: torch.Tensor, bias: torch.Tensor) -> None:
        """Set layer parameters."""
        self.weight = weight
        self.bias = bias
