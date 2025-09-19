"""FHE-compatible MLP classifier."""

import logging

import tenseal as ts
import torch

from ..core.operations import FHEOperations
from .activations import FHELinearActivation, FHEPolynomialActivation
from .layers import FHELinearLayer

logger = logging.getLogger(__name__)


class FHEMLPClassifier:
    """FHE-compatible Multi-Layer Perceptron classifier."""

    def __init__(
        self,
        input_dim: int = 784,
        hidden_dims: list[int] = None,
        num_classes: int = 10,
        use_polynomial_activation: bool = True,
    ) -> None:
        """
        Initialize FHE MLP classifier.

        Args:
            input_dim: Input dimension
            hidden_dims: List of hidden layer dimensions
            num_classes: Number of output classes
            use_polynomial_activation: Whether to use polynomial or linear activation
        """
        if hidden_dims is None:
            hidden_dims = [256, 128, 64]  # Deep architecture for high accuracy

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.num_classes = num_classes
        self.use_polynomial_activation = use_polynomial_activation

        self.layers: list[FHELinearLayer] = []
        self.activations: list[FHEPolynomialActivation | FHELinearActivation] = []

        # Build network layers
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            self.layers.append(FHELinearLayer(prev_dim, hidden_dim))
            if use_polynomial_activation:
                self.activations.append(FHEPolynomialActivation())
            else:
                self.activations.append(FHELinearActivation())
            prev_dim = hidden_dim

        # Output layer - always linear
        self.layers.append(FHELinearLayer(prev_dim, num_classes))

    def forward_encrypted(self, x: ts.CKKSTensor, operations: FHEOperations) -> ts.CKKSTensor:
        """
        Forward pass with encrypted input.

        Args:
            x: Encrypted input tensor
            operations: FHE operations handler

        Returns:
            Encrypted output predictions
        """
        # Process through hidden layers with activations
        for i, layer in enumerate(self.layers[:-1]):
            x = layer.forward_encrypted(x, operations)
            x = self.activations[i].forward(x, operations)

        # Output layer (no activation)
        x = self.layers[-1].forward_encrypted(x, operations)

        return x

    def forward_plain(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with plain input.

        Args:
            x: Plain input tensor

        Returns:
            Plain output predictions
        """
        # Process through hidden layers with activations
        for i, layer in enumerate(self.layers[:-1]):
            x = layer.forward_plain(x)
            x = self.activations[i].forward(x)

        # Output layer (no activation)
        x = self.layers[-1].forward_plain(x)

        return x

    def get_parameters(self) -> list[torch.Tensor]:
        """
        Get all model parameters.

        Returns:
            List of parameter tensors (weights and biases)
        """
        params = []
        for layer in self.layers:
            params.extend(layer.get_parameters())
        return params

    def set_parameters(self, params: list[torch.Tensor]) -> None:
        """
        Set all model parameters.

        Args:
            params: List of parameter tensors

        Raises:
            ValueError: If number of parameters doesn't match model structure
        """
        expected_params = len(self.layers) * 2  # weight + bias per layer
        if len(params) != expected_params:
            raise ValueError(
                f"Expected {expected_params} parameters, got {len(params)}"
            )

        idx = 0
        for layer in self.layers:
            layer.set_parameters(params[idx], params[idx + 1])
            idx += 2

    def get_architecture_info(self) -> dict:
        """
        Get information about the model architecture.

        Returns:
            Dictionary with architecture details
        """
        return {
            "input_dim": self.input_dim,
            "hidden_dims": self.hidden_dims,
            "num_classes": self.num_classes,
            "use_polynomial_activation": self.use_polynomial_activation,
            "total_layers": len(self.layers),
            "total_parameters": sum(p.numel() for p in self.get_parameters()),
        }
