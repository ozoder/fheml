"""FHE-compatible activation functions."""

import logging

import tenseal as ts
import torch

from ..core.operations import FHEOperations

logger = logging.getLogger(__name__)


class FHEPolynomialActivation:
    """
    Enhanced FHE-friendly polynomial approximation with proper scale management.

    Uses ReLU approximation: f(x) ≈ max(0, x) ≈ x * σ(x) where σ is sigmoid-like
    Implemented as: f(x) ≈ 0.125 * (x + |x|) ≈ 0.125 * x * (1 + sign(x))
    For FHE: f(x) ≈ 0.5 * x + 0.25 * x^2 (degree-2 polynomial)
    """

    @staticmethod
    def forward(
        x: ts.CKKSTensor | torch.Tensor,
        operations: FHEOperations = None,
    ) -> ts.CKKSTensor | torch.Tensor:
        """
        Apply polynomial activation.

        Args:
            x: Input tensor (encrypted or plain)
            operations: FHE operations handler (required for encrypted tensors)

        Returns:
            Activated tensor

        Raises:
            ValueError: If operations is None for encrypted tensor
        """
        if isinstance(x, torch.Tensor):
            # For plain computation, use actual ReLU
            return torch.relu(x)
        else:
            if operations is None:
                raise ValueError("FHEOperations required for encrypted tensor")

            # Enhanced polynomial activation with proper scale management
            operations.check_scale_health(x, "activation_input")

            # Normalize input to avoid scale explosion
            normalized_x = x * 0.1  # Scale down input
            normalized_x = operations.safe_rescale(normalized_x)

            # Compute x^2 term carefully
            x_squared = operations.safe_square(normalized_x)
            x_squared = operations.safe_rescale(x_squared)

            # Polynomial: 0.5*x + 0.25*x^2 (after normalization)
            linear_term = normalized_x * 5.0  # Compensate for normalization
            quadratic_term = x_squared * 2.5  # Scaled quadratic term

            result = linear_term + quadratic_term
            result = operations.safe_rescale(result)
            operations.check_scale_health(result, "activation_output")

            return result


class FHELinearActivation:
    """Linear activation (identity function) - no multiplicative depth."""

    @staticmethod
    def forward(
        x: ts.CKKSTensor | torch.Tensor,
        operations: FHEOperations = None,
    ) -> ts.CKKSTensor | torch.Tensor:
        """
        Apply linear activation (identity).

        Args:
            x: Input tensor (encrypted or plain)
            operations: FHE operations handler (unused for linear activation)

        Returns:
            Input tensor unchanged
        """
        return x
