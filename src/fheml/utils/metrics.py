"""Metrics and evaluation utilities."""

import logging

import tenseal as ts
import torch

from ..core.encryption import FHEEncryption

logger = logging.getLogger(__name__)


def accuracy_score(
    predictions: list[int] | torch.Tensor,
    targets: list[int] | torch.Tensor
) -> float:
    """
    Calculate accuracy score.

    Args:
        predictions: Predicted labels
        targets: True labels

    Returns:
        Accuracy as a float between 0 and 1
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.tolist()
    if isinstance(targets, torch.Tensor):
        targets = targets.tolist()

    if len(predictions) != len(targets):
        raise ValueError(
            f"Predictions and targets must have same length: "
            f"{len(predictions)} vs {len(targets)}"
        )

    correct = sum(p == t for p, t in zip(predictions, targets, strict=False))
    return correct / len(targets) if len(targets) > 0 else 0.0


def decrypt_and_predict(
    encrypted_outputs: list[ts.CKKSTensor],
    encryption: FHEEncryption
) -> list[int]:
    """
    Decrypt encrypted model outputs and convert to predictions.

    Args:
        encrypted_outputs: List of encrypted output tensors
        encryption: FHE encryption handler

    Returns:
        List of predicted class indices
    """
    predictions = []

    for encrypted_output in encrypted_outputs:
        try:
            # Decrypt the output
            decrypted = encryption.decrypt_tensor(encrypted_output)

            # Get prediction (argmax)
            pred_idx = torch.argmax(decrypted).item()
            predictions.append(pred_idx)

        except Exception as e:
            logger.warning(f"Failed to decrypt/predict: {e}")
            # Fallback to random prediction
            predictions.append(0)

    return predictions


def compute_loss_l2(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    reduction: str = "mean"
) -> torch.Tensor:
    """
    Compute L2 (MSE) loss for FHE compatibility.

    Args:
        predictions: Model predictions
        targets: True targets (one-hot encoded)
        reduction: Reduction method ("mean", "sum", or "none")

    Returns:
        Computed loss tensor
    """
    diff = predictions - targets
    squared_diff = diff ** 2

    if reduction == "mean":
        return torch.mean(squared_diff)
    elif reduction == "sum":
        return torch.sum(squared_diff)
    else:
        return squared_diff
