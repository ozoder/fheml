"""Checkpoint management for FHE training."""

import gc
import logging
import os
from datetime import datetime
from typing import Any

import torch

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Manages model checkpoints during training."""

    def __init__(
        self,
        checkpoint_dir: str = ".checkpoints",
        save_best_only: bool = False,
    ) -> None:
        """
        Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory to save checkpoints
            save_best_only: Whether to save only the best checkpoint
        """
        self.checkpoint_dir = checkpoint_dir
        self.save_best_only = save_best_only
        self.best_accuracy = 0.0
        self.checkpoints: list[dict[str, Any]] = []

        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)

    def save_checkpoint(
        self,
        model_state: dict[str, Any],
        epoch: int,
        accuracy: float,
        loss: float,
        metadata: dict[str, Any] | None = None,
        force_save_to_disk: bool = False,
    ) -> str | None:
        """
        Save a training checkpoint.

        Args:
            model_state: Model state dictionary
            epoch: Current epoch
            accuracy: Current accuracy
            loss: Current loss
            metadata: Optional metadata
            force_save_to_disk: Whether to force saving to disk

        Returns:
            Checkpoint file path if saved to disk, None otherwise
        """
        checkpoint = {
            "epoch": epoch,
            "model_state": model_state,
            "accuracy": accuracy,
            "loss": loss,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {},
        }

        # Determine if we should save to disk
        save_to_disk = force_save_to_disk
        if self.save_best_only:
            save_to_disk = save_to_disk or accuracy > self.best_accuracy
        else:
            save_to_disk = True

        if save_to_disk:
            # Save to disk
            checkpoint_path = os.path.join(
                self.checkpoint_dir, f"checkpoint_epoch_{epoch}.pt"
            )

            # Create a copy without the model_state for logging
            log_checkpoint = {k: v for k, v in checkpoint.items() if k != "model_state"}
            logger.info(f"Saving checkpoint to {checkpoint_path}: {log_checkpoint}")

            torch.save(checkpoint, checkpoint_path)

            # Update best accuracy
            if accuracy > self.best_accuracy:
                self.best_accuracy = accuracy

            # Clear model state from memory to save space
            del checkpoint["model_state"]
            checkpoint["saved_to_disk"] = checkpoint_path

            # Clean up memory
            gc.collect()

            return checkpoint_path
        else:
            # Keep in memory only
            self.checkpoints.append(checkpoint)
            logger.info(f"Checkpoint kept in memory for epoch {epoch}")
            return None

    def load_checkpoint(self, checkpoint_path: str) -> dict[str, Any]:
        """
        Load a checkpoint from disk.

        Args:
            checkpoint_path: Path to checkpoint file

        Returns:
            Loaded checkpoint data

        Raises:
            FileNotFoundError: If checkpoint file doesn't exist
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        return checkpoint

    def get_best_checkpoint(self) -> dict[str, Any] | None:
        """
        Get the best checkpoint (highest accuracy).

        Returns:
            Best checkpoint data or None if no checkpoints exist
        """
        if not self.checkpoints:
            return None

        best_checkpoint = max(self.checkpoints, key=lambda x: x["accuracy"])
        return best_checkpoint

    def cleanup_old_checkpoints(self, keep_n: int = 3) -> None:
        """
        Clean up old checkpoint files, keeping only the N most recent.

        Args:
            keep_n: Number of checkpoints to keep
        """
        checkpoint_files = []
        for filename in os.listdir(self.checkpoint_dir):
            if filename.startswith("checkpoint_epoch_") and filename.endswith(".pt"):
                filepath = os.path.join(self.checkpoint_dir, filename)
                # Extract epoch number for sorting
                try:
                    epoch_str = filename.replace("checkpoint_epoch_", "").replace(".pt", "")
                    epoch = int(epoch_str)
                    checkpoint_files.append((epoch, filepath))
                except ValueError:
                    continue

        # Sort by epoch and keep only the most recent
        checkpoint_files.sort(key=lambda x: x[0])
        files_to_remove = checkpoint_files[:-keep_n]

        for epoch, filepath in files_to_remove:
            try:
                os.remove(filepath)
                logger.info(f"Removed old checkpoint: {filepath}")
            except OSError as e:
                logger.warning(f"Failed to remove checkpoint {filepath}: {e}")

    def get_checkpoint_summary(self) -> dict[str, Any]:
        """
        Get a summary of all checkpoints.

        Returns:
            Summary dictionary with checkpoint statistics
        """
        total_checkpoints = len(self.checkpoints)
        disk_checkpoints = len([f for f in os.listdir(self.checkpoint_dir) if f.endswith(".pt")])

        summary = {
            "total_memory_checkpoints": total_checkpoints,
            "total_disk_checkpoints": disk_checkpoints,
            "best_accuracy": self.best_accuracy,
            "checkpoint_dir": self.checkpoint_dir,
        }

        if self.checkpoints:
            accuracies = [cp["accuracy"] for cp in self.checkpoints]
            summary.update({
                "min_accuracy": min(accuracies),
                "max_accuracy": max(accuracies),
                "avg_accuracy": sum(accuracies) / len(accuracies),
            })

        return summary
