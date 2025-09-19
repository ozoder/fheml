"""FHE encryption and decryption operations."""

import logging

import tenseal as ts
import torch

logger = logging.getLogger(__name__)


class FHEEncryption:
    """Handles FHE encryption and decryption operations."""

    def __init__(self, context: ts.Context) -> None:
        """
        Initialize FHE encryption handler.

        Args:
            context: TenSEAL context for encryption operations
        """
        self.context = context

    def encrypt_tensor(self, tensor: torch.Tensor) -> ts.CKKSTensor:
        """
        Encrypt a PyTorch tensor.

        Args:
            tensor: Input tensor to encrypt

        Returns:
            Encrypted CKKS tensor

        Raises:
            ValueError: If tensor encryption fails
        """
        try:
            if len(tensor.shape) == 1:
                encrypted = ts.ckks_tensor(self.context, tensor.numpy().tolist())
            else:
                tensor_flat = tensor.flatten()
                encrypted = ts.ckks_tensor(self.context, tensor_flat.numpy().tolist())

            return encrypted

        except Exception as e:
            logger.error(f"Failed to encrypt tensor: {e}")
            raise ValueError(f"Tensor encryption failed: {e}") from e

    def decrypt_tensor(self, encrypted_tensor: ts.CKKSTensor) -> torch.Tensor:
        """
        Decrypt a CKKS tensor to PyTorch tensor.

        Args:
            encrypted_tensor: Encrypted CKKS tensor

        Returns:
            Decrypted PyTorch tensor

        Raises:
            ValueError: If tensor decryption fails
        """
        try:
            decrypted = encrypted_tensor.decrypt()
            if hasattr(decrypted, "tolist"):
                decrypted = decrypted.tolist()
            return torch.tensor(decrypted, dtype=torch.float32)

        except Exception as e:
            logger.error(f"Failed to decrypt tensor: {e}")
            raise ValueError(f"Tensor decryption failed: {e}") from e

    def encrypt_batch(
        self, images: torch.Tensor, labels: torch.Tensor
    ) -> tuple[list[ts.CKKSTensor], torch.Tensor]:
        """
        Encrypt a batch of images for training.

        Args:
            images: Batch of input images
            labels: Corresponding labels

        Returns:
            Tuple of (encrypted images list, labels tensor)
        """
        batch_size = images.shape[0]
        images_flat = images.view(batch_size, -1)

        encrypted_images = []
        for i in range(batch_size):
            encrypted_img = self.encrypt_tensor(images_flat[i])
            encrypted_images.append(encrypted_img)

        return encrypted_images, labels
