"""Data loading and preprocessing utilities."""

import logging

import tenseal as ts
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from ..core.encryption import FHEEncryption

logger = logging.getLogger(__name__)


class MNISTDataLoader:
    """MNIST data loader with configurable preprocessing."""

    def __init__(self, batch_size: int = 64, normalize: bool = True) -> None:
        """
        Initialize MNIST data loader.

        Args:
            batch_size: Batch size for data loading
            normalize: Whether to normalize data
        """
        self.batch_size = batch_size
        self.normalize = normalize

    def get_dataloader(self, train: bool = True) -> DataLoader:
        """
        Get MNIST dataloader.

        Args:
            train: Whether to load training or test set

        Returns:
            PyTorch DataLoader
        """
        transform_list = [transforms.ToTensor()]

        if self.normalize:
            # Standard MNIST normalization
            transform_list.append(transforms.Normalize((0.1307,), (0.3081,)))

        transform = transforms.Compose(transform_list)

        dataset = datasets.MNIST(
            root="./data", train=train, download=True, transform=transform
        )

        return DataLoader(dataset, batch_size=self.batch_size, shuffle=train)


class FHEDataset:
    """FHE dataset that encrypts data on initialization."""

    def __init__(
        self,
        encryption: FHEEncryption,
        dataloader: DataLoader,
        max_samples: int | None = None,
    ) -> None:
        """
        Initialize FHE dataset.

        Args:
            encryption: FHE encryption handler
            dataloader: Source PyTorch dataloader
            max_samples: Maximum number of samples to encrypt
        """
        self.encryption = encryption
        self.encrypted_data: list[ts.CKKSTensor] = []
        self.labels: list[int] = []

        logger.info(f"Encrypting dataset with max_samples={max_samples}")

        sample_count = 0
        for images, labels in dataloader:
            if max_samples and sample_count >= max_samples:
                break

            # Flatten images and encrypt each one
            images_flat = images.view(images.shape[0], -1)
            for i, img in enumerate(images_flat):
                if max_samples and sample_count >= max_samples:
                    break

                encrypted_img = self.encryption.encrypt_tensor(img)
                self.encrypted_data.append(encrypted_img)
                self.labels.append(labels[i].item())
                sample_count += 1

        logger.info(f"Encrypted {len(self.encrypted_data)} samples")

    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.encrypted_data)

    def __getitem__(self, idx: int) -> tuple[ts.CKKSTensor, int]:
        """Get encrypted sample and label."""
        return self.encrypted_data[idx], self.labels[idx]

    def get_batch(self, indices: list[int]) -> tuple[list[ts.CKKSTensor], list[int]]:
        """
        Get a batch of encrypted samples.

        Args:
            indices: List of sample indices

        Returns:
            Tuple of (encrypted samples, labels)
        """
        batch_data = [self.encrypted_data[i] for i in indices]
        batch_labels = [self.labels[i] for i in indices]
        return batch_data, batch_labels


def one_hot_encode(labels: torch.Tensor, num_classes: int = 10) -> torch.Tensor:
    """
    Convert labels to one-hot encoding.

    Args:
        labels: Integer labels tensor
        num_classes: Number of classes

    Returns:
        One-hot encoded tensor
    """
    batch_size = labels.shape[0]
    one_hot = torch.zeros(batch_size, num_classes)
    one_hot[torch.arange(batch_size), labels] = 1
    return one_hot
