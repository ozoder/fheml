from typing import Optional, Tuple

import tenseal as ts
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def create_context(
    poly_modulus_degree: int = 8192,
    coeff_mod_bit_sizes: Optional[list] = None,
    scale_bits: int = 40,
) -> ts.Context:
    if coeff_mod_bit_sizes is None:
        # Use appropriate coefficient modulus for the given polynomial degree
        if poly_modulus_degree == 4096:
            coeff_mod_bit_sizes = [40, 40, 40]
        elif poly_modulus_degree == 8192:
            coeff_mod_bit_sizes = [60, 40, 40, 60]
        else:
            coeff_mod_bit_sizes = [60, 40, 40, 60]

    # Ensure valid polynomial modulus degree
    if poly_modulus_degree < 8192:
        poly_modulus_degree = 8192

    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=poly_modulus_degree,
        coeff_mod_bit_sizes=coeff_mod_bit_sizes,
    )

    context.global_scale = 2**scale_bits
    context.generate_galois_keys()

    return context


def encrypt_tensor(context: ts.Context, tensor: torch.Tensor) -> ts.CKKSTensor:
    if len(tensor.shape) == 1:
        encrypted = ts.ckks_tensor(context, tensor.numpy().tolist())
    else:
        tensor_flat = tensor.flatten()
        encrypted = ts.ckks_tensor(context, tensor_flat.numpy().tolist())

    return encrypted


def decrypt_tensor(encrypted_tensor: ts.CKKSTensor) -> torch.Tensor:
    decrypted = encrypted_tensor.decrypt()
    if hasattr(decrypted, 'tolist'):
        decrypted = decrypted.tolist()
    return torch.tensor(decrypted, dtype=torch.float32)


def load_mnist_data(
    batch_size: int = 64, train: bool = True, normalize: bool = True
) -> DataLoader:
    transform_list = [transforms.ToTensor()]

    if normalize:
        transform_list.append(transforms.Normalize((0.1307,), (0.3081,)))

    transform = transforms.Compose(transform_list)

    dataset = datasets.MNIST(
        root="./data", train=train, download=True, transform=transform
    )

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=train)

    return dataloader


def prepare_encrypted_batch(
    context: ts.Context, images: torch.Tensor, labels: torch.Tensor
) -> Tuple[list, torch.Tensor]:
    batch_size = images.shape[0]
    images_flat = images.view(batch_size, -1)

    encrypted_images = []
    for i in range(batch_size):
        encrypted_img = encrypt_tensor(context, images_flat[i])
        encrypted_images.append(encrypted_img)

    return encrypted_images, labels


def one_hot_encode(labels: torch.Tensor, num_classes: int = 10) -> torch.Tensor:
    batch_size = labels.shape[0]
    one_hot = torch.zeros(batch_size, num_classes)
    one_hot[torch.arange(batch_size), labels] = 1
    return one_hot


class FHEDataset:
    def __init__(
        self,
        context: ts.Context,
        dataloader: DataLoader,
        max_samples: Optional[int] = None,
    ):
        self.context = context
        self.encrypted_data = []
        self.labels = []

        sample_count = 0
        for images, labels in dataloader:
            if max_samples and sample_count >= max_samples:
                break

            encrypted_batch, _ = prepare_encrypted_batch(
                context, images, labels
            )
            self.encrypted_data.extend(encrypted_batch)
            self.labels.append(labels)

            sample_count += len(images)

            if max_samples and sample_count >= max_samples:
                self.encrypted_data = self.encrypted_data[:max_samples]
                break

        self.labels = torch.cat(self.labels)
        if max_samples:
            self.labels = self.labels[:max_samples]

    def __len__(self):
        return len(self.encrypted_data)

    def __getitem__(self, idx):
        return self.encrypted_data[idx], self.labels[idx]
