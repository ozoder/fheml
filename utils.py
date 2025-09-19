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
        # Use conservative, TenSEAL-compatible parameters with more levels for FHE operations
        if poly_modulus_degree == 4096:
            coeff_mod_bit_sizes = [50, 40, 40, 50]  # More levels for 4096
        elif poly_modulus_degree == 8192:
            coeff_mod_bit_sizes = [
                60,
                40,
                40,
                40,
                40,
                60,
            ]  # More levels for scale management
        elif poly_modulus_degree == 16384:
            coeff_mod_bit_sizes = [60, 40, 40, 40, 40, 40, 40, 60]
        else:
            coeff_mod_bit_sizes = [60, 40, 40, 40, 60]  # Default with more levels

    # Ensure valid polynomial modulus degree
    if poly_modulus_degree < 8192:
        poly_modulus_degree = 8192

    print(f"Creating FHE context: poly_degree={poly_modulus_degree}, coeff_bits={coeff_mod_bit_sizes}, scale_bits={scale_bits}")

    try:
        context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=poly_modulus_degree,
            coeff_mod_bit_sizes=coeff_mod_bit_sizes,
        )

        context.global_scale = 2**scale_bits
        context.generate_galois_keys()

        print(f"Successfully created FHE context with global scale: {context.global_scale}")
        return context

    except Exception as e:
        print(f"Error creating FHE context: {e}")
        # Fallback to simpler parameters
        print("Trying fallback parameters...")
        fallback_coeff_mod_bit_sizes = [60, 40, 40, 60]
        context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=8192,
            coeff_mod_bit_sizes=fallback_coeff_mod_bit_sizes,
        )
        context.global_scale = 2**scale_bits
        context.generate_galois_keys()
        print(f"Fallback context created with global scale: {context.global_scale}")
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
    if hasattr(decrypted, "tolist"):
        decrypted = decrypted.tolist()
    return torch.tensor(decrypted, dtype=torch.float32)


def safe_rescale(
    tensor: ts.CKKSTensor, target_scale: float = None
) -> ts.CKKSTensor:
    """Safely rescale a tensor to manage scale growth."""
    try:
        if hasattr(tensor, "rescale_to_next"):
            # More aggressive rescaling - rescale if scale is larger than global scale
            scale_ratio = tensor.scale() / tensor.context().global_scale
            if scale_ratio > 2.0:  # Rescale if scale is much larger than global
                print(f"Rescaling tensor: scale ratio {scale_ratio:.3f}")
                tensor.rescale_to_next()
        return tensor
    except Exception as e:
        print(f"Warning: Rescaling failed: {e}")
        return tensor


def check_scale_health(
    tensor: ts.CKKSTensor, operation_name: str = ""
) -> bool:
    """Check if tensor scale is within safe bounds."""
    try:
        scale_ratio = tensor.scale() / tensor.context().global_scale
        if scale_ratio > 10:  # Scale too large - reduced from 100 for earlier detection
            print(
                f"Warning: {operation_name} - Scale too large: {scale_ratio:.2f}x global scale"
            )
            return False
        elif scale_ratio < 0.1:  # Scale too small - increased from 0.01 for earlier detection
            print(
                f"Warning: {operation_name} - Scale too small: {scale_ratio:.4f}x global scale"
            )
            return False
        elif scale_ratio > 5:  # Warn earlier but don't fail
            print(
                f"Info: {operation_name} - Scale getting large: {scale_ratio:.2f}x global scale"
            )
        return True
    except Exception as e:
        print(f"Error checking scale health: {e}")
        return False


def bootstrap_if_needed(
    tensor: ts.CKKSTensor, min_scale_ratio: float = 0.1
) -> ts.CKKSTensor:
    """Bootstrap tensor if scale becomes too small."""
    try:
        # Check if tensor has scale and context
        if hasattr(tensor, "scale") and hasattr(tensor, "context"):
            try:
                context = tensor.context()
                scale_ratio = tensor.scale() / context.global_scale
                if scale_ratio < min_scale_ratio:
                    print(
                        f"Bootstrapping tensor (scale ratio: {scale_ratio:.4f})"
                    )
                    # For now, we'll decrypt and re-encrypt as a simple bootstrap
                    # In production, you'd use proper bootstrapping if available
                    decrypted = decrypt_tensor(tensor)
                    return encrypt_tensor(context, decrypted)
            except Exception as inner_e:
                print(f"Bootstrap scale check failed: {inner_e}")
    except Exception as e:
        print(f"Bootstrap failed: {e}")
    return tensor


def safe_multiply(a: ts.CKKSTensor, b: ts.CKKSTensor) -> ts.CKKSTensor:
    """Safely multiply two encrypted tensors with scale management."""
    try:
        result = a * b
        result = safe_rescale(result)
        return result
    except ValueError as e:
        if "scale out of bounds" in str(e):
            print("Scale out of bounds detected, attempting recovery...")
            # Try bootstrapping the operands
            a_boot = bootstrap_if_needed(a)
            b_boot = bootstrap_if_needed(b)
            result = a_boot * b_boot
            result = safe_rescale(result)
            return result
        else:
            raise e


def safe_square(tensor: ts.CKKSTensor) -> ts.CKKSTensor:
    """Safely square a tensor with scale management."""
    try:
        # Check scale health before squaring
        if not check_scale_health(tensor, "pre-square"):
            tensor = bootstrap_if_needed(tensor)

        result = tensor.square()
        result = safe_rescale(result)

        # Check result scale health
        check_scale_health(result, "post-square")

        return result
    except ValueError as e:
        if "scale out of bounds" in str(e):
            print(
                "Scale out of bounds in square operation, attempting recovery..."
            )
            tensor_boot = bootstrap_if_needed(tensor)
            result = tensor_boot.square()
            result = safe_rescale(result)
            return result
        else:
            raise e


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


def one_hot_encode(
    labels: torch.Tensor, num_classes: int = 10
) -> torch.Tensor:
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
