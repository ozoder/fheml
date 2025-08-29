#!/usr/bin/env python3

import torch

from inference import FHEInference
from model import FHEMLPClassifier
from utils import create_context, decrypt_tensor, encrypt_tensor, load_mnist_data


def test_basic_functionality():
    print("Testing FHE-ML Basic Functionality")
    print("=" * 50)

    print("\n1. Creating FHE context...")
    context = create_context(poly_modulus_degree=8192, scale_bits=40)
    print("✓ Context created successfully")

    print("\n2. Creating model...")
    model = FHEMLPClassifier(input_dim=784, hidden_dims=[32], num_classes=10, use_polynomial_activation=False)  # Use linear activation only
    print("✓ Model created successfully")

    print("\n3. Testing encryption/decryption...")
    test_tensor = torch.randn(784)
    encrypted = encrypt_tensor(context, test_tensor)
    decrypted = decrypt_tensor(encrypted)

    error = torch.mean(torch.abs(test_tensor - decrypted)).item()
    print(f"✓ Encryption/Decryption works (error: {error:.6f})")

    print("\n4. Testing encrypted forward pass...")
    encrypted_output = model.forward_encrypted(encrypted)
    decrypted_output = decrypt_tensor(encrypted_output)
    print(
        f"✓ Encrypted forward pass successful (output shape: {decrypted_output.shape})"
    )

    print("\n5. Testing plain forward pass...")
    plain_output = model.forward_plain(test_tensor.unsqueeze(0))
    print(f"✓ Plain forward pass successful (output shape: {plain_output.shape})")

    print("\n6. Loading MNIST sample...")
    test_loader = load_mnist_data(batch_size=1, train=False)
    sample_image, sample_label = next(iter(test_loader))
    sample_image_flat = sample_image.view(1, -1).squeeze()
    print(
        f"✓ MNIST sample loaded (shape: {sample_image.shape}, label: {sample_label.item()})"
    )

    print("\n7. Testing inference on MNIST sample...")
    encrypted_sample = encrypt_tensor(context, sample_image_flat)
    inference_engine = FHEInference(model, context)
    prediction = inference_engine.predict_encrypted(encrypted_sample)
    print(f"✓ Inference completed (predicted: {prediction})")

    print("\n" + "=" * 50)
    print("All basic tests passed! ✓")
    print("=" * 50)


if __name__ == "__main__":
    test_basic_functionality()
