#!/usr/bin/env python3
"""Simple integration test to verify refactored architecture works."""

import torch

from src.fheml.core.context import FHEContextManager
from src.fheml.core.encryption import FHEEncryption
from src.fheml.models.classifier import FHEMLPClassifier
from src.fheml.training.trainer import FHETrainer, TrainingConfig
from src.fheml.utils.data import FHEDataset, MNISTDataLoader


def test_architecture_integration():
    """Test that all components work together."""
    print("🧪 Testing Clean Architecture Integration...")

    # 1. Test Context Manager
    print("1. Testing FHE Context Manager...")
    context_manager = FHEContextManager()
    context = context_manager.context
    assert context is not None
    print("   ✅ FHE context created successfully")

    # 2. Test Encryption
    print("2. Testing Encryption/Decryption...")
    encryption = FHEEncryption(context)
    test_tensor = torch.randn(10)
    encrypted = encryption.encrypt_tensor(test_tensor)
    decrypted = encryption.decrypt_tensor(encrypted)
    assert torch.allclose(test_tensor, decrypted, atol=1e-2)
    print("   ✅ Encryption/decryption working correctly")

    # 3. Test Model
    print("3. Testing Model Architecture...")
    model = FHEMLPClassifier(
        input_dim=10,  # Small for testing
        hidden_dims=[8, 4],
        num_classes=3,
        use_polynomial_activation=False,
    )
    test_input = torch.randn(1, 10)
    output = model.forward_plain(test_input)
    assert output.shape == (1, 3)
    print("   ✅ Model forward pass working")

    # 4. Test Configuration
    print("4. Testing Training Configuration...")
    config = TrainingConfig(
        epochs=1,
        learning_rate=0.01,
        batch_size=2,
        max_train_samples=4,
        max_test_samples=2,
    )
    assert config.epochs == 1
    print("   ✅ Configuration working")

    # 5. Test Trainer Initialization
    print("5. Testing Trainer Initialization...")
    trainer = FHETrainer(model=model, context_manager=context_manager, config=config)
    assert trainer.model is not None
    assert trainer.context is not None
    assert trainer.encryption is not None
    print("   ✅ Trainer initialization working")

    # 6. Test Data Loading
    print("6. Testing Data Loading...")
    data_loader = MNISTDataLoader(batch_size=8)
    train_loader = data_loader.get_dataloader(train=True)
    assert train_loader is not None

    # Get one batch to verify
    for images, labels in train_loader:
        assert images.shape[0] <= 8  # batch size
        assert images.shape[1:] == (1, 28, 28)  # MNIST shape
        break
    print("   ✅ Data loading working")

    print("\n🎉 All architecture components working correctly!")
    print("✅ Clean architecture refactoring verified!")
    return True


if __name__ == "__main__":
    try:
        test_architecture_integration()
    except Exception as e:
        print(f"❌ Test failed: {e}")
        raise
