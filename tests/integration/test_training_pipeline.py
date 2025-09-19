"""Integration tests for the complete training pipeline."""

import pytest
import torch

from src.fheml.core.context import FHEContextManager
from src.fheml.core.encryption import FHEEncryption
from src.fheml.models.classifier import FHEMLPClassifier
from src.fheml.training.trainer import FHETrainer, TrainingConfig
from src.fheml.utils.data import FHEDataset, MNISTDataLoader


class TestTrainingPipeline:
    """Integration tests for complete training pipeline."""

    @pytest.fixture
    def context_manager(self):
        """Create FHE context manager."""
        return FHEContextManager()

    @pytest.fixture
    def model(self):
        """Create small FHE model for testing."""
        return FHEMLPClassifier(
            input_dim=784,
            hidden_dims=[32, 16],  # Small for testing
            num_classes=10,
            use_polynomial_activation=False,  # Linear for simplicity
        )

    @pytest.fixture
    def config(self):
        """Create training configuration."""
        return TrainingConfig(
            epochs=1,
            learning_rate=0.1,
            batch_size=2,
            max_train_samples=4,
            max_test_samples=2,
        )

    def test_complete_pipeline(self, context_manager, model, config):
        """Test complete training pipeline end-to-end."""
        # Create encryption
        context = context_manager.context
        encryption = FHEEncryption(context)

        # Create minimal datasets
        train_loader = MNISTDataLoader(batch_size=16).get_dataloader(train=True)
        test_loader = MNISTDataLoader(batch_size=16).get_dataloader(train=False)

        train_dataset = FHEDataset(encryption, train_loader, max_samples=4)
        test_dataset = FHEDataset(encryption, test_loader, max_samples=2)

        # Create trainer
        trainer = FHETrainer(
            model=model,
            context_manager=context_manager,
            config=config,
        )

        # Run training
        history = trainer.train(train_dataset, test_dataset)

        # Verify training completed
        assert len(history.train_loss) == 1
        assert len(history.test_loss) == 1
        assert len(history.test_accuracy) == 1
        assert len(history.epoch_times) == 1

        # Verify metrics are reasonable
        assert 0 <= history.test_accuracy[0] <= 1.0
        assert history.train_loss[0] >= 0
        assert history.test_loss[0] >= 0
        assert history.epoch_times[0] > 0

    def test_model_forward_passes(self, context_manager, model):
        """Test model forward passes with encrypted and plain data."""
        context = context_manager.context
        encryption = FHEEncryption(context)

        # Test data
        x_plain = torch.randn(784)

        # Test plain forward pass
        output_plain = model.forward_plain(x_plain.unsqueeze(0))
        assert output_plain.shape == (1, 10)

        # Test encrypted forward pass
        x_encrypted = encryption.encrypt_tensor(x_plain)
        from src.fheml.core.operations import FHEOperations
        operations = FHEOperations(context)

        output_encrypted = model.forward_encrypted(x_encrypted, operations)
        output_decrypted = encryption.decrypt_tensor(output_encrypted)

        assert len(output_decrypted) == 10

    def test_data_encryption_decryption(self, context_manager):
        """Test data encryption and decryption round trip."""
        encryption = FHEEncryption(context_manager.context)

        # Test single tensor
        original = torch.randn(784)
        encrypted = encryption.encrypt_tensor(original)
        decrypted = encryption.decrypt_tensor(encrypted)

        # Should be approximately equal (FHE has numerical errors)
        assert torch.allclose(original, decrypted, atol=1e-2)

    def test_trainer_initialization(self, context_manager, model, config):
        """Test trainer initializes correctly with all components."""
        trainer = FHETrainer(
            model=model,
            context_manager=context_manager,
            config=config,
        )

        assert trainer.model == model
        assert trainer.context_manager == context_manager
        assert trainer.config == config
        assert trainer.context is not None
        assert trainer.encryption is not None
        assert trainer.operations is not None
        assert trainer.memory_manager is not None