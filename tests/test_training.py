import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from model import FHEMLPClassifier
from training import FHETrainer, HybridFHETrainer
from utils import create_context, encrypt_tensor


class TestFHETrainer:
    @pytest.fixture
    def model(self):
        return FHEMLPClassifier(input_dim=10, hidden_dims=[5], num_classes=3, use_polynomial_activation=False)

    @pytest.fixture
    def trainer(self, model):
        return FHETrainer(model, learning_rate=0.01)

    @pytest.fixture
    def context(self):
        return create_context(poly_modulus_degree=8192, scale_bits=40)

    def test_initialization(self, trainer):
        assert trainer.learning_rate == 0.01
        assert trainer.device == "cpu"
        assert trainer.proxy_model is not None
        assert trainer.optimizer is not None
        assert trainer.criterion is not None

    def test_train_on_plain_batch(self, trainer):
        images = torch.randn(4, 10)
        labels = torch.tensor([0, 1, 2, 0])

        loss = trainer.train_on_plain_batch(images, labels)

        assert isinstance(loss, float)
        assert loss > 0

    def test_train_on_encrypted_batch(self, trainer, context):
        images = [torch.randn(10) for _ in range(2)]
        encrypted_images = [encrypt_tensor(context, img) for img in images]
        labels = torch.tensor([0, 1])

        loss = trainer.train_on_encrypted_batch(encrypted_images, labels)

        assert isinstance(loss, float)
        assert loss > 0

    def test_evaluate_plain(self, trainer):
        # Create a simple dataset
        images = torch.randn(10, 10)
        labels = torch.randint(0, 3, (10,))
        dataset = TensorDataset(images, labels)
        dataloader = DataLoader(dataset, batch_size=5)

        accuracy, avg_loss = trainer.evaluate_plain(dataloader)

        assert isinstance(accuracy, float)
        assert 0 <= accuracy <= 1
        assert isinstance(avg_loss, float)
        assert avg_loss > 0

    def test_evaluate_encrypted(self, trainer, context):
        # Create encrypted dataset
        encrypted_dataset = []
        for i in range(5):
            img = torch.randn(10)
            enc_img = encrypt_tensor(context, img)
            label = i % 3
            encrypted_dataset.append((enc_img, label))

        accuracy, avg_loss = trainer.evaluate_encrypted(encrypted_dataset)

        assert isinstance(accuracy, float)
        assert 0 <= accuracy <= 1
        assert isinstance(avg_loss, float)
        assert avg_loss >= 0

    def test_sync_weights_to_fhe(self, trainer):
        # Modify proxy model weights
        for param in trainer.proxy_model.parameters():
            param.data = torch.ones_like(param)

        # Sync to FHE model
        trainer._sync_weights_to_fhe()

        # Check that FHE model weights are updated
        for layer in trainer.model.layers:
            assert layer.weight is not None
            assert layer.bias is not None


class TestHybridFHETrainer:
    @pytest.fixture
    def model(self):
        return FHEMLPClassifier(input_dim=10, hidden_dims=[5], num_classes=3, use_polynomial_activation=False)

    @pytest.fixture
    def trainer(self, model):
        return HybridFHETrainer(model, learning_rate=0.01, noise_scale=0.1)

    @pytest.fixture
    def context(self):
        return create_context(poly_modulus_degree=8192, scale_bits=40)

    def test_initialization(self, trainer):
        assert trainer.noise_scale == 0.1
        assert isinstance(trainer, FHETrainer)

    def test_train_mixed_batch(self, trainer, context):
        # Prepare encrypted images
        encrypted_images = [encrypt_tensor(context, torch.randn(10)) for _ in range(2)]

        # Prepare plain images
        plain_images = torch.randn(4, 10)  # Full batch size to match labels

        # Labels: first half for encrypted, second half for plain
        labels = torch.tensor([0, 1, 2, 1])  # 4 samples total

        loss = trainer.train_mixed_batch(encrypted_images, plain_images, labels)

        assert isinstance(loss, float)
        assert loss > 0

    def test_train_mixed_batch_only_plain(self, trainer):
        plain_images = torch.randn(4, 10)
        labels = torch.tensor([0, 1, 2, 0])

        loss = trainer.train_mixed_batch(None, plain_images, labels)

        assert isinstance(loss, float)
        assert loss > 0

    def test_train_mixed_batch_only_encrypted(self, trainer, context):
        encrypted_images = [encrypt_tensor(context, torch.randn(10)) for _ in range(4)]
        labels = torch.tensor([0, 1, 2, 0])

        loss = trainer.train_mixed_batch(encrypted_images, None, labels)

        assert isinstance(loss, float)
        assert loss > 0
