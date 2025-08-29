import pytest
import tenseal as ts
import torch
from torch.utils.data import DataLoader

from utils import (
    FHEDataset,
    create_context,
    decrypt_tensor,
    encrypt_tensor,
    load_mnist_data,
    one_hot_encode,
    prepare_encrypted_batch,
)


class TestContextCreation:
    def test_create_context_default(self):
        context = create_context()
        assert context is not None
        assert context.is_private()

    def test_create_context_custom_params(self):
        context = create_context(
            poly_modulus_degree=8192,
            coeff_mod_bit_sizes=[60, 40, 40, 60],
            scale_bits=40,
        )
        assert context is not None
        assert context.is_private()


class TestEncryption:
    @pytest.fixture
    def context(self):
        return create_context(poly_modulus_degree=8192, scale_bits=40)

    def test_encrypt_decrypt_vector(self, context):
        tensor = torch.randn(10)
        encrypted = encrypt_tensor(context, tensor)
        decrypted = decrypt_tensor(encrypted)

        assert decrypted.shape == tensor.shape
        error = torch.mean(torch.abs(tensor - decrypted)).item()
        assert error < 0.001

    def test_encrypt_decrypt_matrix(self, context):
        tensor = torch.randn(5, 10)
        encrypted = encrypt_tensor(context, tensor)
        decrypted = decrypt_tensor(encrypted)

        assert decrypted.shape == (50,)  # Flattened
        tensor_flat = tensor.flatten()
        error = torch.mean(torch.abs(tensor_flat - decrypted)).item()
        assert error < 0.001


class TestDataLoading:
    def test_load_mnist_train(self):
        dataloader = load_mnist_data(batch_size=32, train=True)
        assert isinstance(dataloader, DataLoader)

        batch_images, batch_labels = next(iter(dataloader))
        assert batch_images.shape == (32, 1, 28, 28)
        assert batch_labels.shape == (32,)

    def test_load_mnist_test(self):
        dataloader = load_mnist_data(batch_size=16, train=False)
        assert isinstance(dataloader, DataLoader)

        batch_images, batch_labels = next(iter(dataloader))
        assert batch_images.shape == (16, 1, 28, 28)
        assert batch_labels.shape == (16,)


class TestDataPreparation:
    @pytest.fixture
    def context(self):
        return create_context(poly_modulus_degree=8192, scale_bits=40)

    def test_prepare_encrypted_batch(self, context):
        images = torch.randn(4, 1, 28, 28)
        labels = torch.tensor([0, 1, 2, 3])

        encrypted_images, batch_labels = prepare_encrypted_batch(
            context, images, labels
        )

        assert len(encrypted_images) == 4
        assert torch.equal(batch_labels, labels)

        for enc_img in encrypted_images:
            assert isinstance(enc_img, ts.CKKSTensor)

    def test_one_hot_encode(self):
        labels = torch.tensor([0, 3, 5, 9])
        one_hot = one_hot_encode(labels, num_classes=10)

        assert one_hot.shape == (4, 10)
        assert torch.sum(one_hot).item() == 4

        for i, label in enumerate(labels):
            assert one_hot[i, label] == 1
            assert torch.sum(one_hot[i]) == 1


class TestFHEDataset:
    @pytest.fixture
    def small_dataloader(self):
        dataloader = load_mnist_data(batch_size=2, train=False)
        return dataloader

    @pytest.fixture
    def context(self):
        return create_context(poly_modulus_degree=8192, scale_bits=40)

    def test_fhe_dataset_creation(self, context, small_dataloader):
        dataset = FHEDataset(context, small_dataloader, max_samples=4)

        assert len(dataset) == 4
        assert len(dataset.labels) == 4

        enc_img, label = dataset[0]
        assert isinstance(enc_img, ts.CKKSTensor)
        assert isinstance(label.item(), int)

    def test_fhe_dataset_max_samples(self, context, small_dataloader):
        dataset = FHEDataset(context, small_dataloader, max_samples=3)
        assert len(dataset) == 3
