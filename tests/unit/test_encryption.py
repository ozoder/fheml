"""Unit tests for FHE encryption operations."""

import pytest
import torch
import tenseal as ts

from src.fheml.core.context import FHEContextManager
from src.fheml.core.encryption import FHEEncryption


class TestFHEEncryption:
    """Test cases for FHE encryption operations."""

    @pytest.fixture
    def context(self):
        """Create FHE context for testing."""
        manager = FHEContextManager()
        return manager.create_context()

    @pytest.fixture
    def encryption(self, context):
        """Create encryption handler for testing."""
        return FHEEncryption(context)

    def test_init(self, context):
        """Test encryption handler initialization."""
        encryption = FHEEncryption(context)
        assert encryption.context == context

    def test_encrypt_1d_tensor(self, encryption):
        """Test encryption of 1D tensor."""
        tensor = torch.tensor([1.0, 2.0, 3.0, 4.0])
        encrypted = encryption.encrypt_tensor(tensor)

        assert isinstance(encrypted, ts.CKKSTensor)

    def test_encrypt_2d_tensor(self, encryption):
        """Test encryption of 2D tensor (flattened)."""
        tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        encrypted = encryption.encrypt_tensor(tensor)

        assert isinstance(encrypted, ts.CKKSTensor)

    def test_decrypt_tensor(self, encryption):
        """Test tensor decryption."""
        original = torch.tensor([1.0, 2.0, 3.0, 4.0])
        encrypted = encryption.encrypt_tensor(original)
        decrypted = encryption.decrypt_tensor(encrypted)

        assert isinstance(decrypted, torch.Tensor)
        assert decrypted.dtype == torch.float32
        # Allow for small numerical errors in FHE
        assert torch.allclose(original, decrypted, atol=1e-3)

    def test_encrypt_decrypt_round_trip(self, encryption):
        """Test encrypt-decrypt round trip preserves values."""
        original = torch.tensor([1.5, -2.3, 0.0, 4.7])
        encrypted = encryption.encrypt_tensor(original)
        decrypted = encryption.decrypt_tensor(encrypted)

        # FHE introduces small numerical errors
        assert torch.allclose(original, decrypted, atol=1e-2)

    def test_encrypt_batch(self, encryption):
        """Test batch encryption of images."""
        batch_size = 3
        image_size = 28 * 28
        images = torch.randn(batch_size, 1, 28, 28)  # MNIST-like
        labels = torch.tensor([0, 1, 2])

        encrypted_images, returned_labels = encryption.encrypt_batch(images, labels)

        assert len(encrypted_images) == batch_size
        assert torch.equal(returned_labels, labels)

        for encrypted_img in encrypted_images:
            assert isinstance(encrypted_img, ts.CKKSTensor)

    def test_encrypt_batch_correct_shape(self, encryption):
        """Test that batch encryption preserves the correct flattened shape."""
        images = torch.randn(2, 3, 4, 4)  # 2 images, 3 channels, 4x4
        labels = torch.tensor([0, 1])

        encrypted_images, _ = encryption.encrypt_batch(images, labels)

        # Each encrypted image should correspond to flattened version
        for i, encrypted_img in enumerate(encrypted_images):
            original_flat = images[i].flatten()
            decrypted = encryption.decrypt_tensor(encrypted_img)

            # Should have same number of elements
            assert len(decrypted) == len(original_flat)
            # Values should be approximately equal
            assert torch.allclose(original_flat, decrypted, atol=1e-2)

    def test_encrypt_tensor_invalid_input(self, encryption):
        """Test encryption with invalid input."""
        # Test with non-tensor input should raise error
        with pytest.raises(ValueError, match="Tensor encryption failed"):
            encryption.encrypt_tensor("not a tensor")

    def test_decrypt_tensor_invalid_input(self, encryption):
        """Test decryption with invalid input."""
        # Test with non-encrypted input should raise ValueError
        with pytest.raises(ValueError):
            encryption.decrypt_tensor("not encrypted")