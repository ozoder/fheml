"""Unit tests for FHE neural network layers."""

import pytest
import torch
import tenseal as ts

from src.fheml.core.context import FHEContextManager
from src.fheml.core.operations import FHEOperations
from src.fheml.core.encryption import FHEEncryption
from src.fheml.models.layers import FHELinearLayer


class TestFHELinearLayer:
    """Test cases for FHE linear layer."""

    @pytest.fixture
    def context(self):
        """Create FHE context for testing."""
        manager = FHEContextManager()
        return manager.create_context()

    @pytest.fixture
    def operations(self, context):
        """Create FHE operations for testing."""
        return FHEOperations(context)

    @pytest.fixture
    def encryption(self, context):
        """Create FHE encryption for testing."""
        return FHEEncryption(context)

    def test_init(self):
        """Test layer initialization."""
        layer = FHELinearLayer(in_features=10, out_features=5)

        assert layer.in_features == 10
        assert layer.out_features == 5
        assert layer.weight.shape == (5, 10)
        assert layer.bias.shape == (5,)

    def test_xavier_initialization(self):
        """Test Xavier initialization produces reasonable values."""
        layer = FHELinearLayer(in_features=100, out_features=50)

        # Xavier initialization should have std = sqrt(2/(in+out))
        expected_std = (2.0 / (100 + 50)) ** 0.5

        # Check that weights are roughly in expected range
        weight_std = layer.weight.std().item()
        assert 0.5 * expected_std < weight_std < 2.0 * expected_std

        # Bias should be initialized to zero
        assert torch.allclose(layer.bias, torch.zeros(50))

    def test_forward_plain(self):
        """Test forward pass with plain tensors."""
        layer = FHELinearLayer(in_features=3, out_features=2)

        # Set known weights for testing
        layer.weight = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        layer.bias = torch.tensor([0.1, 0.2])

        # Test input
        x = torch.tensor([[1.0, 2.0, 3.0]])  # batch_size=1

        # Forward pass
        output = layer.forward_plain(x)

        # Expected: [1*1 + 2*2 + 3*3] + 0.1 = 14.1
        #          [1*4 + 2*5 + 3*6] + 0.2 = 32.2
        expected = torch.tensor([[14.1, 32.2]])

        assert torch.allclose(output, expected, atol=1e-5)

    def test_forward_encrypted(self, encryption, operations):
        """Test forward pass with encrypted tensors."""
        layer = FHELinearLayer(in_features=3, out_features=2)

        # Set known weights
        layer.weight = torch.tensor([[1.0, 0.5, 0.0], [0.0, 1.0, 0.5]])
        layer.bias = torch.tensor([0.1, 0.2])

        # Test input
        x_plain = torch.tensor([2.0, 4.0, 6.0])
        x_encrypted = encryption.encrypt_tensor(x_plain)

        # Forward pass
        output_encrypted = layer.forward_encrypted(x_encrypted, operations)

        # Decrypt and verify
        output_decrypted = encryption.decrypt_tensor(output_encrypted)

        # Expected calculation manually:
        # output[0] = 2.0*1.0 + 4.0*0.5 + 6.0*0.0 + 0.1 = 2.0 + 2.0 + 0.0 + 0.1 = 4.1
        # output[1] = 2.0*0.0 + 4.0*1.0 + 6.0*0.5 + 0.2 = 0.0 + 4.0 + 3.0 + 0.2 = 7.2
        expected = torch.tensor([4.1, 7.2])

        assert torch.allclose(output_decrypted, expected, atol=1e-1)  # FHE has larger tolerance

    def test_update_weights(self):
        """Test weight and bias updates."""
        layer = FHELinearLayer(in_features=2, out_features=3)

        new_weight = torch.randn(3, 2)
        new_bias = torch.randn(3)

        layer.update_weights(new_weight, new_bias)

        assert torch.equal(layer.weight, new_weight)
        assert torch.equal(layer.bias, new_bias)

    def test_get_parameters(self):
        """Test parameter retrieval."""
        layer = FHELinearLayer(in_features=2, out_features=3)

        params = layer.get_parameters()

        assert len(params) == 2
        assert torch.equal(params[0], layer.weight)
        assert torch.equal(params[1], layer.bias)

    def test_set_parameters(self):
        """Test parameter setting."""
        layer = FHELinearLayer(in_features=2, out_features=3)

        new_weight = torch.randn(3, 2)
        new_bias = torch.randn(3)

        layer.set_parameters(new_weight, new_bias)

        assert torch.equal(layer.weight, new_weight)
        assert torch.equal(layer.bias, new_bias)