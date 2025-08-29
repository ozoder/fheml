import torch

from model import (
    FHELinearLayer,
    FHEMLPClassifier,
    FHEPolynomialActivation,
    TorchMLPClassifier,
)


class TestFHELinearLayer:
    def test_initialization(self):
        layer = FHELinearLayer(10, 5)
        assert layer.in_features == 10
        assert layer.out_features == 5
        assert layer.weight.shape == (5, 10)
        assert layer.bias.shape == (5,)

    def test_forward_plain(self):
        layer = FHELinearLayer(10, 5)
        x = torch.randn(3, 10)
        output = layer.forward_plain(x)
        assert output.shape == (3, 5)

    def test_update_weights(self):
        layer = FHELinearLayer(10, 5)
        new_weight = torch.ones(5, 10)
        new_bias = torch.zeros(5)
        layer.update_weights(new_weight, new_bias)
        assert torch.allclose(layer.weight, new_weight)
        assert torch.allclose(layer.bias, new_bias)


class TestFHEPolynomialActivation:
    def test_forward_plain(self):
        activation = FHEPolynomialActivation()
        x = torch.tensor([1.0, 2.0, -1.0])  # Test values within reasonable range
        output = activation.forward(x)
        # For plain computation, it should use ReLU
        expected = torch.relu(x)
        assert torch.allclose(output, expected)


class TestFHEMLPClassifier:
    def test_initialization(self):
        model = FHEMLPClassifier(input_dim=784, hidden_dims=[128, 64], num_classes=10)
        assert len(model.layers) == 3
        assert model.layers[0].in_features == 784
        assert model.layers[0].out_features == 128
        assert model.layers[1].in_features == 128
        assert model.layers[1].out_features == 64
        assert model.layers[2].in_features == 64
        assert model.layers[2].out_features == 10

    def test_forward_plain(self):
        model = FHEMLPClassifier(input_dim=10, hidden_dims=[5], num_classes=3, use_polynomial_activation=False)
        x = torch.randn(1, 10)  # Add batch dimension
        output = model.forward_plain(x)
        assert output.shape == (1, 3)

    def test_get_set_parameters(self):
        model = FHEMLPClassifier(input_dim=10, hidden_dims=[5], num_classes=3, use_polynomial_activation=False)
        params = model.get_parameters()
        assert len(params) == 4  # 2 layers × (weight + bias)

        new_params = [torch.ones_like(p) for p in params]
        model.set_parameters(new_params)
        updated_params = model.get_parameters()

        for new_p, updated_p in zip(new_params, updated_params):
            assert torch.allclose(new_p, updated_p)


class TestTorchMLPClassifier:
    def test_initialization(self):
        model = TorchMLPClassifier(input_dim=784, hidden_dims=[128, 64], num_classes=10)
        assert isinstance(model, torch.nn.Module)

    def test_forward(self):
        model = TorchMLPClassifier(input_dim=10, hidden_dims=[5], num_classes=3)
        x = torch.randn(2, 10)
        output = model(x)
        assert output.shape == (2, 3)
