from typing import List, Union

import tenseal as ts
import torch
import torch.nn as nn


class FHELinearLayer:
    def __init__(self, in_features: int, out_features: int):
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.randn(out_features, in_features) * 0.01
        self.bias = torch.zeros(out_features)

    def forward_encrypted(self, x: ts.CKKSTensor) -> ts.CKKSTensor:
        bias_plain = self.bias.numpy().tolist()

        # For vector x matrix multiplication, we need to transpose
        # x is shape (input_dim,) and weight is (out_features, in_features)
        # We want: x @ weight.T = (input_dim,) @ (in_features, out_features)
        weight_t_plain = self.weight.t().numpy().tolist()
        
        result = x.dot(weight_t_plain)
        # Rescale to manage noise growth
        if hasattr(result, 'rescale_to_next'):
            result.rescale_to_next()
        result = result + bias_plain

        return result

    def forward_plain(self, x: torch.Tensor) -> torch.Tensor:
        return torch.mm(x, self.weight.t()) + self.bias

    def update_weights(self, weight: torch.Tensor, bias: torch.Tensor):
        self.weight = weight
        self.bias = bias


class FHEPolynomialActivation:
    """FHE-friendly polynomial approximation of ReLU using degree-2 polynomial.
    
    Uses the approximation: f(x) ≈ 0.5 * x + 0.25 * x^2 for x in [-1, 1]
    This approximates ReLU behavior while using only one multiplication.
    """
    @staticmethod
    def forward(
        x: Union[ts.CKKSTensor, torch.Tensor],
    ) -> Union[ts.CKKSTensor, torch.Tensor]:
        if isinstance(x, torch.Tensor):
            # For plain computation, use actual ReLU
            return torch.relu(x)
        else:
            # FHE polynomial approximation: 0.5 * x + 0.25 * x^2
            x_squared = x.square()
            if hasattr(x_squared, 'rescale_to_next'):
                x_squared.rescale_to_next()
            
            result = x * 0.5 + x_squared * 0.25
            return result


class FHELinearActivation:
    """Linear activation (identity function) - no multiplicative depth."""
    @staticmethod
    def forward(
        x: Union[ts.CKKSTensor, torch.Tensor],
    ) -> Union[ts.CKKSTensor, torch.Tensor]:
        return x


class FHEMLPClassifier:
    def __init__(
        self,
        input_dim: int = 784,
        hidden_dims: List[int] = [128],  # Reduced to single hidden layer
        num_classes: int = 10,
        use_polynomial_activation: bool = True,
    ) -> None:
        self.layers: list[FHELinearLayer] = []
        self.activations = []
        self.use_polynomial_activation = use_polynomial_activation

        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            self.layers.append(FHELinearLayer(prev_dim, hidden_dim))
            if use_polynomial_activation:
                self.activations.append(FHEPolynomialActivation())
            else:
                self.activations.append(FHELinearActivation())
            prev_dim = hidden_dim

        # Output layer - always linear
        self.layers.append(FHELinearLayer(prev_dim, num_classes))

    def forward_encrypted(self, x: ts.CKKSTensor) -> ts.CKKSTensor:
        for i, layer in enumerate(self.layers[:-1]):
            x = layer.forward_encrypted(x)
            x = self.activations[i].forward(x)

        x = self.layers[-1].forward_encrypted(x)

        return x

    def forward_plain(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers[:-1]):
            x = layer.forward_plain(x)
            x = self.activations[i].forward(x)

        x = self.layers[-1].forward_plain(x)

        return x

    def get_parameters(self) -> List[torch.Tensor]:
        params = []
        for layer in self.layers:
            params.append(layer.weight)
            params.append(layer.bias)

        return params

    def set_parameters(self, params: List[torch.Tensor]):
        idx = 0
        for layer in self.layers:
            layer.weight = params[idx]
            layer.bias = params[idx + 1]
            idx += 2


class TorchMLPClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int = 784,
        hidden_dims: List[int] = [128, 64],
        num_classes: int = 10,
    ):
        super().__init__()
        layers = []

        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, num_classes))

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
