from typing import List, Optional, Tuple

import tenseal as ts
import torch
import torch.nn as nn
import torch.optim as optim

from model import FHEMLPClassifier, TorchMLPClassifier
from utils import decrypt_tensor


class FHETrainer:
    def __init__(
        self, model: FHEMLPClassifier, learning_rate: float = 0.01, device: str = "cpu"
    ):
        self.model = model
        self.learning_rate = learning_rate
        self.device = device

        self.proxy_model = self._create_proxy_model()
        self.optimizer = optim.SGD(self.proxy_model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()

    def _create_proxy_model(self) -> TorchMLPClassifier:
        assert len(self.model.layers) > 0

        input_dim = self.model.layers[0].in_features
        hidden_dims = []
        for i in range(len(self.model.layers) - 1):
            hidden_dims.append(self.model.layers[i].out_features)
        num_classes = self.model.layers[-1].out_features

        proxy = TorchMLPClassifier(input_dim, hidden_dims, num_classes)

        idx = 0
        for layer in self.model.layers:
            proxy.model[
                idx * 2 if idx < len(self.model.layers) - 1 else -1
            ].weight.data = layer.weight
            proxy.model[
                idx * 2 if idx < len(self.model.layers) - 1 else -1
            ].bias.data = layer.bias
            idx += 1

        return proxy.to(self.device)

    def _sync_weights_to_fhe(self):
        for i, layer in enumerate(self.model.layers):
            if i < len(self.model.layers) - 1:
                linear_idx = i * 2
            else:
                linear_idx = -1

            layer.update_weights(
                self.proxy_model.model[linear_idx].weight.data.cpu(),
                self.proxy_model.model[linear_idx].bias.data.cpu(),
            )

    def train_on_encrypted_batch(
        self, encrypted_images: List[ts.CKKSTensor], labels: torch.Tensor
    ) -> float:
        batch_loss = 0.0

        for enc_img, label in zip(encrypted_images, labels):
            encrypted_output = self.model.forward_encrypted(enc_img)

            decrypted_output = decrypt_tensor(encrypted_output)

            decrypted_output = decrypted_output.unsqueeze(0).requires_grad_(True)
            label = label.unsqueeze(0)

            self.optimizer.zero_grad()
            loss = self.criterion(decrypted_output, label)
            loss.backward()

            self._approximate_gradients(enc_img, label, decrypted_output)

            self.optimizer.step()
            self._sync_weights_to_fhe()

            batch_loss += loss.item()

        return batch_loss / len(encrypted_images)

    def _approximate_gradients(
        self, encrypted_input: ts.CKKSTensor, label: torch.Tensor, output: torch.Tensor
    ):
        # For FHE training, we approximate gradients using the decrypted output
        # and encrypted input. This is a simplified approach for demonstration.
        
        # Get target one-hot encoding
        output_flat = output.squeeze() if output.dim() > 1 else output
        target_one_hot = torch.zeros_like(output_flat)
        label_item = label.item() if label.dim() > 0 else label
        target_one_hot[label_item] = 1.0
        
        # Compute output gradient (cross-entropy gradient) - for reference
        # softmax_output = torch.softmax(output, dim=0)
        # grad_output = softmax_output - target_one_hot
        
        # For the encrypted training, we need to approximate the input gradients
        # In a full FHE implementation, this would be more sophisticated
        # Here we use the current proxy model to compute approximate gradients
        
        # Create a dummy input tensor to compute gradients
        decrypted_input = decrypt_tensor(encrypted_input)
        decrypted_input_batch = decrypted_input.unsqueeze(0).requires_grad_(True)
        
        # Forward pass through proxy model to get gradients
        proxy_output = self.proxy_model(decrypted_input_batch)
        label_tensor = torch.tensor([label_item]) if isinstance(label_item, int) else torch.tensor([label_item.item()])
        proxy_loss = self.criterion(proxy_output, label_tensor)
        
        # Compute gradients with respect to the input (for gradient approximation)
        proxy_loss.backward(retain_graph=True)
        
        # The gradients are automatically accumulated in the proxy model parameters
        # This approximation allows us to train on encrypted data by using
        # the decrypted outputs to guide the learning process

    def train_on_plain_batch(self, images: torch.Tensor, labels: torch.Tensor) -> float:
        images = images.view(images.shape[0], -1).to(self.device)
        labels = labels.to(self.device)

        self.optimizer.zero_grad()
        outputs = self.proxy_model(images)
        loss = self.criterion(outputs, labels)
        loss.backward()
        self.optimizer.step()

        self._sync_weights_to_fhe()

        return loss.item()

    def evaluate_encrypted(
        self, encrypted_dataset: List[Tuple[ts.CKKSTensor, int]]
    ) -> Tuple[float, float]:
        correct = 0
        total = 0
        total_loss = 0.0

        with torch.no_grad():
            for enc_img, label in encrypted_dataset:
                encrypted_output = self.model.forward_encrypted(enc_img)
                decrypted_output = decrypt_tensor(encrypted_output)

                pred = torch.argmax(decrypted_output)
                correct += (pred == label).item()
                total += 1

                loss = self.criterion(
                    decrypted_output.unsqueeze(0), torch.tensor([label])
                )
                total_loss += loss.item()

        accuracy = correct / total if total > 0 else 0
        avg_loss = total_loss / total if total > 0 else 0

        return accuracy, avg_loss

    def evaluate_plain(self, dataloader) -> Tuple[float, float]:
        self.proxy_model.eval()
        correct = 0
        total = 0
        total_loss = 0.0

        with torch.no_grad():
            for images, labels in dataloader:
                images = images.view(images.shape[0], -1).to(self.device)
                labels = labels.to(self.device)

                outputs = self.proxy_model(images)
                loss = self.criterion(outputs, labels)

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                total_loss += loss.item() * labels.size(0)

        self.proxy_model.train()

        accuracy = correct / total if total > 0 else 0
        avg_loss = total_loss / total if total > 0 else 0

        return accuracy, avg_loss


class HybridFHETrainer(FHETrainer):
    def __init__(
        self,
        model: FHEMLPClassifier,
        learning_rate: float = 0.01,
        device: str = "cpu",
        noise_scale: float = 0.1,
    ):
        super().__init__(model, learning_rate, device)
        self.noise_scale = noise_scale

    def train_mixed_batch(
        self,
        encrypted_images: Optional[List[ts.CKKSTensor]],
        plain_images: Optional[torch.Tensor],
        labels: torch.Tensor,
    ) -> float:
        total_loss = 0.0
        count = 0

        if encrypted_images:
            enc_loss = self.train_on_encrypted_batch(
                encrypted_images[: len(labels) // 2], labels[: len(labels) // 2]
            )
            total_loss += enc_loss * (len(labels) // 2)
            count += len(labels) // 2

        if plain_images is not None and len(plain_images) > 0:
            plain_loss = self.train_on_plain_batch(
                plain_images[len(labels) // 2 :], labels[len(labels) // 2 :]
            )
            total_loss += plain_loss * (len(labels) - len(labels) // 2)
            count += len(labels) - len(labels) // 2

        return total_loss / count if count > 0 else 0.0
