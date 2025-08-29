from typing import List, Optional, Tuple

import numpy as np
import tenseal as ts
import torch
from tqdm import tqdm

from model import FHEMLPClassifier
from utils import create_context, decrypt_tensor


class FHEInference:
    def __init__(self, model: FHEMLPClassifier, context: ts.Context):
        self.model = model
        self.context = context

    def predict_encrypted(self, encrypted_input: ts.CKKSTensor) -> int:
        encrypted_output = self.model.forward_encrypted(encrypted_input)

        decrypted_output = decrypt_tensor(encrypted_output)

        prediction = torch.argmax(decrypted_output).item()

        return prediction

    def predict_batch_encrypted(
        self, encrypted_inputs: List[ts.CKKSTensor], show_progress: bool = True
    ) -> List[int]:
        predictions = []

        iterator = (
            tqdm(encrypted_inputs, desc="Inferring")
            if show_progress
            else encrypted_inputs
        )

        for enc_input in iterator:
            pred = self.predict_encrypted(enc_input)
            predictions.append(pred)

        return predictions

    def predict_with_confidence(
        self, encrypted_input: ts.CKKSTensor
    ) -> Tuple[int, float]:
        encrypted_output = self.model.forward_encrypted(encrypted_input)

        decrypted_output = decrypt_tensor(encrypted_output)

        probabilities = torch.softmax(decrypted_output, dim=0)

        prediction = torch.argmax(probabilities).item()
        confidence = probabilities[prediction].item()

        return prediction, confidence

    def predict_plain(self, plain_input: torch.Tensor) -> int:
        if len(plain_input.shape) == 1:
            plain_input = plain_input.unsqueeze(0)

        plain_input = plain_input.view(plain_input.shape[0], -1)

        output = self.model.forward_plain(plain_input)

        prediction = torch.argmax(output, dim=-1)

        return prediction.item() if prediction.numel() == 1 else prediction.tolist()

    def evaluate_accuracy(
        self,
        encrypted_inputs: List[ts.CKKSTensor],
        true_labels: List[int],
        show_progress: bool = True,
    ) -> float:
        if len(encrypted_inputs) != len(true_labels):
            raise ValueError("Number of inputs and labels must match")

        correct = 0
        total = len(encrypted_inputs)

        iterator = zip(encrypted_inputs, true_labels)
        if show_progress:
            iterator = tqdm(iterator, total=total, desc="Evaluating")

        for enc_input, true_label in iterator:
            prediction = self.predict_encrypted(enc_input)
            if prediction == true_label:
                correct += 1

        accuracy = correct / total if total > 0 else 0.0
        return accuracy

    def get_confusion_matrix(
        self,
        encrypted_inputs: List[ts.CKKSTensor],
        true_labels: List[int],
        num_classes: int = 10,
    ) -> np.ndarray:
        confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)

        for enc_input, true_label in zip(encrypted_inputs, true_labels):
            prediction = self.predict_encrypted(enc_input)
            confusion_matrix[true_label, prediction] += 1

        return confusion_matrix


class SecureInferenceServer:
    def __init__(
        self, model_path: Optional[str] = None, context: Optional[ts.Context] = None
    ):
        if context is None:
            self.context = create_context()
        else:
            self.context = context

        self.model = None
        if model_path:
            self.load_model(model_path)

    def load_model(self, model_path: str):
        checkpoint = torch.load(model_path, map_location="cpu")

        model_config = checkpoint.get("model_config", {})
        self.model = FHEMLPClassifier(
            input_dim=model_config.get("input_dim", 784),
            hidden_dims=model_config.get("hidden_dims", [64]),  # Updated default
            num_classes=model_config.get("num_classes", 10),
            use_polynomial_activation=model_config.get("use_polynomial_activation", False),  # Default to linear
        )

        self.model.set_parameters(checkpoint["model_state"])

        self.inference_engine = FHEInference(self.model, self.context)

    def process_encrypted_request(
        self, encrypted_data: ts.CKKSTensor
    ) -> Tuple[int, float]:
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        prediction, confidence = self.inference_engine.predict_with_confidence(
            encrypted_data
        )

        return prediction, confidence

    def batch_process(
        self, encrypted_batch: List[ts.CKKSTensor]
    ) -> List[Tuple[int, float]]:
        results = []

        for enc_data in encrypted_batch:
            result = self.process_encrypted_request(enc_data)
            results.append(result)

        return results
