import warnings
from typing import List, Optional, Tuple, Union

import numpy as np
import tenseal as ts
import torch
from tqdm import tqdm

from model import FHEMLPClassifier
from utils import create_context, decrypt_tensor, encrypt_tensor


class ProductionFHEInference:
    """Production-ready FHE inference engine supporting both encrypted and plain inputs.

    This inference engine is designed for production deployment where:
    1. Clients can submit encrypted data for complete privacy
    2. Internal systems can use plaintext for performance when appropriate
    3. The server never needs to see sensitive client data
    """

    def __init__(
        self, model: FHEMLPClassifier, context: Optional[ts.Context] = None
    ):
        self.model = model
        self.context = context if context else create_context()

    def predict(
        self,
        input_data: Union[torch.Tensor, ts.CKKSTensor],
        return_confidence: bool = False,
    ) -> Union[int, Tuple[int, float]]:
        """Universal prediction method that handles both encrypted and plain inputs.

        Args:
            input_data: Either encrypted (ts.CKKSTensor) or plain (torch.Tensor) input
            return_confidence: Whether to return confidence score along with prediction

        Returns:
            Prediction class (int) or (prediction, confidence) if return_confidence=True
        """
        if isinstance(input_data, ts.CKKSTensor):
            # Encrypted input path
            return self._predict_encrypted(input_data, return_confidence)
        elif isinstance(input_data, torch.Tensor):
            # Plain input path
            return self._predict_plain(input_data, return_confidence)
        else:
            raise TypeError(
                f"Input must be torch.Tensor or ts.CKKSTensor, got {type(input_data)}"
            )

    def _predict_encrypted(
        self, encrypted_input: ts.CKKSTensor, return_confidence: bool = False
    ) -> Union[int, Tuple[int, float]]:
        """Predict on encrypted input without decrypting the input data."""
        # Forward pass entirely in encrypted space
        encrypted_output = self.model.forward_encrypted(encrypted_input)

        # Decrypt only the output (not the sensitive input)
        decrypted_output = decrypt_tensor(encrypted_output)

        if return_confidence:
            probabilities = torch.softmax(decrypted_output, dim=0)
            prediction = torch.argmax(probabilities).item()
            confidence = probabilities[prediction].item()
            return prediction, confidence
        else:
            prediction = torch.argmax(decrypted_output).item()
            return prediction

    def _predict_plain(
        self, plain_input: torch.Tensor, return_confidence: bool = False
    ) -> Union[int, Tuple[int, float]]:
        """Predict on plaintext input for performance when privacy is not required."""
        # Ensure correct input shape
        if len(plain_input.shape) == 1:
            plain_input = plain_input.unsqueeze(0)
        plain_input = plain_input.view(plain_input.shape[0], -1)

        # Forward pass on plaintext
        output = self.model.forward_plain(plain_input)

        if return_confidence:
            probabilities = torch.softmax(output, dim=-1)
            prediction = torch.argmax(probabilities, dim=-1)
            confidence = probabilities[0, prediction].item()
            prediction = prediction.item()
            return prediction, confidence
        else:
            prediction = torch.argmax(output, dim=-1)
            return (
                prediction.item()
                if prediction.numel() == 1
                else prediction.tolist()
            )

    def predict_batch(
        self,
        inputs: List[Union[torch.Tensor, ts.CKKSTensor]],
        return_confidence: bool = False,
        show_progress: bool = False,
    ) -> Union[List[int], List[Tuple[int, float]]]:
        """Batch prediction supporting mixed encrypted/plain inputs.

        Args:
            inputs: List of inputs (can be mix of encrypted and plain)
            return_confidence: Whether to return confidence scores
            show_progress: Whether to show progress bar

        Returns:
            List of predictions or (prediction, confidence) tuples
        """
        results = []

        iterator = (
            tqdm(inputs, desc="Batch inference") if show_progress else inputs
        )

        for input_data in iterator:
            result = self.predict(input_data, return_confidence)
            results.append(result)

        return results

    def encrypt_input(self, plain_input: torch.Tensor) -> ts.CKKSTensor:
        """Encrypt plaintext input for clients who want to submit encrypted data.

        This method can be used by clients to encrypt their data before sending
        to the inference server, ensuring their data remains private.
        """
        if len(plain_input.shape) > 1:
            plain_input = plain_input.flatten()
        return encrypt_tensor(self.context, plain_input)

    def evaluate_accuracy(
        self,
        inputs: List[Union[torch.Tensor, ts.CKKSTensor]],
        true_labels: List[int],
        show_progress: bool = True,
    ) -> float:
        """Evaluate accuracy on mixed encrypted/plain inputs."""
        if len(inputs) != len(true_labels):
            raise ValueError("Number of inputs and labels must match")

        correct = 0
        total = len(inputs)

        predictions = self.predict_batch(
            inputs, return_confidence=False, show_progress=show_progress
        )

        for pred, true_label in zip(predictions, true_labels):
            if pred == true_label:
                correct += 1

        return correct / total if total > 0 else 0.0

    def get_performance_stats(
        self,
        inputs: List[Union[torch.Tensor, ts.CKKSTensor]],
        true_labels: List[int],
    ) -> dict:
        """Get comprehensive performance statistics."""
        predictions_with_confidence = self.predict_batch(
            inputs, return_confidence=True
        )
        predictions = [pred for pred, _ in predictions_with_confidence]
        confidences = [conf for _, conf in predictions_with_confidence]

        # Basic accuracy
        accuracy = sum(
            1 for pred, true in zip(predictions, true_labels) if pred == true
        ) / len(predictions)

        # Confidence statistics
        avg_confidence = np.mean(confidences)
        confidence_std = np.std(confidences)

        # Per-class accuracy (for MNIST 0-9)
        per_class_accuracy = {}
        for class_id in range(10):
            class_preds = [
                pred
                for pred, true in zip(predictions, true_labels)
                if true == class_id
            ]
            class_labels = [true for true in true_labels if true == class_id]
            if len(class_labels) > 0:
                class_acc = sum(
                    1
                    for pred, true in zip(class_preds, class_labels)
                    if pred == true
                ) / len(class_labels)
                per_class_accuracy[class_id] = class_acc

        # Input type breakdown
        encrypted_count = sum(
            1 for inp in inputs if isinstance(inp, ts.CKKSTensor)
        )
        plain_count = len(inputs) - encrypted_count

        return {
            "overall_accuracy": accuracy,
            "average_confidence": avg_confidence,
            "confidence_std": confidence_std,
            "per_class_accuracy": per_class_accuracy,
            "total_samples": len(inputs),
            "encrypted_samples": encrypted_count,
            "plain_samples": plain_count,
            "predictions": predictions,
            "confidences": confidences,
        }


class SecureInferenceServer:
    """Production inference server that handles encrypted client requests.

    This server is designed for deployment scenarios where clients submit
    encrypted data and receive encrypted or plain results based on their needs.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        context: Optional[ts.Context] = None,
        allow_plain_input: bool = True,
    ):
        """Initialize the secure inference server.

        Args:
            model_path: Path to saved model checkpoint
            context: FHE context (will create default if None)
            allow_plain_input: Whether to accept plaintext inputs (for performance)
        """
        self.context = context if context else create_context()
        self.allow_plain_input = allow_plain_input
        self.model = None
        self.inference_engine = None

        if model_path:
            self.load_model(model_path)

    def load_model(self, model_path: str):
        """Load a trained FHE model from checkpoint."""
        checkpoint = torch.load(model_path, map_location="cpu")

        model_config = checkpoint.get("model_config", {})
        self.model = FHEMLPClassifier(
            input_dim=model_config.get("input_dim", 784),
            hidden_dims=model_config.get("hidden_dims", [64]),
            num_classes=model_config.get("num_classes", 10),
            use_polynomial_activation=model_config.get(
                "use_polynomial_activation", False
            ),
        )

        self.model.set_parameters(checkpoint["model_state"])
        self.inference_engine = ProductionFHEInference(
            self.model, self.context
        )

        print(f"Model loaded successfully from {model_path}")
        print(f"Architecture: {model_config}")

    def process_request(
        self,
        input_data: Union[torch.Tensor, ts.CKKSTensor],
        request_confidence: bool = False,
        client_id: Optional[str] = None,
    ) -> dict:
        """Process an inference request from a client.

        Args:
            input_data: Client's input data (encrypted or plain)
            request_confidence: Whether client wants confidence score
            client_id: Optional client identifier for logging

        Returns:
            Response dictionary with prediction and metadata
        """
        if self.inference_engine is None:
            raise ValueError("No model loaded. Call load_model() first.")

        # Check input type permissions
        if isinstance(input_data, torch.Tensor) and not self.allow_plain_input:
            raise ValueError(
                "This server only accepts encrypted inputs for security."
            )

        # Process the request
        start_time = (
            torch.cuda.Event(enable_timing=True)
            if torch.cuda.is_available()
            else None
        )

        try:
            result = self.inference_engine.predict(
                input_data, request_confidence
            )

            if request_confidence:
                prediction, confidence = result
                response = {
                    "prediction": prediction,
                    "confidence": confidence,
                    "input_type": "encrypted"
                    if isinstance(input_data, ts.CKKSTensor)
                    else "plain",
                    "status": "success",
                }
            else:
                response = {
                    "prediction": result,
                    "input_type": "encrypted"
                    if isinstance(input_data, ts.CKKSTensor)
                    else "plain",
                    "status": "success",
                }

            if client_id:
                response["client_id"] = client_id

            return response

        except Exception as e:
            return {
                "status": "error",
                "error_message": str(e),
                "input_type": "encrypted"
                if isinstance(input_data, ts.CKKSTensor)
                else "plain",
            }

    def batch_process(self, batch_requests: List[dict]) -> List[dict]:
        """Process a batch of inference requests.

        Args:
            batch_requests: List of request dictionaries with keys:
                - 'input_data': The input tensor/encrypted tensor
                - 'request_confidence': Whether to return confidence (default False)
                - 'client_id': Optional client identifier

        Returns:
            List of response dictionaries
        """
        responses = []

        for request in batch_requests:
            input_data = request["input_data"]
            request_confidence = request.get("request_confidence", False)
            client_id = request.get("client_id")

            response = self.process_request(
                input_data, request_confidence, client_id
            )
            responses.append(response)

        return responses

    def get_server_stats(self) -> dict:
        """Get server configuration and status information."""
        return {
            "model_loaded": self.model is not None,
            "allow_plain_input": self.allow_plain_input,
            "fhe_context_configured": self.context is not None,
            "model_architecture": {
                "input_dim": self.model.layers[0].in_features
                if self.model
                else None,
                "hidden_dims": [
                    layer.out_features for layer in self.model.layers[:-1]
                ]
                if self.model
                else None,
                "num_classes": self.model.layers[-1].out_features
                if self.model
                else None,
            }
            if self.model
            else None,
        }
