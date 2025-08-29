import numpy as np
import pytest
import torch

from inference import FHEInference, SecureInferenceServer
from model import FHEMLPClassifier
from utils import create_context, encrypt_tensor


class TestFHEInference:
    @pytest.fixture
    def context(self):
        return create_context(poly_modulus_degree=8192, scale_bits=40)

    @pytest.fixture
    def model(self):
        return FHEMLPClassifier(input_dim=10, hidden_dims=[5], num_classes=3, use_polynomial_activation=False)

    @pytest.fixture
    def inference_engine(self, model, context):
        return FHEInference(model, context)

    def test_predict_encrypted(self, inference_engine, context):
        input_tensor = torch.randn(10)
        encrypted_input = encrypt_tensor(context, input_tensor)

        prediction = inference_engine.predict_encrypted(encrypted_input)

        assert isinstance(prediction, int)
        assert 0 <= prediction < 3

    def test_predict_with_confidence(self, inference_engine, context):
        input_tensor = torch.randn(10)
        encrypted_input = encrypt_tensor(context, input_tensor)

        prediction, confidence = inference_engine.predict_with_confidence(
            encrypted_input
        )

        assert isinstance(prediction, int)
        assert 0 <= prediction < 3
        assert isinstance(confidence, float)
        assert 0 <= confidence <= 1

    def test_predict_plain(self, inference_engine):
        input_tensor = torch.randn(10)

        prediction = inference_engine.predict_plain(input_tensor)

        assert isinstance(prediction, int)
        assert 0 <= prediction < 3

    def test_predict_batch_encrypted(self, inference_engine, context):
        inputs = [torch.randn(10) for _ in range(3)]
        encrypted_inputs = [encrypt_tensor(context, inp) for inp in inputs]

        predictions = inference_engine.predict_batch_encrypted(
            encrypted_inputs, show_progress=False
        )

        assert len(predictions) == 3
        for pred in predictions:
            assert isinstance(pred, int)
            assert 0 <= pred < 3

    def test_evaluate_accuracy(self, inference_engine, context):
        inputs = [torch.randn(10) for _ in range(5)]
        encrypted_inputs = [encrypt_tensor(context, inp) for inp in inputs]
        true_labels = [0, 1, 2, 0, 1]

        accuracy = inference_engine.evaluate_accuracy(
            encrypted_inputs, true_labels, show_progress=False
        )

        assert isinstance(accuracy, float)
        assert 0 <= accuracy <= 1

    def test_get_confusion_matrix(self, inference_engine, context):
        inputs = [torch.randn(10) for _ in range(5)]
        encrypted_inputs = [encrypt_tensor(context, inp) for inp in inputs]
        true_labels = [0, 1, 2, 0, 1]

        confusion_matrix = inference_engine.get_confusion_matrix(
            encrypted_inputs, true_labels, num_classes=3
        )

        assert confusion_matrix.shape == (3, 3)
        assert np.sum(confusion_matrix) == 5


class TestSecureInferenceServer:
    @pytest.fixture
    def context(self):
        return create_context(poly_modulus_degree=8192, scale_bits=40)

    @pytest.fixture
    def server(self, context):
        return SecureInferenceServer(context=context)

    def test_initialization(self, server):
        assert server.model is None
        assert server.context is not None

    def test_process_without_model(self, server, context):
        input_tensor = torch.randn(10)
        encrypted_input = encrypt_tensor(context, input_tensor)

        with pytest.raises(ValueError, match="Model not loaded"):
            server.process_encrypted_request(encrypted_input)

    def test_batch_process(self, server, context, tmp_path):
        # Create and save a model
        model = FHEMLPClassifier(input_dim=10, hidden_dims=[5], num_classes=3, use_polynomial_activation=False)
        model_path = tmp_path / "test_model.pt"

        checkpoint = {
            "model_state": model.get_parameters(),
            "model_config": {"input_dim": 10, "hidden_dims": [5], "num_classes": 3, "use_polynomial_activation": False},
        }
        torch.save(checkpoint, model_path)

        # Load model in server
        server.load_model(str(model_path))

        # Test batch processing
        inputs = [torch.randn(10) for _ in range(3)]
        encrypted_inputs = [encrypt_tensor(context, inp) for inp in inputs]

        results = server.batch_process(encrypted_inputs)

        assert len(results) == 3
        for pred, conf in results:
            assert isinstance(pred, int)
            assert 0 <= pred < 3
            assert isinstance(conf, float)
            assert 0 <= conf <= 1
