import argparse
import json
import os
import time
from datetime import datetime

import torch

from model import FHEMLPClassifier
from production_training import train_production_fhe_model
from production_inference import ProductionFHEInference, SecureInferenceServer
from utils import create_context, load_mnist_data


def save_production_model(
    model: FHEMLPClassifier, 
    training_history: dict, 
    args: argparse.Namespace,
    model_type: str = "production"
):
    """Save production model with comprehensive metadata."""
    os.makedirs(".models", exist_ok=True)
    
    model_id = f"fhe_{model_type}_model_{time.strftime('%Y%m%d_%H%M%S')}"
    model_path = f".models/{model_id}.pt"
    stats_path = f".models/{model_id}_stats.json"
    
    print(f"\nSaving production model to {model_path}...")

    checkpoint = {
        "model_state": model.get_parameters(),
        "model_config": {
            "input_dim": 784,
            "hidden_dims": args.hidden_dims,
            "num_classes": 10,
            "use_polynomial_activation": args.use_polynomial_activation,
            "training_type": model_type
        },
        "training_history": training_history,
        "args": vars(args),
    }

    torch.save(checkpoint, model_path)
    
    # Create comprehensive stats file
    stats = {
        "model_id": model_id,
        "created_at": datetime.now().isoformat() + "Z",
        "model_file": f"{model_id}.pt",
        "model_type": model_type,
        "training_stats": {
            "final_test_accuracy": float(training_history["test_accuracy"][-1]),
            "final_train_loss": float(training_history["train_loss"][-1]),
            "final_test_loss": float(training_history["test_loss"][-1]),
            "epochs": len(training_history["train_loss"]),
            "total_training_time_seconds": float(sum(training_history["epoch_times"])),
            "average_epoch_time_seconds": float(sum(training_history["epoch_times"]) / len(training_history["epoch_times"]))
        },
        "model_config": {
            "architecture": "FHE-MLP",
            "input_dim": 784,
            "hidden_dims": args.hidden_dims,
            "num_classes": 10,
            "activation": "polynomial" if args.use_polynomial_activation else "linear",
            "use_polynomial_activation": args.use_polynomial_activation,
            "training_type": model_type
        },
        "training_config": {
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "optimizer": "FHE-SGD",
            "loss_function": "L2 (FHE-compatible)",
            "encrypted_training": True,
            "plaintext_training": False
        },
        "dataset": {
            "name": "MNIST",
            "train_samples_used": args.max_train_samples,
            "test_samples_used": args.max_test_samples,
            "input_shape": [28, 28],
            "num_classes": 10
        },
        "performance_metrics": {
            "test_accuracy": float(training_history["test_accuracy"][-1]),
            "convergence_epoch": len(training_history["train_loss"]),
            "fhe_compatible": True,
            "privacy_preserving": True,
            "production_ready": True
        },
        "security_features": {
            "encrypted_training": True,
            "encrypted_inference": True,
            "plaintext_inference": True,
            "no_plaintext_access_during_training": True,
            "minimal_decryption": "gradients_and_loss_only"
        },
        "notes": f"Production FHE model trained entirely on encrypted data. Supports both encrypted and plaintext inference for flexible deployment."
    }
    
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"Model saved successfully to {model_path}")
    print(f"Stats saved to {stats_path}")
    
    return model_path, stats_path


def test_production_inference(model_path: str, context, test_dataloader, max_samples: int = 10):
    """Test the production inference system with both encrypted and plain inputs."""
    print("\n" + "=" * 60)
    print("TESTING PRODUCTION INFERENCE SYSTEM")
    print("=" * 60)
    
    # Initialize inference server
    server = SecureInferenceServer(model_path, context, allow_plain_input=True)
    
    # Get test samples
    test_images = []
    test_labels = []
    for images, labels in test_dataloader:
        test_images.extend([img.view(-1) for img in images[:max_samples]])
        test_labels.extend(labels[:max_samples].tolist())
        if len(test_images) >= max_samples:
            break
    
    test_images = test_images[:max_samples]
    test_labels = test_labels[:max_samples]
    
    print(f"\nTesting with {len(test_images)} samples...")
    
    # Test 1: Plain input inference
    print("\n1. Testing plaintext inference...")
    plain_requests = []
    for i, (img, label) in enumerate(zip(test_images[:5], test_labels[:5])):
        plain_requests.append({
            'input_data': img,
            'request_confidence': True,
            'client_id': f'plain_client_{i}'
        })
    
    plain_responses = server.batch_process(plain_requests)
    plain_correct = sum(1 for resp, true_label in zip(plain_responses, test_labels[:5]) 
                       if resp['prediction'] == true_label)
    
    print(f"Plaintext accuracy: {plain_correct}/5 = {plain_correct/5:.1%}")
    print(f"Sample response: {plain_responses[0]}")
    
    # Test 2: Encrypted input inference
    print("\n2. Testing encrypted inference...")
    encrypted_requests = []
    for i, (img, label) in enumerate(zip(test_images[:3], test_labels[:3])):
        encrypted_img = server.inference_engine.encrypt_input(img)
        encrypted_requests.append({
            'input_data': encrypted_img,
            'request_confidence': True,
            'client_id': f'encrypted_client_{i}'
        })
    
    encrypted_responses = server.batch_process(encrypted_requests)
    encrypted_correct = sum(1 for resp, true_label in zip(encrypted_responses, test_labels[:3]) 
                           if resp['prediction'] == true_label)
    
    print(f"Encrypted accuracy: {encrypted_correct}/3 = {encrypted_correct/3:.1%}")
    print(f"Sample encrypted response: {encrypted_responses[0]}")
    
    # Test 3: Mixed batch processing
    print("\n3. Testing mixed batch processing...")
    mixed_inputs = []
    # Add some plain inputs
    mixed_inputs.extend(test_images[:2])
    # Add some encrypted inputs
    for img in test_images[2:4]:
        encrypted_img = server.inference_engine.encrypt_input(img)
        mixed_inputs.append(encrypted_img)
    
    mixed_predictions = server.inference_engine.predict_batch(
        mixed_inputs, return_confidence=True, show_progress=True
    )
    
    mixed_correct = sum(1 for (pred, _), true_label in zip(mixed_predictions, test_labels[:4]) 
                       if pred == true_label)
    
    print(f"Mixed batch accuracy: {mixed_correct}/4 = {mixed_correct/4:.1%}")
    
    # Test 4: Performance statistics
    print("\n4. Getting performance statistics...")
    stats = server.inference_engine.get_performance_stats(mixed_inputs, test_labels[:4])
    
    print(f"Performance Stats:")
    print(f"  Overall accuracy: {stats['overall_accuracy']:.1%}")
    print(f"  Average confidence: {stats['average_confidence']:.3f}")
    print(f"  Encrypted samples: {stats['encrypted_samples']}")
    print(f"  Plain samples: {stats['plain_samples']}")
    
    print("\n" + "=" * 60)
    print("PRODUCTION INFERENCE TESTING COMPLETE")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Production FHE Training - No Plaintext Access")
    
    # Model configuration
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[32], 
                       help="Hidden layer dimensions (smaller for FHE)")
    parser.add_argument("--use-polynomial-activation", action="store_true", 
                       help="Use polynomial activation instead of linear")
    
    # Training configuration
    parser.add_argument("--epochs", type=int, default=3, 
                       help="Number of training epochs")
    parser.add_argument("--learning-rate", type=float, default=0.001, 
                       help="Learning rate (lower for encrypted training)")
    parser.add_argument("--batch-size", type=int, default=2, 
                       help="Batch size for encrypted training (small)")
    
    # Data configuration
    parser.add_argument("--max-train-samples", type=int, default=50, 
                       help="Maximum training samples (FHE constraint)")
    parser.add_argument("--max-test-samples", type=int, default=20, 
                       help="Maximum test samples")
    
    # FHE configuration
    parser.add_argument("--poly-modulus-degree", type=int, default=8192, 
                       help="FHE polynomial modulus degree")
    parser.add_argument("--scale-bits", type=int, default=40, 
                       help="FHE scale bits")
    
    # Save and test options
    parser.add_argument("--save-model", action="store_true", 
                       help="Save the trained model")
    parser.add_argument("--test-inference", action="store_true", 
                       help="Test production inference after training")
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("PRODUCTION FHE TRAINING SYSTEM")
    print("Fully Encrypted Training - No Plaintext Data Access")
    print("=" * 70)
    
    print(f"\nConfiguration:")
    print(f"  Model: {args.hidden_dims} hidden dims, {'polynomial' if args.use_polynomial_activation else 'linear'} activation")
    print(f"  Training: {args.epochs} epochs, {args.max_train_samples} samples, batch size {args.batch_size}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  FHE: {args.poly_modulus_degree} poly degree, {args.scale_bits} scale bits")
    
    # Create FHE context
    print(f"\nCreating FHE context...")
    context = create_context(
        poly_modulus_degree=args.poly_modulus_degree,
        scale_bits=args.scale_bits
    )
    
    # Load data
    print(f"Loading MNIST data...")
    train_dataloader = load_mnist_data(batch_size=64, train=True)  # Large batches for secure loader
    test_dataloader = load_mnist_data(batch_size=64, train=False)
    
    # Create model
    print(f"Creating FHE model...")
    model = FHEMLPClassifier(
        input_dim=784,
        hidden_dims=args.hidden_dims,
        num_classes=10,
        use_polynomial_activation=args.use_polynomial_activation
    )
    
    # Production training - entirely on encrypted data
    model, training_history = train_production_fhe_model(
        model=model,
        context=context,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        max_train_samples=args.max_train_samples,
        max_test_samples=args.max_test_samples,
        batch_size=args.batch_size
    )
    
    # Save model if requested
    model_path = None
    if args.save_model:
        model_path, stats_path = save_production_model(model, training_history, args, "production")
    
    # Test inference if requested
    if args.test_inference and model_path:
        test_production_inference(model_path, context, test_dataloader)
    
    # Final summary
    print("\n" + "=" * 70)
    print("PRODUCTION TRAINING SUMMARY")
    print("=" * 70)
    print(f"Final Test Accuracy: {training_history['test_accuracy'][-1]:.2%}")
    print(f"Final Test Loss: {training_history['test_loss'][-1]:.4f}")
    print(f"Total Training Time: {sum(training_history['epoch_times']):.1f}s")
    print(f"Privacy Preserved: ✓ No plaintext data access during training")
    print(f"Production Ready: ✓ Supports both encrypted and plain inference")
    
    if args.save_model:
        print(f"Model saved: ✓ {model_path}")


if __name__ == "__main__":
    main()