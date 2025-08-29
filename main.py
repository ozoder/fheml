import argparse
import time

import torch
from tqdm import tqdm

from inference import FHEInference
from model import FHEMLPClassifier
from training import HybridFHETrainer
from utils import FHEDataset, create_context, load_mnist_data


def train_fhe_model(args):
    print("=" * 50)
    print("FHE-ML Training on MNIST")
    print("=" * 50)

    print("\n[1/6] Creating FHE context...")
    context = create_context(
        poly_modulus_degree=args.poly_modulus_degree, scale_bits=args.scale_bits
    )

    print("\n[2/6] Loading MNIST dataset...")
    train_loader = load_mnist_data(batch_size=args.batch_size, train=True)
    test_loader = load_mnist_data(batch_size=args.batch_size, train=False)

    print("\n[3/6] Creating FHE model...")
    model = FHEMLPClassifier(
        input_dim=784, hidden_dims=args.hidden_dims, num_classes=10, use_polynomial_activation=False  # Use linear for FHE stability
    )

    print("\n[4/6] Initializing trainer...")
    trainer = HybridFHETrainer(
        model=model, learning_rate=args.learning_rate, device=args.device
    )

    print("\n[5/6] Preparing encrypted dataset (this may take a while)...")
    if args.use_encrypted:
        print(f"Encrypting {args.encrypted_samples} training samples...")
        encrypted_train_dataset = FHEDataset(
            context, train_loader, max_samples=args.encrypted_samples
        )

        print(f"Encrypting {args.encrypted_test_samples} test samples...")
        encrypted_test_dataset = FHEDataset(
            context, test_loader, max_samples=args.encrypted_test_samples
        )

    print("\n[6/6] Starting training...")
    print("-" * 50)

    training_history = {"train_loss": [], "test_accuracy": [], "epoch_times": []}

    for epoch in range(args.epochs):
        epoch_start = time.time()
        epoch_loss = 0.0
        num_batches = 0

        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        if args.use_encrypted and epoch < args.encrypted_epochs:
            print("Training on encrypted data...")

            batch_size = min(args.encrypted_batch_size, len(encrypted_train_dataset))
            num_batches_encrypted = len(encrypted_train_dataset) // batch_size

            pbar = tqdm(range(num_batches_encrypted), desc="Encrypted batches")
            for i in pbar:
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(encrypted_train_dataset))

                batch_images = []
                batch_labels = []

                for j in range(start_idx, end_idx):
                    enc_img, label = encrypted_train_dataset[j]
                    batch_images.append(enc_img)
                    batch_labels.append(label)

                batch_labels = torch.tensor(batch_labels)

                loss = trainer.train_on_encrypted_batch(batch_images, batch_labels)
                epoch_loss += loss
                num_batches += 1

                pbar.set_postfix({"loss": f"{loss:.4f}"})

        else:
            print("Training on plaintext data...")

            pbar = tqdm(train_loader, desc="Plain batches")
            for images, labels in pbar:
                loss = trainer.train_on_plain_batch(images, labels)
                epoch_loss += loss
                num_batches += 1

                pbar.set_postfix({"loss": f"{loss:.4f}"})

        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0

        print("Evaluating on test set...")
        test_accuracy, test_loss = trainer.evaluate_plain(test_loader)

        epoch_time = time.time() - epoch_start

        training_history["train_loss"].append(avg_loss)
        training_history["test_accuracy"].append(test_accuracy)
        training_history["epoch_times"].append(epoch_time)

        print(f"Epoch {epoch + 1} Summary:")
        print(f"  Average Loss: {avg_loss:.4f}")
        print(f"  Test Accuracy: {test_accuracy:.2%}")
        print(f"  Time: {epoch_time:.2f}s")

    print("\n" + "=" * 50)
    print("Training Complete!")
    print("=" * 50)

    if args.save_model:
        save_path = f"fhe_mnist_model_{time.strftime('%Y%m%d_%H%M%S')}.pt"
        print(f"\nSaving model to {save_path}...")

        checkpoint = {
            "model_state": model.get_parameters(),
            "model_config": {
                "input_dim": 784,
                "hidden_dims": args.hidden_dims,
                "num_classes": 10,
                "use_polynomial_activation": False,  # Save activation type
            },
            "training_history": training_history,
            "args": vars(args),
        }

        torch.save(checkpoint, save_path)
        print("Model saved successfully!")

    if args.test_encrypted_inference:
        print("\n" + "=" * 50)
        print("Testing Encrypted Inference")
        print("=" * 50)

        inference_engine = FHEInference(model, context)

        print(f"\nTesting on {args.encrypted_test_samples} encrypted samples...")
        test_labels = encrypted_test_dataset.labels[
            : args.encrypted_test_samples
        ].tolist()
        test_images = [
            encrypted_test_dataset[i][0]
            for i in range(
                min(args.encrypted_test_samples, len(encrypted_test_dataset))
            )
        ]

        accuracy = inference_engine.evaluate_accuracy(
            test_images, test_labels, show_progress=True
        )

        print(f"Encrypted Inference Accuracy: {accuracy:.2%}")

    return model, training_history


def main():
    parser = argparse.ArgumentParser(description="Train FHE-ML model on MNIST")

    parser.add_argument(
        "--epochs", type=int, default=5, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Batch size for plain training"
    )
    parser.add_argument(
        "--encrypted-batch-size",
        type=int,
        default=4,
        help="Batch size for encrypted training",
    )
    parser.add_argument(
        "--learning-rate", type=float, default=0.01, help="Learning rate"
    )
    parser.add_argument(
        "--hidden-dims",
        type=int,
        nargs="+",
        default=[64],  # Reduced for FHE compatibility
        help="Hidden layer dimensions",
    )

    parser.add_argument(
        "--use-encrypted", action="store_true", help="Use encrypted training"
    )
    parser.add_argument(
        "--encrypted-epochs",
        type=int,
        default=1,
        help="Number of epochs to train on encrypted data",
    )
    parser.add_argument(
        "--encrypted-samples",
        type=int,
        default=100,
        help="Number of encrypted training samples",
    )
    parser.add_argument(
        "--encrypted-test-samples",
        type=int,
        default=20,
        help="Number of encrypted test samples",
    )

    parser.add_argument(
        "--poly-modulus-degree",
        type=int,
        default=8192,
        help="Polynomial modulus degree for FHE",
    )
    parser.add_argument("--scale-bits", type=int, default=40, help="Scale bits for FHE")

    parser.add_argument(
        "--device", type=str, default="cpu", help="Device to use (cpu/cuda)"
    )
    parser.add_argument("--save-model", action="store_true", help="Save trained model")
    parser.add_argument(
        "--test-encrypted-inference",
        action="store_true",
        help="Test encrypted inference after training",
    )

    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"

    model, history = train_fhe_model(args)

    print("\n" + "=" * 50)
    print("Summary Statistics:")
    print("=" * 50)
    print(f"Final Training Loss: {history['train_loss'][-1]:.4f}")
    print(f"Final Test Accuracy: {history['test_accuracy'][-1]:.2%}")
    print(f"Total Training Time: {sum(history['epoch_times']):.2f}s")
    print(
        f"Average Epoch Time: {sum(history['epoch_times'])/len(history['epoch_times']):.2f}s"
    )


if __name__ == "__main__":
    main()
