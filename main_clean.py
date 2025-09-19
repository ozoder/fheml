#!/usr/bin/env python3
"""
Clean architecture main entry point for FHE training.
"""

import argparse
import logging
from datetime import datetime

from src.fheml.core.context import FHEContextManager
from src.fheml.core.encryption import FHEEncryption
from src.fheml.models.classifier import FHEMLPClassifier
from src.fheml.training.checkpoints import CheckpointManager
from src.fheml.training.memory import MemoryManager
from src.fheml.training.trainer import FHETrainer, TrainingConfig
from src.fheml.utils.data import FHEDataset, MNISTDataLoader
from src.fheml.utils.logging import setup_logging


def main() -> None:
    """Main entry point for clean FHE training."""
    # Set up argument parsing
    parser = argparse.ArgumentParser(
        description="Clean Architecture FHE Training System"
    )

    # Model configuration
    parser.add_argument(
        "--hidden-dims",
        type=int,
        nargs="+",
        default=[128, 64],
        help="Hidden layer dimensions",
    )
    parser.add_argument(
        "--use-polynomial-activation",
        action="store_true",
        help="Use polynomial activation instead of linear",
    )

    # Training configuration
    parser.add_argument(
        "--epochs", type=int, default=2, help="Number of training epochs"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=0.01, help="Learning rate"
    )
    parser.add_argument(
        "--batch-size", type=int, default=4, help="Batch size for training"
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=50,
        help="Maximum training samples",
    )
    parser.add_argument(
        "--max-test-samples", type=int, default=10, help="Maximum test samples"
    )

    # System configuration
    parser.add_argument(
        "--memory-limit-gb",
        type=float,
        default=8.0,
        help="Memory limit in GB",
    )
    parser.add_argument(
        "--enable-checkpointing",
        action="store_true",
        help="Enable model checkpointing",
    )

    # FHE configuration
    parser.add_argument(
        "--poly-modulus-degree",
        type=int,
        default=8192,
        help="FHE polynomial modulus degree",
    )
    parser.add_argument(
        "--scale-bits", type=int, default=40, help="FHE scale bits"
    )

    args = parser.parse_args()

    # Set up logging
    setup_logging(level=logging.INFO)
    logger = logging.getLogger(__name__)

    print("=" * 70)
    print("CLEAN ARCHITECTURE FHE TRAINING SYSTEM")
    print("Fully Homomorphic Encryption with Modern Architecture")
    print("=" * 70)

    # Create training configuration
    config = TrainingConfig(
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        max_train_samples=args.max_train_samples,
        max_test_samples=args.max_test_samples,
        memory_limit_gb=args.memory_limit_gb,
        enable_checkpointing=args.enable_checkpointing,
    )

    logger.info(f"Training configuration: {config}")

    # Initialize FHE context manager
    context_manager = FHEContextManager(
        poly_modulus_degree=args.poly_modulus_degree,
        scale_bits=args.scale_bits,
    )

    logger.info("Creating FHE context...")
    context = context_manager.context

    # Create encryption handler
    encryption = FHEEncryption(context)

    # Initialize model
    logger.info("Creating FHE model...")
    model = FHEMLPClassifier(
        input_dim=784,
        hidden_dims=args.hidden_dims,
        num_classes=10,
        use_polynomial_activation=args.use_polynomial_activation,
    )

    logger.info(f"Model architecture: {model.get_architecture_info()}")

    # Set up data loaders
    logger.info("Loading MNIST data...")
    train_loader_plain = MNISTDataLoader(batch_size=32).get_dataloader(train=True)
    test_loader_plain = MNISTDataLoader(batch_size=32).get_dataloader(train=False)

    # Create encrypted datasets
    logger.info("Encrypting training data...")
    train_dataset = FHEDataset(
        encryption=encryption,
        dataloader=train_loader_plain,
        max_samples=args.max_train_samples,
    )

    logger.info("Encrypting test data...")
    test_dataset = FHEDataset(
        encryption=encryption,
        dataloader=test_loader_plain,
        max_samples=args.max_test_samples,
    )

    # Initialize management components
    memory_manager = MemoryManager(memory_limit_gb=args.memory_limit_gb)

    checkpoint_manager = None
    if args.enable_checkpointing:
        checkpoint_manager = CheckpointManager(
            checkpoint_dir=".checkpoints_clean",
            save_best_only=True,
        )

    # Initialize trainer
    trainer = FHETrainer(
        model=model,
        context_manager=context_manager,
        config=config,
        memory_manager=memory_manager,
        checkpoint_manager=checkpoint_manager,
    )

    # Run training
    logger.info("Starting training...")
    start_time = datetime.now()

    try:
        history = trainer.train(train_dataset, test_dataset)

        # Training completed successfully
        end_time = datetime.now()
        total_time = end_time - start_time

        print("\n" + "=" * 70)
        print("TRAINING COMPLETED SUCCESSFULLY")
        print("=" * 70)
        print(f"Total training time: {total_time}")
        print(f"Final test accuracy: {history.test_accuracy[-1]:.2%}")
        print(f"Final test loss: {history.test_loss[-1]:.4f}")

        # Show training progress
        print("\nTraining Progress:")
        for epoch, (acc, loss) in enumerate(
            zip(history.test_accuracy, history.test_loss, strict=False), 1
        ):
            print(f"  Epoch {epoch}: Accuracy={acc:.2%}, Loss={loss:.4f}")

        # Memory usage summary
        if history.memory_usage:
            final_memory = history.memory_usage[-1]
            print(f"\nFinal memory usage: {final_memory['mb']}MB")

        print("\n✓ Clean architecture FHE training completed!")

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        print("\nTraining interrupted by user.")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        print(f"\nTraining failed: {e}")
        raise


if __name__ == "__main__":
    main()
