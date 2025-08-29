# FHE-ML: Fully Homomorphic Encryption for Machine Learning

A comprehensive implementation of privacy-preserving machine learning using Fully Homomorphic Encryption (FHE). This project demonstrates how to train and perform inference on neural networks while keeping data encrypted throughout the entire process.

## Overview

FHE-ML provides a complete framework for:
- Training neural networks on encrypted data using hybrid approaches
- Performing encrypted inference on trained models
- Maintaining data privacy during both training and inference phases
- Working with real datasets (MNIST) under homomorphic encryption constraints

The implementation uses the [TenSEAL](https://github.com/OpenMined/TenSEAL) library for FHE operations and PyTorch for neural network components.

## Key Features

### 🔐 **Privacy-Preserving Operations**
- Complete data encryption during training and inference
- No plaintext data exposure during computation
- Secure model parameter updates through encrypted gradients

### 🧠 **FHE-Optimized Neural Networks**
- Custom MLP implementation designed for homomorphic encryption
- FHE-friendly polynomial activation functions
- Linear activations for minimal multiplicative depth
- Configurable architecture with multiple hidden layers

### 🚀 **Hybrid Training Approach**
- Combines encrypted and plaintext training for optimal performance
- Proxy model synchronization for gradient computation
- Flexible training strategies (encrypted-only, plaintext-only, or mixed)

### 📊 **Comprehensive Evaluation**
- Encrypted inference with confidence scoring
- Model accuracy evaluation on encrypted test sets
- Confusion matrix generation for detailed analysis

## Architecture

### Core Components

1. **`model.py`** - Neural network implementations
   - `FHEMLPClassifier`: Main FHE-compatible neural network
   - `FHELinearLayer`: Custom linear layers for encrypted computation
   - `FHEPolynomialActivation`: Polynomial approximation of ReLU
   - `TorchMLPClassifier`: Standard PyTorch model for proxy training

2. **`training.py`** - Training algorithms
   - `FHETrainer`: Base trainer for FHE models
   - `HybridFHETrainer`: Advanced trainer supporting mixed encrypted/plaintext batches
   - Gradient approximation techniques for encrypted data

3. **`inference.py`** - Encrypted inference engine
   - `FHEInference`: Core inference functionality
   - `SecureInferenceServer`: Production-ready inference server
   - Batch processing and confidence estimation

4. **`utils.py`** - Utility functions
   - FHE context creation and management
   - Data encryption/decryption operations
   - MNIST data loading and preprocessing
   - `FHEDataset`: Custom dataset class for encrypted data

5. **`main.py`** - Training orchestration
   - Complete training pipeline with progress tracking
   - Flexible command-line interface
   - Model checkpointing and evaluation

## Installation

### Prerequisites
- Python 3.10 (required for TenSEAL compatibility)
- PDM package manager

### Setup
```bash
# Clone the repository
git clone <repository-url>
cd fheml

# Install dependencies using PDM
pdm install

# Activate the virtual environment
pdm shell
```

### Dependencies
- **TenSEAL** (≥0.3.16): Homomorphic encryption library
- **PyTorch** (≥2.8.0): Deep learning framework
- **TorchVision** (≥0.23.0): Computer vision utilities
- **NumPy** (≥2.2.6): Numerical computing
- **Scikit-learn** (≥1.7.1): Machine learning utilities
- **Matplotlib** (≥3.10.5): Plotting and visualization
- **tqdm** (≥4.67.1): Progress bars

## Usage

### Quick Start

#### 1. Basic Training
```bash
python main.py --epochs 3 --hidden-dims 64 --learning-rate 0.01
```

#### 2. Encrypted Training
```bash
python main.py --use-encrypted --encrypted-epochs 2 --encrypted-samples 100 --save-model
```

#### 3. Full Pipeline with Encrypted Inference
```bash
python main.py --use-encrypted --encrypted-samples 200 --test-encrypted-inference --save-model
```

### Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--epochs` | Number of training epochs | 5 |
| `--batch-size` | Batch size for plaintext training | 64 |
| `--encrypted-batch-size` | Batch size for encrypted training | 4 |
| `--learning-rate` | Learning rate | 0.01 |
| `--hidden-dims` | Hidden layer dimensions | [64] |
| `--use-encrypted` | Enable encrypted training | False |
| `--encrypted-epochs` | Epochs of encrypted training | 1 |
| `--encrypted-samples` | Number of encrypted training samples | 100 |
| `--encrypted-test-samples` | Number of encrypted test samples | 20 |
| `--poly-modulus-degree` | FHE polynomial modulus degree | 8192 |
| `--scale-bits` | FHE scale bits | 40 |
| `--save-model` | Save trained model | False |
| `--test-encrypted-inference` | Test encrypted inference | False |

### Python API Usage

#### Basic Model Creation and Training
```python
from model import FHEMLPClassifier
from training import HybridFHETrainer
from utils import create_context, load_mnist_data

# Create FHE context
context = create_context(poly_modulus_degree=8192, scale_bits=40)

# Create model
model = FHEMLPClassifier(
    input_dim=784, 
    hidden_dims=[64], 
    num_classes=10,
    use_polynomial_activation=False
)

# Setup training
trainer = HybridFHETrainer(model, learning_rate=0.01)
train_loader = load_mnist_data(batch_size=64, train=True)

# Train on plaintext data
for images, labels in train_loader:
    loss = trainer.train_on_plain_batch(images, labels)
```

#### Encrypted Inference
```python
from inference import FHEInference
from utils import encrypt_tensor

# Setup inference engine
inference_engine = FHEInference(model, context)

# Encrypt input data
sample_image = torch.randn(784)
encrypted_input = encrypt_tensor(context, sample_image)

# Perform encrypted inference
prediction = inference_engine.predict_encrypted(encrypted_input)
prediction_with_confidence = inference_engine.predict_with_confidence(encrypted_input)
```

## Technical Details

### FHE Implementation

The project uses the CKKS scheme from TenSEAL, which supports approximate arithmetic on encrypted real numbers. Key technical considerations:

1. **Multiplicative Depth**: Limited by the encryption parameters
   - Polynomial modulus degree: 8192 (default)
   - Coefficient modulus: [60, 40, 40, 60] bits
   - Scale: 2^40

2. **Activation Functions**: 
   - Linear activation (identity) for minimal depth
   - Polynomial approximation: `f(x) ≈ 0.5x + 0.25x²` for ReLU-like behavior

3. **Noise Management**:
   - Automatic rescaling after multiplications
   - Bootstrap operations when needed
   - Careful parameter selection for noise budget

### Training Strategy

The hybrid training approach combines:
1. **Plaintext Training**: Fast convergence on unencrypted data
2. **Encrypted Training**: Privacy-preserving fine-tuning
3. **Proxy Model**: Standard PyTorch model for gradient computation
4. **Weight Synchronization**: Keeps FHE and proxy models aligned

### Performance Considerations

- **Encrypted operations are ~1000x slower** than plaintext
- **Small batch sizes** recommended for encrypted training (2-8 samples)
- **Limited model complexity** due to multiplicative depth constraints
- **Preprocessing** data encryption can be done offline

## Testing

Run the test suite to verify functionality:

```bash
# Basic functionality test
python tests/test_basic.py

# Full test suite
python -m pytest tests/
```

The basic test covers:
- FHE context creation
- Model instantiation
- Encryption/decryption operations
- Forward pass on encrypted data
- MNIST data loading
- End-to-end inference

## Project Structure

```
fheml/
├── main.py              # Main training script
├── model.py             # Neural network implementations
├── training.py          # Training algorithms
├── inference.py         # Inference engines
├── utils.py             # Utility functions
├── tests/               # Test suite
│   ├── test_basic.py    # Basic functionality tests
│   ├── test_inference.py # Inference tests
│   ├── test_model.py    # Model tests
│   ├── test_training.py # Training tests
│   └── test_utils.py    # Utility tests
├── data/                # Dataset storage
│   └── MNIST/           # MNIST dataset
├── pyproject.toml       # Project configuration
├── pdm.lock            # Dependency lock file
└── README.md           # This file
```

## Limitations and Future Work

### Current Limitations
1. **Performance**: Encrypted operations are computationally expensive
2. **Model Size**: Limited by FHE multiplicative depth constraints
3. **Batch Size**: Small encrypted batch sizes for practical training times
4. **Activation Functions**: Restricted to low-degree polynomials

### Future Enhancements
- Support for convolutional layers
- Advanced bootstrapping techniques
- Distributed encrypted training
- Integration with federated learning
- Support for other FHE schemes (BFV, BGV)

## Contributing

Contributions are welcome! Please ensure:
1. All tests pass: `python -m pytest tests/`
2. Code follows the existing style
3. New features include appropriate tests
4. Security considerations are documented

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## References

1. **TenSEAL**: A Library for Encrypted Tensor Operations Using Homomorphic Encryption
2. **CKKS Scheme**: Homomorphic Encryption for Approximate Arithmetic
3. **Privacy-Preserving Machine Learning**: Techniques and Applications

## Security Notice

This implementation is for **research and educational purposes**. For production use:
- Conduct thorough security audits
- Use appropriate key management
- Consider side-channel attack mitigation
- Validate against specific threat models
