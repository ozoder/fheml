# Full MNIST Training Configuration - Optimized for Performance

## Training Parameters (Optimized for Excellence)

### Model Architecture
- **Hidden Dimensions**: [64, 32] - Two hidden layers for better representation
- **Activation**: Polynomial activation for better FHE performance
- **Input**: 784 (MNIST 28x28 flattened)
- **Output**: 10 classes

### Training Configuration
- **Epochs**: 8 - Sufficient for convergence without overfitting
- **Learning Rate**: 0.0005 - Lower rate for stable convergence with FHE
- **Batch Size**: 8 - Balanced between efficiency and FHE constraints
- **Training Samples**: 1000 - Substantial dataset for good performance
- **Test Samples**: 200 - Good evaluation set

### FHE Configuration
- **Polynomial Modulus Degree**: 8192 - Standard for good security/performance
- **Scale Bits**: 40 - Adequate precision for training

### Expected Performance
- **Target Accuracy**: 70-85%
- **Training Time**: 45-90 minutes
- **Memory Usage**: 4-8GB
- **Log Files**: Continuous timestamped logging

## Command
```bash
python main.py \
  --epochs 8 \
  --max-train-samples 1000 \
  --max-test-samples 200 \
  --batch-size 8 \
  --hidden-dims 64 32 \
  --learning-rate 0.0005 \
  --use-polynomial-activation \
  --save-model \
  --test-inference
```

## Success Criteria
- Training completes all epochs without errors
- Final accuracy > 70%
- All logs preserved
- Model saved successfully
- Inference testing passes