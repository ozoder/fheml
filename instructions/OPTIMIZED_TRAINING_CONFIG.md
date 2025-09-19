# Optimized MNIST Training Configuration - Memory Efficient

## Analysis of Previous Failure
- **Issue**: Exit code 137 (process killed - memory exhaustion)
- **Cause**: 1000 training samples too large for FHE memory constraints
- **Solution**: Reduce parameters while maintaining good performance

## New Optimized Parameters

### Model Architecture (Unchanged - Good Balance)
- **Hidden Dimensions**: [64, 32] - Two hidden layers
- **Activation**: Polynomial activation for better FHE performance

### Training Configuration (Reduced for Memory)
- **Epochs**: 5 - Still sufficient for convergence
- **Learning Rate**: 0.001 - Slightly higher for faster convergence
- **Batch Size**: 4 - Reduced for memory efficiency
- **Training Samples**: 200 - Much more manageable for FHE
- **Test Samples**: 50 - Good evaluation set

### Expected Performance
- **Target Accuracy**: 60-75% (still excellent for FHE)
- **Training Time**: 15-30 minutes
- **Memory Usage**: 2-4GB (much more reasonable)

## Optimized Command
```bash
python main.py \
  --epochs 5 \
  --max-train-samples 200 \
  --max-test-samples 50 \
  --batch-size 4 \
  --hidden-dims 64 32 \
  --learning-rate 0.001 \
  --use-polynomial-activation \
  --save-model \
  --test-inference
```

This configuration balances performance with FHE computational constraints.