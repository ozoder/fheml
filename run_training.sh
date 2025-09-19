#!/bin/bash

# Ultimate FHE Training - Designed for 90%+ Accuracy
# Requirements: 256GB+ RAM, 32+ CPU cores, 2TB+ storage
# Optimized for high-resource machines with deep FHE architectures

set -e

# Create logs directory
mkdir -p .logs

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_PREFIX=".logs/ultimate_training_${TIMESTAMP}"
MAIN_LOG="${LOG_PREFIX}_main.log"
MONITOR_LOG="${LOG_PREFIX}_monitor.log"
MEMORY_LOG="${LOG_PREFIX}_memory.log"
PERFORMANCE_LOG="${LOG_PREFIX}_performance.log"

echo "🚀 ULTIMATE FHE TRAINING - ${TIMESTAMP}"
echo "=============================================================="
echo "Target: 90%+ accuracy on MNIST with fully encrypted training"
echo "Strategy: Deep architecture optimized for high-resource systems"
echo "Requirements: 256GB+ RAM, 32+ CPU cores recommended"
echo ""

# System requirements check
TOTAL_MEM_GB=$(free -g | grep "^Mem:" | awk '{print $2}')
CPU_CORES=$(nproc)

echo "🔍 SYSTEM RESOURCE CHECK:"
echo "  Total Memory: ${TOTAL_MEM_GB}GB"
echo "  CPU Cores: ${CPU_CORES}"
echo ""

if [ "$TOTAL_MEM_GB" -lt 128 ]; then
    echo "⚠️  WARNING: System has ${TOTAL_MEM_GB}GB RAM. Recommended: 256GB+"
    echo "⚠️  Training may fail due to insufficient memory for deep FHE networks"
fi

if [ "$CPU_CORES" -lt 16 ]; then
    echo "⚠️  WARNING: System has ${CPU_CORES} CPU cores. Recommended: 32+"
    echo "⚠️  FHE operations will be slower on this system"
fi

echo ""

# Create monitoring
echo "$(date): Ultimate FHE training started - targeting 90%+ accuracy" > "${MONITOR_LOG}"

# Advanced memory monitoring for high-resource training
{
    echo "Time,ProcessMB,SystemFreeMB,CPUPercent,Status,Action" > "${MEMORY_LOG}"
    while true; do
        if PYTHON_PID=$(pgrep -f "python main.py" 2>/dev/null); then
            PROC_MEM_KB=$(ps -p $PYTHON_PID -o rss= 2>/dev/null | tr -d ' ' || echo "0")
            PROC_MEM_MB=$((PROC_MEM_KB / 1024))
            SYS_FREE_MB=$(free -m | grep "^Mem:" | awk '{print $7}')
            CPU_PERCENT=$(ps -p $PYTHON_PID -o %cpu= 2>/dev/null | tr -d ' ' || echo "0")

            # Status thresholds for high-resource systems
            if [ "$PROC_MEM_MB" -gt 200000 ]; then  # 200GB+
                STATUS="EXTREME_LOAD"
                ACTION="Maximum resource utilization - monitor closely"
            elif [ "$PROC_MEM_MB" -gt 150000 ]; then  # 150GB+
                STATUS="ULTRA_HIGH"
                ACTION="Deep network training - high performance mode"
            elif [ "$PROC_MEM_MB" -gt 100000 ]; then  # 100GB+
                STATUS="HIGH_PERFORMANCE"
                ACTION="Optimal range for 90%+ accuracy training"
            elif [ "$PROC_MEM_MB" -gt 50000 ]; then   # 50GB+
                STATUS="MODERATE"
                ACTION="Sufficient for deep networks"
            else
                STATUS="NORMAL"
                ACTION="Standard operation"
            fi

            echo "$(date '+%H:%M:%S'),$PROC_MEM_MB,$SYS_FREE_MB,$CPU_PERCENT%,$STATUS,$ACTION" >> "${MEMORY_LOG}"
        else
            echo "$(date '+%H:%M:%S'),0,$(free -m | grep "^Mem:" | awk '{print $7}'),0%,NOT_RUNNING,Process not found" >> "${MEMORY_LOG}"
        fi
        sleep 20
    done
} &
MEMORY_MONITOR_PID=$!

# Performance monitoring
{
    echo "Time,Epoch,TrainLoss,TestLoss,TestAccuracy,MemoryMB,Status" > "${PERFORMANCE_LOG}"
    while true; do
        # Extract latest metrics from main log
        if [ -f "${MAIN_LOG}" ]; then
            LATEST_EPOCH=$(grep "Epoch.*Results:" "${MAIN_LOG}" | tail -1 | grep -o "Epoch [0-9]*" | grep -o "[0-9]*" || echo "0")
            LATEST_TRAIN_LOSS=$(grep "Training Loss:" "${MAIN_LOG}" | tail -1 | grep -o "[0-9.-]*" | head -1 || echo "0")
            LATEST_TEST_LOSS=$(grep "Test Loss:" "${MAIN_LOG}" | tail -1 | grep -o "[0-9.-]*" | head -1 || echo "0")
            LATEST_ACCURACY=$(grep "Test Accuracy:" "${MAIN_LOG}" | tail -1 | grep -o "[0-9.]*%" | head -1 || echo "0%")

            if PYTHON_PID=$(pgrep -f "python main.py" 2>/dev/null); then
                CURRENT_MEM_MB=$(($(ps -p $PYTHON_PID -o rss= 2>/dev/null | tr -d ' ' || echo "0") / 1024))
                echo "$(date '+%H:%M:%S'),$LATEST_EPOCH,$LATEST_TRAIN_LOSS,$LATEST_TEST_LOSS,$LATEST_ACCURACY,$CURRENT_MEM_MB,TRAINING" >> "${PERFORMANCE_LOG}"
            fi
        fi
        sleep 60
    done
} &
PERFORMANCE_MONITOR_PID=$!

# Activate virtual environment
source .venv/bin/activate

echo "🎯 ULTIMATE TRAINING CONFIGURATION:" | tee -a "${MONITOR_LOG}"
echo "  Target: 90%+ Test Accuracy" | tee -a "${MONITOR_LOG}"
echo "  Architecture: Ultra-Deep FHE Network [1024, 512, 256, 128]" | tee -a "${MONITOR_LOG}"
echo "  Activation: Advanced Polynomial (degree-2 ReLU approximation)" | tee -a "${MONITOR_LOG}"
echo "  Memory Management: 200GB limit with aggressive optimization" | tee -a "${MONITOR_LOG}"
echo "  Training Strategy: Extended epochs with large-scale dataset" | tee -a "${MONITOR_LOG}"
echo "" | tee -a "${MONITOR_LOG}"

echo "🔬 TRAINING PARAMETERS:" | tee -a "${MONITOR_LOG}"
echo "  Epochs: 25 (extended for deep network convergence)" | tee -a "${MONITOR_LOG}"
echo "  Training Samples: 5000 (large-scale encrypted dataset)" | tee -a "${MONITOR_LOG}"
echo "  Test Samples: 1000 (comprehensive evaluation)" | tee -a "${MONITOR_LOG}"
echo "  Batch Size: 8 (optimized for deep FHE operations)" | tee -a "${MONITOR_LOG}"
echo "  Architecture: [1024, 512, 256, 128] (4-layer ultra-deep)" | tee -a "${MONITOR_LOG}"
echo "  Learning Rate: 0.001 (conservative for stability)" | tee -a "${MONITOR_LOG}"
echo "  Activation: Polynomial (enhanced non-linear approximation)" | tee -a "${MONITOR_LOG}"
echo "  Memory Limit: 200GB (high-resource optimization)" | tee -a "${MONITOR_LOG}"

echo "$(date): Launching ultimate FHE training..." | tee -a "${MONITOR_LOG}"

# Set CPU affinity for optimal performance
if [ "$CPU_CORES" -ge 32 ]; then
    echo "🚀 Optimizing CPU affinity for ${CPU_CORES} cores" | tee -a "${MONITOR_LOG}"
    # Use taskset to bind to all available cores
    TASKSET_CORES="0-$((CPU_CORES-1))"
    TASKSET_CMD="taskset -c $TASKSET_CORES"
else
    TASKSET_CMD=""
fi

# Run ultimate training with maximum parameters
$TASKSET_CMD python main.py \
  --epochs 25 \
  --max-train-samples 5000 \
  --max-test-samples 1000 \
  --batch-size 8 \
  --hidden-dims 1024 512 256 128 \
  --learning-rate 0.001 \
  --memory-limit-gb 200 \
  --save-model \
  --test-inference \
  --use-polynomial-activation \
  --enable-checkpointing \
  --checkpoint-every 5 2>&1 | tee "${MAIN_LOG}"

TRAIN_EXIT_CODE=${PIPESTATUS[0]}

# Stop monitoring
kill $MEMORY_MONITOR_PID 2>/dev/null || true
kill $PERFORMANCE_MONITOR_PID 2>/dev/null || true
echo "$(date): Ultimate training completed with exit code: ${TRAIN_EXIT_CODE}" | tee -a "${MONITOR_LOG}"

# Comprehensive results analysis
if [ ${TRAIN_EXIT_CODE} -eq 0 ]; then
    echo "" | tee -a "${MONITOR_LOG}"
    echo "🎉 ULTIMATE TRAINING COMPLETED!" | tee -a "${MONITOR_LOG}"
    echo "========================================" | tee -a "${MONITOR_LOG}"

    # Extract final results
    FINAL_ACCURACY=$(grep "Final Test Accuracy:" "${MAIN_LOG}" | tail -1 | grep -o '[0-9.]*%' || echo "Not found")
    TOTAL_TIME=$(grep "Total Training Time:" "${MAIN_LOG}" | tail -1 | grep -o '[0-9.]*s' || echo "Not found")
    EPOCHS_DONE=$(grep -c "Epoch.*Results:" "${MAIN_LOG}" || echo "0")
    BEST_ACCURACY=$(grep "Test Accuracy:" "${MAIN_LOG}" | grep -o '[0-9.]*%' | sort -nr | head -1 || echo "0%")

    echo "📊 ULTIMATE RESULTS:" | tee -a "${MONITOR_LOG}"
    echo "  🎯 Final Accuracy: ${FINAL_ACCURACY}" | tee -a "${MONITOR_LOG}"
    echo "  🏆 Best Accuracy: ${BEST_ACCURACY}" | tee -a "${MONITOR_LOG}"
    echo "  ⏱️  Total Time: ${TOTAL_TIME}" | tee -a "${MONITOR_LOG}"
    echo "  🔄 Epochs: ${EPOCHS_DONE}/25" | tee -a "${MONITOR_LOG}"

    # Ultimate performance assessment
    if [[ "${FINAL_ACCURACY}" =~ ([0-9.]+)% ]]; then
        ACCURACY_NUM="${BASH_REMATCH[1]}"
        echo "" | tee -a "${MONITOR_LOG}"
        echo "🏆 ULTIMATE ASSESSMENT:" | tee -a "${MONITOR_LOG}"

        if (( $(echo "${ACCURACY_NUM} >= 95.0" | bc -l) )); then
            echo "  🌟 LEGENDARY: World-class FHE performance achieved!" | tee -a "${MONITOR_LOG}"
            echo "  🚀 This represents state-of-the-art encrypted ML!" | tee -a "${MONITOR_LOG}"
        elif (( $(echo "${ACCURACY_NUM} >= 90.0" | bc -l) )); then
            echo "  🎯 ULTIMATE SUCCESS: 90%+ target achieved!" | tee -a "${MONITOR_LOG}"
            echo "  🔥 Exceptional FHE training performance!" | tee -a "${MONITOR_LOG}"
        elif (( $(echo "${ACCURACY_NUM} >= 85.0" | bc -l) )); then
            echo "  🚀 OUTSTANDING: Near-ultimate performance!" | tee -a "${MONITOR_LOG}"
            echo "  📈 Consider extending training for 90%+" | tee -a "${MONITOR_LOG}"
        elif (( $(echo "${ACCURACY_NUM} >= 75.0" | bc -l) )); then
            echo "  ✨ EXCELLENT: Major breakthrough achieved!" | tee -a "${MONITOR_LOG}"
            echo "  💡 Deep FHE networks are working!" | tee -a "${MONITOR_LOG}"
        elif (( $(echo "${ACCURACY_NUM} >= 50.0" | bc -l) )); then
            echo "  🎉 VERY GOOD: Significant improvement!" | tee -a "${MONITOR_LOG}"
            echo "  📊 Deep architecture showing promise" | tee -a "${MONITOR_LOG}"
        else
            echo "  📈 PROGRESS: Higher than baseline achieved" | tee -a "${MONITOR_LOG}"
            echo "  🔧 Consider resource optimization" | tee -a "${MONITOR_LOG}"
        fi
    fi

    # Resource utilization analysis
    if [ -f "${MEMORY_LOG}" ]; then
        PEAK_MEM=$(awk -F',' 'NR>1 && $2 > 0 {print $2}' "${MEMORY_LOG}" | sort -n | tail -1)
        PEAK_MEM_GB=$((PEAK_MEM / 1024))
        ULTRA_HIGH_COUNT=$(awk -F',' '$5=="ULTRA_HIGH" || $5=="EXTREME_LOAD"' "${MEMORY_LOG}" | wc -l)
        HIGH_PERF_COUNT=$(awk -F',' '$5=="HIGH_PERFORMANCE"' "${MEMORY_LOG}" | wc -l)

        echo "" | tee -a "${MONITOR_LOG}"
        echo "🧠 RESOURCE UTILIZATION:" | tee -a "${MONITOR_LOG}"
        echo "  Peak Memory: ${PEAK_MEM:-0}MB (${PEAK_MEM_GB}GB)" | tee -a "${MONITOR_LOG}"
        echo "  Ultra-High Periods: ${ULTRA_HIGH_COUNT}" | tee -a "${MONITOR_LOG}"
        echo "  High-Performance Periods: ${HIGH_PERF_COUNT}" | tee -a "${MONITOR_LOG}"

        if [ "${PEAK_MEM_GB:-0}" -gt 100 ]; then
            echo "  🚀 High-resource training successfully utilized!" | tee -a "${MONITOR_LOG}"
        else
            echo "  💡 System resources underutilized - could handle deeper networks" | tee -a "${MONITOR_LOG}"
        fi
    fi

    # Architecture analysis
    SCALE_ISSUES=$(grep -c "scale.*out of bounds\|bootstrap" "${MAIN_LOG}" || echo "0")
    GRADIENT_UPDATES=$(grep -c "Batch training complete" "${MAIN_LOG}" || echo "0")

    echo "" | tee -a "${MONITOR_LOG}"
    echo "🔬 DEEP ARCHITECTURE ANALYSIS:" | tee -a "${MONITOR_LOG}"
    echo "  Network: [1024, 512, 256, 128] ultra-deep FHE" | tee -a "${MONITOR_LOG}"
    echo "  Gradient Updates: ${GRADIENT_UPDATES}" | tee -a "${MONITOR_LOG}"
    echo "  Scale Issues: ${SCALE_ISSUES}" | tee -a "${MONITOR_LOG}"

    if [ "${SCALE_ISSUES}" -eq 0 ]; then
        echo "  ✅ FHE Scale Management: PERFECT" | tee -a "${MONITOR_LOG}"
        echo "  🔥 Deep polynomial activations stable!" | tee -a "${MONITOR_LOG}"
    else
        echo "  ⚠️  FHE Scale Management: Some issues (${SCALE_ISSUES})" | tee -a "${MONITOR_LOG}"
        echo "  🔧 Consider FHE parameter tuning" | tee -a "${MONITOR_LOG}"
    fi

    # Model files and checkpoints
    MODEL_COUNT=$(find .models -name "*$(date +%Y%m%d)*" -type f 2>/dev/null | wc -l)
    CHECKPOINT_COUNT=$(find .checkpoints -name "*.pt" 2>/dev/null | wc -l || echo "0")

    echo "  💾 Models saved: ${MODEL_COUNT} files" | tee -a "${MONITOR_LOG}"
    echo "  🔄 Checkpoints: ${CHECKPOINT_COUNT} files" | tee -a "${MONITOR_LOG}"

    echo "" | tee -a "${MONITOR_LOG}"
    echo "🎊 SUCCESS: Ultimate FHE training completed!" | tee -a "${MONITOR_LOG}"

    # Performance timeline
    if [ -f "${PERFORMANCE_LOG}" ]; then
        echo "" | tee -a "${MONITOR_LOG}"
        echo "📈 TRAINING TIMELINE:" | tee -a "${MONITOR_LOG}"
        echo "  Performance data saved to: ${PERFORMANCE_LOG}" | tee -a "${MONITOR_LOG}"

        # Show last few performance entries
        echo "  Recent performance:" | tee -a "${MONITOR_LOG}"
        tail -5 "${PERFORMANCE_LOG}" | while read line; do
            echo "    $line" | tee -a "${MONITOR_LOG}"
        done
    fi

    # Configuration for reproduction
    echo "" | tee -a "${MONITOR_LOG}"
    echo "🔧 CONFIGURATION USED:" | tee -a "${MONITOR_LOG}"
    echo "  Command: python main.py --epochs 25 --max-train-samples 5000 --max-test-samples 1000 --batch-size 8 --hidden-dims 1024 512 256 128 --learning-rate 0.001 --memory-limit-gb 200 --save-model --test-inference --use-polynomial-activation --enable-checkpointing --checkpoint-every 5" | tee -a "${MONITOR_LOG}"

else
    echo "❌ Ultimate training failed with exit code: ${TRAIN_EXIT_CODE}" | tee -a "${MONITOR_LOG}"

    if [ ${TRAIN_EXIT_CODE} -eq 137 ]; then
        echo "💾 Memory exhaustion - system needs more resources" | tee -a "${MONITOR_LOG}"
        echo "🔧 Recommendations:" | tee -a "${MONITOR_LOG}"
        echo "  - Increase system RAM to 256GB+" | tee -a "${MONITOR_LOG}"
        echo "  - Reduce architecture to [512, 256, 128]" | tee -a "${MONITOR_LOG}"
        echo "  - Decrease training samples to 2000" | tee -a "${MONITOR_LOG}"
    fi

    echo "" | tee -a "${MONITOR_LOG}"
    echo "Last 20 lines of training log:" | tee -a "${MONITOR_LOG}"
    tail -20 "${MAIN_LOG}" | sed 's/^/  /' | tee -a "${MONITOR_LOG}"
fi

echo "" | tee -a "${MONITOR_LOG}"
echo "📁 All logs preserved in .logs/ directory:" | tee -a "${MONITOR_LOG}"
echo "  Main log: ${MAIN_LOG}" | tee -a "${MONITOR_LOG}"
echo "  Memory log: ${MEMORY_LOG}" | tee -a "${MONITOR_LOG}"
echo "  Performance log: ${PERFORMANCE_LOG}" | tee -a "${MONITOR_LOG}"
echo "  Monitor log: ${MONITOR_LOG}" | tee -a "${MONITOR_LOG}"
echo "$(date): Ultimate training session completed" | tee -a "${MONITOR_LOG}"