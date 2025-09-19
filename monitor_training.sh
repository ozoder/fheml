#!/bin/bash

# Monitor the full training run in real-time

echo "🔍 FHE Training Monitor - $(date)"
echo "Monitoring training progress..."
echo ""

while true; do
    # Find the most recent log files
    MAIN_LOG=$(ls -t full_mnist_training_*_main.log 2>/dev/null | head -1)
    MONITOR_LOG=$(ls -t full_mnist_training_*_monitor.log 2>/dev/null | head -1)

    if [ -n "$MAIN_LOG" ] && [ -f "$MAIN_LOG" ]; then
        echo "📊 Training Progress Summary - $(date)"
        echo "=================================="

        # Show current status from monitor log
        if [ -f "$MONITOR_LOG" ]; then
            echo "🏃 Status Updates:"
            tail -5 "$MONITOR_LOG" | sed 's/^/  /'
            echo ""
        fi

        # Show recent training progress
        echo "📈 Recent Training Activity:"
        tail -10 "$MAIN_LOG" | grep -E "(Epoch|Batch|Training Loss|Test Accuracy|COMPLETE)" | tail -5 | sed 's/^/  /'
        echo ""

        # Show any recent errors
        echo "⚠️  Recent Warnings/Errors:"
        tail -20 "$MAIN_LOG" | grep -E "(ERROR|Error|WARNING|Warning|Failed|failed)" | tail -3 | sed 's/^/  /' || echo "  None detected"
        echo ""

        # Check if training is complete
        if grep -q "PRODUCTION TRAINING COMPLETE" "$MAIN_LOG"; then
            echo "✅ Training completed! Extracting final results..."

            FINAL_ACCURACY=$(grep "Final Test Accuracy:" "$MAIN_LOG" | tail -1)
            FINAL_TIME=$(grep "Total Training Time:" "$MAIN_LOG" | tail -1)

            echo "🎯 Results:"
            echo "  $FINAL_ACCURACY"
            echo "  $FINAL_TIME"

            # Check for saved models
            MODEL_COUNT=$(ls -1 .models/*$(date +%Y%m%d)* 2>/dev/null | wc -l)
            echo "  Models saved: $MODEL_COUNT files"

            echo ""
            echo "🎉 Full training session completed successfully!"
            break
        fi

        # Check if process is still running
        if ! pgrep -f "python main.py" > /dev/null; then
            echo "⚠️  Training process not detected. Checking completion status..."
            if ! grep -q "PRODUCTION TRAINING COMPLETE" "$MAIN_LOG"; then
                echo "❌ Training may have stopped unexpectedly"
                echo "   Check $MAIN_LOG for details"
            fi
            break
        fi

    else
        echo "⏳ Waiting for training to start..."
    fi

    echo "Next update in 30 seconds..."
    echo "=============================================="
    echo ""

    sleep 30
done

echo "Monitor session ended at $(date)"