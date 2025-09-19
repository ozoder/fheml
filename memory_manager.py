#!/usr/bin/env python3

import gc
import os
import psutil
import time
import logging
from typing import Optional, Callable, Any
from functools import wraps

logger = logging.getLogger(__name__)

class GracefulMemoryManager:
    """Manages memory usage gracefully without process termination."""

    def __init__(self,
                 memory_limit_gb: float = 25.0,  # Conservative limit for stable FHE operations
                 warning_threshold: float = 0.8,
                 critical_threshold: float = 0.9):
        self.memory_limit_bytes = int(memory_limit_gb * 1024 * 1024 * 1024)
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.process = psutil.Process(os.getpid())
        self.degradation_level = 0  # 0=normal, 1=reduced, 2=minimal

    def get_memory_usage(self) -> tuple[int, float]:
        """Get current memory usage in bytes and percentage of limit."""
        memory_info = self.process.memory_info()
        rss_bytes = memory_info.rss
        percentage = rss_bytes / self.memory_limit_bytes
        return rss_bytes, percentage

    def check_memory_status(self) -> str:
        """Check current memory status and return recommendation."""
        _, percentage = self.get_memory_usage()

        if percentage >= self.critical_threshold:
            return "CRITICAL"
        elif percentage >= self.warning_threshold:
            return "WARNING"
        else:
            return "NORMAL"

    def force_garbage_collection(self):
        """Aggressive garbage collection."""
        logger.info("Performing aggressive garbage collection...")
        for i in range(3):
            collected = gc.collect()
            logger.debug(f"GC round {i+1}: collected {collected} objects")

        # Force garbage collection of all generations
        gc.collect(0)
        gc.collect(1)
        gc.collect(2)

    def get_memory_reduction_strategy(self) -> dict:
        """Get strategy for reducing memory usage based on current level."""
        memory_bytes, percentage = self.get_memory_usage()
        memory_mb = memory_bytes // (1024 * 1024)

        if percentage >= self.critical_threshold:
            # CRITICAL: Reduce but maintain architecture for high accuracy
            self.degradation_level = 2
            return {
                "batch_size": 2,
                "max_samples": min(200, max(100, int(300 * (1.2 - percentage)))),  # Minimum 100 samples
                "hidden_dims": [128, 64],  # Maintain reasonable depth
                "enable_checkpointing": True,
                "clear_cache_frequency": 1,  # Clear after every batch
                "message": f"CRITICAL: {memory_mb}MB used. Reduced architecture for stability."
            }
        elif percentage >= self.warning_threshold:
            # WARNING: Moderate reduction, maintain high-accuracy architecture
            self.degradation_level = 1
            return {
                "batch_size": 3,
                "max_samples": min(300, max(150, int(500 * (1.1 - percentage)))),  # Minimum 150 samples
                "hidden_dims": [256, 128],  # Maintain depth for accuracy
                "enable_checkpointing": True,
                "clear_cache_frequency": 2,
                "message": f"WARNING: {memory_mb}MB used. Moderate reduction for stability."
            }
        else:
            # NORMAL: No reduction needed
            self.degradation_level = 0
            return {
                "batch_size": None,  # Use default
                "max_samples": None,  # Use default
                "hidden_dims": None,  # Use default
                "enable_checkpointing": False,
                "clear_cache_frequency": 5,
                "message": f"NORMAL: {memory_mb}MB used. No reduction needed."
            }

def memory_aware(memory_manager: GracefulMemoryManager):
    """Decorator that makes functions memory-aware."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # Check memory before function execution
            status = memory_manager.check_memory_status()

            if status in ["WARNING", "CRITICAL"]:
                logger.warning(f"Memory status: {status}")
                memory_manager.force_garbage_collection()

                # Get reduction strategy
                strategy = memory_manager.get_memory_reduction_strategy()
                logger.info(strategy["message"])

                # If function has parameters we can modify, apply reductions
                if hasattr(func, '__name__') and func.__name__ in ['train_batch', 'process_samples']:
                    # Modify function parameters based on strategy
                    if 'batch_size' in kwargs and strategy['batch_size']:
                        kwargs['batch_size'] = strategy['batch_size']
                    if 'max_samples' in kwargs and strategy['max_samples']:
                        kwargs['max_samples'] = strategy['max_samples']

            # Execute function
            result = func(*args, **kwargs)

            # Clean up after execution if in degradation mode
            if memory_manager.degradation_level > 0:
                memory_manager.force_garbage_collection()

            return result
        return wrapper
    return decorator

class AdaptiveTrainingManager:
    """Manages FHE training with adaptive memory management."""

    def __init__(self, memory_limit_gb: float = 25.0):
        self.memory_manager = GracefulMemoryManager(memory_limit_gb)
        self.checkpoints = []
        self.adaptive_params = {
            "batch_size": 5,
            "max_train_samples": 100,  # Proven working size
            "max_test_samples": 40,
            "hidden_dims": [48]  # Proven stable architecture
        }

    def adapt_parameters(self) -> dict:
        """Adapt training parameters based on current memory usage."""
        strategy = self.memory_manager.get_memory_reduction_strategy()

        # Update parameters based on strategy (but preserve original architecture for stability)
        if strategy["batch_size"]:
            self.adaptive_params["batch_size"] = strategy["batch_size"]
        if strategy["max_samples"]:
            self.adaptive_params["max_train_samples"] = strategy["max_samples"]
            self.adaptive_params["max_test_samples"] = max(5, strategy["max_samples"] // 4)
        # Don't override hidden_dims to maintain requested architecture for completion

        logger.info(f"Adapted parameters: {self.adaptive_params}")
        return self.adaptive_params

    def create_checkpoint(self, model_state: dict, epoch: int):
        """Create memory-efficient checkpoint."""
        if self.memory_manager.degradation_level > 0:
            # In degradation mode, save to disk immediately and clear from memory
            checkpoint_path = f".checkpoints/checkpoint_epoch_{epoch}.pt"
            os.makedirs(".checkpoints", exist_ok=True)

            import torch
            torch.save(model_state, checkpoint_path)
            logger.info(f"Checkpoint saved to {checkpoint_path}")

            # Clear the model state from memory
            del model_state
            gc.collect()
        else:
            # Normal mode, keep in memory
            self.checkpoints.append({"epoch": epoch, "state": model_state})

    def monitor_and_log_memory(self, phase: str):
        """Monitor and log memory usage during different phases."""
        memory_bytes, percentage = self.memory_manager.get_memory_usage()
        memory_mb = memory_bytes // (1024 * 1024)
        status = self.memory_manager.check_memory_status()

        logger.info(f"{phase} - Memory: {memory_mb}MB ({percentage:.1%}), Status: {status}")

        if status == "CRITICAL":
            logger.error("CRITICAL memory usage detected!")
            self.memory_manager.force_garbage_collection()