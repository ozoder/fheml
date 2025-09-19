"""Memory management for FHE training."""

import gc
import logging
import os
from functools import wraps
from typing import Any

import psutil

logger = logging.getLogger(__name__)


class MemoryManager:
    """Manages memory usage gracefully without process termination."""

    def __init__(
        self,
        memory_limit_gb: float = 25.0,
        warning_threshold: float = 0.7,
        critical_threshold: float = 0.85,
    ) -> None:
        """
        Initialize memory manager.

        Args:
            memory_limit_gb: Memory limit in GB
            warning_threshold: Warning threshold as fraction of limit
            critical_threshold: Critical threshold as fraction of limit
        """
        self.memory_limit_bytes = int(memory_limit_gb * 1024 * 1024 * 1024)
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.process = psutil.Process(os.getpid())
        self.degradation_level = 0  # 0=normal, 1=reduced, 2=minimal

    def get_memory_usage(self) -> tuple[int, float]:
        """
        Get current memory usage in bytes and percentage of limit.

        Returns:
            Tuple of (memory_bytes, percentage_of_limit)
        """
        memory_info = self.process.memory_info()
        rss_bytes = memory_info.rss
        percentage = rss_bytes / self.memory_limit_bytes
        return rss_bytes, percentage

    def check_memory_status(self) -> str:
        """
        Check current memory status and return recommendation.

        Returns:
            Memory status: "NORMAL", "WARNING", or "CRITICAL"
        """
        _, percentage = self.get_memory_usage()

        if percentage >= self.critical_threshold:
            return "CRITICAL"
        elif percentage >= self.warning_threshold:
            return "WARNING"
        else:
            return "NORMAL"

    def force_garbage_collection(self) -> None:
        """Aggressive garbage collection."""
        logger.info("Performing aggressive garbage collection...")
        for i in range(3):
            collected = gc.collect()
            logger.debug(f"GC round {i+1}: collected {collected} objects")

        # Force garbage collection of all generations
        gc.collect(0)
        gc.collect(1)
        gc.collect(2)

    def get_memory_reduction_strategy(self) -> dict[str, Any]:
        """
        Get strategy for reducing memory usage based on current level.

        Returns:
            Dictionary with reduction parameters
        """
        memory_bytes, percentage = self.get_memory_usage()
        memory_mb = memory_bytes // (1024 * 1024)

        if percentage >= self.critical_threshold:
            # CRITICAL: Moderate reduction while preserving deep architecture
            self.degradation_level = 2
            return {
                "batch_size": 4,
                "max_samples": min(100, max(20, int(200 * (1.1 - percentage)))),
                "hidden_dims": [512, 256, 128],
                "enable_checkpointing": True,
                "clear_cache_frequency": 1,
                "message": f"CRITICAL: {memory_mb}MB used. Moderate reduction.",
            }
        elif percentage >= self.warning_threshold:
            # WARNING: Light reduction, preserve architecture
            self.degradation_level = 1
            return {
                "batch_size": 6,
                "max_samples": min(200, max(50, int(400 * (1.05 - percentage)))),
                "hidden_dims": [1024, 512, 256],
                "enable_checkpointing": True,
                "clear_cache_frequency": 2,
                "message": f"WARNING: {memory_mb}MB used. Light reduction.",
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
                "message": f"NORMAL: {memory_mb}MB used. No reduction needed.",
            }


class AdaptiveTrainingManager:
    """Manages FHE training with adaptive memory management."""

    def __init__(self, memory_limit_gb: float = 25.0) -> None:
        """
        Initialize adaptive training manager.

        Args:
            memory_limit_gb: Memory limit in GB
        """
        self.memory_manager = MemoryManager(memory_limit_gb)
        self.adaptive_params = {
            "batch_size": 8,
            "max_train_samples": 100,
            "max_test_samples": 20,
            "hidden_dims": [256, 128, 64],
        }

    def adapt_parameters(self) -> dict[str, Any]:
        """
        Adapt training parameters based on current memory usage.

        Returns:
            Dictionary of adapted parameters
        """
        strategy = self.memory_manager.get_memory_reduction_strategy()

        # Update parameters based on strategy
        if strategy["batch_size"]:
            self.adaptive_params["batch_size"] = strategy["batch_size"]
        if strategy["max_samples"]:
            self.adaptive_params["max_train_samples"] = strategy["max_samples"]
            self.adaptive_params["max_test_samples"] = max(5, strategy["max_samples"] // 4)

        logger.info(f"Adapted parameters: {self.adaptive_params}")
        return self.adaptive_params

    def monitor_and_log_memory(self, phase: str) -> None:
        """
        Monitor and log memory usage during different phases.

        Args:
            phase: Current training phase
        """
        memory_bytes, percentage = self.memory_manager.get_memory_usage()
        memory_mb = memory_bytes // (1024 * 1024)
        status = self.memory_manager.check_memory_status()

        logger.info(f"{phase} - Memory: {memory_mb}MB ({percentage:.1%}), Status: {status}")

        if status == "CRITICAL":
            logger.error("CRITICAL memory usage detected!")
            self.memory_manager.force_garbage_collection()


def memory_aware(memory_manager: MemoryManager):
    """Decorator that makes functions memory-aware."""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Check memory before function execution
            status = memory_manager.check_memory_status()

            if status in ["WARNING", "CRITICAL"]:
                logger.warning(f"Memory status: {status}")
                memory_manager.force_garbage_collection()

                # Get reduction strategy
                strategy = memory_manager.get_memory_reduction_strategy()
                logger.info(strategy["message"])

                # If function has parameters we can modify, apply reductions
                if hasattr(func, "__name__") and func.__name__ in ["train_batch", "process_samples"]:
                    # Modify function parameters based on strategy
                    if "batch_size" in kwargs and strategy["batch_size"]:
                        kwargs["batch_size"] = strategy["batch_size"]
                    if "max_samples" in kwargs and strategy["max_samples"]:
                        kwargs["max_samples"] = strategy["max_samples"]

            # Execute function
            result = func(*args, **kwargs)

            # Clean up after execution if in degradation mode
            if memory_manager.degradation_level > 0:
                memory_manager.force_garbage_collection()

            return result

        return wrapper

    return decorator
