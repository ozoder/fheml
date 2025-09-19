"""FHE mathematical operations with scale management."""

import logging

import tenseal as ts

logger = logging.getLogger(__name__)


class FHEOperations:
    """Handles FHE mathematical operations with proper scale management."""

    def __init__(self, context: ts.Context) -> None:
        """
        Initialize FHE operations handler.

        Args:
            context: TenSEAL context for operations
        """
        self.context = context

    def safe_rescale(
        self, tensor: ts.CKKSTensor, target_scale: float | None = None
    ) -> ts.CKKSTensor:
        """
        Safely rescale a tensor to manage scale growth.

        Args:
            tensor: Input encrypted tensor
            target_scale: Optional target scale (unused in current implementation)

        Returns:
            Rescaled tensor
        """
        try:
            if hasattr(tensor, "rescale_to_next"):
                scale_ratio = tensor.scale() / self.context.global_scale
                if scale_ratio > 2.0:
                    logger.debug(f"Rescaling tensor: scale ratio {scale_ratio:.3f}")
                    tensor.rescale_to_next()
            return tensor
        except Exception as e:
            logger.warning(f"Rescaling failed: {e}")
            return tensor

    def check_scale_health(
        self, tensor: ts.CKKSTensor, operation_name: str = ""
    ) -> bool:
        """
        Check if tensor scale is within safe bounds.

        Args:
            tensor: Input encrypted tensor
            operation_name: Name of operation for logging

        Returns:
            True if scale is healthy, False otherwise
        """
        try:
            scale_ratio = tensor.scale() / self.context.global_scale

            if scale_ratio > 10:
                logger.warning(
                    f"{operation_name} - Scale too large: {scale_ratio:.2f}x global"
                )
                return False
            elif scale_ratio < 0.1:
                logger.warning(
                    f"{operation_name} - Scale too small: {scale_ratio:.4f}x global"
                )
                return False
            elif scale_ratio > 5:
                logger.info(
                    f"{operation_name} - Scale getting large: {scale_ratio:.2f}x global"
                )

            return True

        except Exception as e:
            logger.error(f"Error checking scale health: {e}")
            return False

    def bootstrap_if_needed(
        self, tensor: ts.CKKSTensor, min_scale_ratio: float = 0.1
    ) -> ts.CKKSTensor:
        """
        Bootstrap tensor if scale becomes too small.

        Args:
            tensor: Input encrypted tensor
            min_scale_ratio: Minimum acceptable scale ratio

        Returns:
            Bootstrapped tensor if needed, original tensor otherwise
        """
        try:
            if hasattr(tensor, "scale") and hasattr(tensor, "context"):
                try:
                    context = tensor.context()
                    scale_ratio = tensor.scale() / context.global_scale
                    if scale_ratio < min_scale_ratio:
                        logger.info(f"Bootstrapping tensor (scale ratio: {scale_ratio:.4f})")
                        # Simple bootstrap via decrypt/re-encrypt
                        # In production, use proper bootstrapping if available
                        from .encryption import FHEEncryption

                        encryption = FHEEncryption(context)
                        decrypted = encryption.decrypt_tensor(tensor)
                        return encryption.encrypt_tensor(decrypted)

                except Exception as inner_e:
                    logger.warning(f"Bootstrap scale check failed: {inner_e}")

        except Exception as e:
            logger.warning(f"Bootstrap failed: {e}")

        return tensor

    def safe_multiply(self, a: ts.CKKSTensor, b: ts.CKKSTensor) -> ts.CKKSTensor:
        """
        Safely multiply two encrypted tensors with scale management.

        Args:
            a: First encrypted tensor
            b: Second encrypted tensor

        Returns:
            Result of multiplication with managed scale

        Raises:
            ValueError: If multiplication fails after recovery attempts
        """
        try:
            result = a * b
            result = self.safe_rescale(result)
            return result

        except ValueError as e:
            if "scale out of bounds" in str(e):
                logger.warning("Scale out of bounds detected, attempting recovery...")
                a_boot = self.bootstrap_if_needed(a)
                b_boot = self.bootstrap_if_needed(b)
                result = a_boot * b_boot
                result = self.safe_rescale(result)
                return result
            else:
                raise

    def safe_square(self, tensor: ts.CKKSTensor) -> ts.CKKSTensor:
        """
        Safely square a tensor with scale management.

        Args:
            tensor: Input encrypted tensor

        Returns:
            Squared tensor with managed scale

        Raises:
            ValueError: If square operation fails after recovery attempts
        """
        try:
            # Check scale health before squaring
            if not self.check_scale_health(tensor, "pre-square"):
                tensor = self.bootstrap_if_needed(tensor)

            result = tensor.square()
            result = self.safe_rescale(result)

            # Check result scale health
            self.check_scale_health(result, "post-square")

            return result

        except ValueError as e:
            if "scale out of bounds" in str(e):
                logger.warning("Scale out of bounds in square operation, attempting recovery...")
                tensor_boot = self.bootstrap_if_needed(tensor)
                result = tensor_boot.square()
                result = self.safe_rescale(result)
                return result
            else:
                raise
