"""FHE context management with proper error handling and type safety."""

import logging

import tenseal as ts

logger = logging.getLogger(__name__)


class FHEContextManager:
    """Manages FHE context creation and configuration."""

    def __init__(
        self,
        poly_modulus_degree: int = 8192,
        coeff_mod_bit_sizes: list[int] | None = None,
        scale_bits: int = 40,
    ) -> None:
        """
        Initialize FHE context manager.

        Args:
            poly_modulus_degree: Polynomial modulus degree for security
            coeff_mod_bit_sizes: Coefficient modulus bit sizes
            scale_bits: Scale bits for CKKS precision
        """
        self.poly_modulus_degree = max(poly_modulus_degree, 8192)
        self.scale_bits = scale_bits
        self.coeff_mod_bit_sizes = coeff_mod_bit_sizes or self._get_default_coeff_bits()
        self._context: ts.Context | None = None

    def _get_default_coeff_bits(self) -> list[int]:
        """Get default coefficient modulus bit sizes based on polynomial degree."""
        if self.poly_modulus_degree == 4096:
            return [50, 40, 40, 50]
        elif self.poly_modulus_degree == 8192:
            return [60, 40, 40, 40, 40, 60]
        elif self.poly_modulus_degree == 16384:
            return [60, 40, 40, 40, 40, 40, 40, 60]
        else:
            return [60, 40, 40, 40, 60]

    def create_context(self) -> ts.Context:
        """
        Create and configure FHE context.

        Returns:
            Configured TenSEAL context

        Raises:
            RuntimeError: If context creation fails
        """
        logger.info(
            f"Creating FHE context: poly_degree={self.poly_modulus_degree}, "
            f"coeff_bits={self.coeff_mod_bit_sizes}, scale_bits={self.scale_bits}"
        )

        try:
            context = ts.context(
                ts.SCHEME_TYPE.CKKS,
                poly_modulus_degree=self.poly_modulus_degree,
                coeff_mod_bit_sizes=self.coeff_mod_bit_sizes,
            )

            context.global_scale = 2**self.scale_bits
            context.generate_galois_keys()

            self._context = context
            logger.info(
                f"Successfully created FHE context with global scale: {context.global_scale}"
            )
            return context

        except Exception as e:
            logger.warning(f"Failed to create FHE context: {e}")
            return self._create_fallback_context()

    def _create_fallback_context(self) -> ts.Context:
        """Create context with fallback parameters."""
        logger.info("Creating fallback FHE context...")
        fallback_coeff_bits = [60, 40, 40, 60]

        try:
            context = ts.context(
                ts.SCHEME_TYPE.CKKS,
                poly_modulus_degree=8192,
                coeff_mod_bit_sizes=fallback_coeff_bits,
            )
            context.global_scale = 2**self.scale_bits
            context.generate_galois_keys()

            self._context = context
            logger.info(
                f"Fallback context created with global scale: {context.global_scale}"
            )
            return context

        except Exception as e:
            logger.error(f"Failed to create fallback context: {e}")
            raise RuntimeError(f"Unable to create FHE context: {e}") from e

    @property
    def context(self) -> ts.Context:
        """Get the current context, creating it if necessary."""
        if self._context is None:
            self._context = self.create_context()
        return self._context
