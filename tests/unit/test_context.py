"""Unit tests for FHE context management."""

import pytest
import tenseal as ts

from src.fheml.core.context import FHEContextManager


class TestFHEContextManager:
    """Test cases for FHE context manager."""

    def test_init_default_parameters(self):
        """Test initialization with default parameters."""
        manager = FHEContextManager()

        assert manager.poly_modulus_degree == 8192
        assert manager.scale_bits == 40
        assert manager.coeff_mod_bit_sizes == [60, 40, 40, 40, 40, 60]

    def test_init_custom_parameters(self):
        """Test initialization with custom parameters."""
        custom_coeff_bits = [50, 30, 30, 50]
        manager = FHEContextManager(
            poly_modulus_degree=8192,  # Use valid degree (>=8192)
            coeff_mod_bit_sizes=custom_coeff_bits,
            scale_bits=35
        )

        assert manager.poly_modulus_degree == 8192
        assert manager.scale_bits == 35
        assert manager.coeff_mod_bit_sizes == custom_coeff_bits

    def test_min_poly_modulus_degree(self):
        """Test minimum polynomial modulus degree enforcement."""
        manager = FHEContextManager(poly_modulus_degree=2048)
        assert manager.poly_modulus_degree == 8192  # Should be corrected to minimum

    def test_get_default_coeff_bits_4096(self):
        """Test default coefficient bits for 4096 degree."""
        # Create manager with 4096 but it gets corrected to 8192
        manager = FHEContextManager(poly_modulus_degree=4096)
        # Since it's corrected to 8192, expect 8192 defaults
        expected = [60, 40, 40, 40, 40, 60]
        assert manager._get_default_coeff_bits() == expected

    def test_get_default_coeff_bits_8192(self):
        """Test default coefficient bits for 8192 degree."""
        manager = FHEContextManager(poly_modulus_degree=8192)
        expected = [60, 40, 40, 40, 40, 60]
        assert manager._get_default_coeff_bits() == expected

    def test_get_default_coeff_bits_16384(self):
        """Test default coefficient bits for 16384 degree."""
        manager = FHEContextManager(poly_modulus_degree=16384)
        expected = [60, 40, 40, 40, 40, 40, 40, 60]
        assert manager._get_default_coeff_bits() == expected

    def test_get_default_coeff_bits_other(self):
        """Test default coefficient bits for other degrees."""
        manager = FHEContextManager(poly_modulus_degree=32768)
        expected = [60, 40, 40, 40, 60]
        assert manager._get_default_coeff_bits() == expected

    def test_create_context_success(self):
        """Test successful context creation."""
        manager = FHEContextManager()
        context = manager.create_context()

        assert isinstance(context, ts.Context)
        assert context.global_scale == 2**40
        assert manager._context == context

    def test_context_property(self):
        """Test context property access."""
        manager = FHEContextManager()

        # First access should create context
        context1 = manager.context
        assert isinstance(context1, ts.Context)

        # Second access should return same context
        context2 = manager.context
        assert context1 is context2

    def test_create_fallback_context(self):
        """Test fallback context creation."""
        manager = FHEContextManager()

        # Force fallback by setting invalid parameters
        manager.coeff_mod_bit_sizes = [1000, 2000]  # Invalid bit sizes

        # This should trigger fallback
        context = manager.create_context()
        assert isinstance(context, ts.Context)
        assert context.global_scale == 2**40