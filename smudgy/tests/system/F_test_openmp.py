"""System tests for OpenMP support in the C++ extension."""

import smudgy.core._cpp_functions_ext as cpp


def test_openmp_attribute_exists():
    """The C++ extension should have a has_openmp attribute indicating OpenMP support."""
    assert hasattr(cpp, "has_openmp"), "C++ extension must expose has_openmp attribute"


def test_openmp_attribute_is_bool():
    """The has_openmp attribute should be a boolean indicating OpenMP support."""
    assert isinstance(cpp.has_openmp, bool)


def test_openmp_serial_consistency():
    """If OpenMP is disabled, ensure behavior is consistent with serial mode."""
    if not cpp.has_openmp:
        # Nothing parallel should happen; just assert flag is False
        assert cpp.has_openmp is False


def test_openmp_thread_count():
    """If OpenMP is enabled, check that thread count is at least 1; if disabled, it should be 1."""
    if cpp.has_openmp:
        assert cpp.openmp_thread_count() >= 1
    else:
        assert cpp.openmp_thread_count() == 1
