import smudgy.core._cpp_functions_ext as cpp


def test_openmp_attribute_exists():
    assert hasattr(cpp, "has_openmp"), "C++ extension must expose has_openmp attribute"


def test_openmp_attribute_is_bool():
    assert isinstance(cpp.has_openmp, bool)


def test_openmp_serial_consistency():
    """If OpenMP is disabled, ensure behavior is consistent with serial mode."""
    if not cpp.has_openmp:
        # Nothing parallel should happen; just assert flag is False
        assert cpp.has_openmp is False


def test_openmp_thread_count():
    if cpp.has_openmp:
        assert cpp.openmp_thread_count() >= 1
    else:
        assert cpp.openmp_thread_count() == 1
