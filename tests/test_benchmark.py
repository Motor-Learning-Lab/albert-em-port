import pytest

pytest.skip(
    "Performance benchmarks have been removed to keep the test suite fast and clear.\n"
    "Use tests/test_smoke.py for correctness checks.",
    allow_module_level=True,
)
