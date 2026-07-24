import pytest

from .audit_predictive_core_retention import retention_drop


def test_retention_drop_is_reference_minus_candidate() -> None:
    assert retention_drop(0.80, 0.73) == pytest.approx(0.07)
    assert retention_drop(0.70, 0.75) == pytest.approx(-0.05)
