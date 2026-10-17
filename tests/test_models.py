"""models モジュールのテスト。"""

import math

import numpy as np
import pytest

from codeevaluator.models import (
    FIT_CANDIDATES,
    ComplexityClass,
    ComplexityTerm,
    EmpiricalResult,
    FitResult,
)


class TestComplexityClass:
    def test_label(self):
        assert ComplexityClass.LINEARITHMIC.label() == "O(n log n)"
        assert ComplexityClass.QUADRATIC.label() == "O(n²)"
        assert ComplexityClass.UNKNOWN.label() == "不明"

    def test_model_values(self):
        n = np.array([2.0, 4.0, 8.0])
        np.testing.assert_allclose(ComplexityClass.LINEAR.model(n), n)
        np.testing.assert_allclose(
            ComplexityClass.LINEARITHMIC.model(n), n * np.log2(n)
        )
        np.testing.assert_allclose(ComplexityClass.QUADRATIC.model(n), n ** 2)
        np.testing.assert_allclose(ComplexityClass.CUBIC.model(n), n ** 3)
        np.testing.assert_allclose(ComplexityClass.LOGARITHMIC.model(n), np.log2(n))

    def test_model_unknown_raises(self):
        with pytest.raises(ValueError):
            ComplexityClass.UNKNOWN.model(np.array([1.0]))

    def test_fit_candidates_all_have_model(self):
        n = np.array([16.0, 256.0])
        for c in FIT_CANDIDATES:
            assert np.all(c.model(n) > 0)


class TestComplexityTerm:
    def test_times(self):
        # n × n = n²、n × log n = n log n
        assert ComplexityTerm.N.times(ComplexityTerm.N) == ComplexityTerm(2, 0)
        assert ComplexityTerm.N.times(ComplexityTerm.LOG_N) == ComplexityTerm(1, 1)

    def test_ordering(self):
        # 直列合成の max 用: n² > n log n > n > log n > 1
        assert ComplexityTerm(2, 0) > ComplexityTerm(1, 1)
        assert ComplexityTerm(1, 1) > ComplexityTerm(1, 0)
        assert ComplexityTerm(1, 0) > ComplexityTerm(0, 1)
        assert ComplexityTerm(0, 1) > ComplexityTerm(0, 0)

    def test_to_class(self):
        assert ComplexityTerm(0, 0).to_class() is ComplexityClass.CONSTANT
        assert ComplexityTerm(1, 0).to_class() is ComplexityClass.LINEAR
        assert ComplexityTerm(1, 1).to_class() is ComplexityClass.LINEARITHMIC
        assert ComplexityTerm(2, 0).to_class() is ComplexityClass.QUADRATIC
        assert ComplexityTerm(3, 0).to_class() is ComplexityClass.CUBIC
        assert ComplexityTerm(4, 0).to_class() is ComplexityClass.UNKNOWN

    def test_label_unknown_term(self):
        assert ComplexityTerm(2, 1).label() == "O(n² log n)"


class TestEmpiricalResult:
    def _mk(self, exponent: float, best: ComplexityClass | None):
        fit = (
            FitResult(complexity=best, scale=1e-9, r_squared=0.99, aic=-10.0)
            if best
            else None
        )
        return EmpiricalResult(
            function_name="f",
            measurements=[],
            fits=[fit] if fit else [],
            best_fit=fit,
            exponent=exponent,
            exponent_stderr=0.01,
        )

    def test_summary_label(self):
        r = self._mk(1.98, ComplexityClass.QUADRATIC)
        assert r.summary_label() == "O(n^1.98) ≈ O(n²)"

    def test_summary_label_no_fit(self):
        assert self._mk(math.nan, None).summary_label() == "判定不能"
