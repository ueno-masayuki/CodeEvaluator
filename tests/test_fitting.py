"""fitting (モデル選択・指数フィット) のテスト。合成データのみで高速・決定的。"""

import math

import numpy as np
import pytest

from codeevaluator.fitting import classes_agree, fit_measurements
from codeevaluator.models import ComplexityClass

_RNG = np.random.default_rng(12345)
_SIZES = [2 ** e for e in range(8, 14)]


def _synth(model_fn, scale=3e-9, noise=0.05):
    """t = scale * f(n) * (1 + ノイズ) の合成データ。"""
    times = []
    for n in _SIZES:
        base = scale * model_fn(n)
        times.append(base * (1.0 + _RNG.uniform(-noise, noise)))
    return _SIZES, times


def test_quadratic_data_identified():
    sizes, times = _synth(lambda n: n ** 2)
    fits, exponent, stderr, notes = fit_measurements(sizes, times)
    assert fits[0].complexity is ComplexityClass.QUADRATIC
    assert abs(exponent - 2.0) < 0.1
    assert stderr < 0.1
    assert fits[0].r_squared > 0.99


def test_linearithmic_data_identified():
    sizes, times = _synth(lambda n: n * math.log2(n))
    fits, exponent, _, _ = fit_measurements(sizes, times)
    assert fits[0].complexity is ComplexityClass.LINEARITHMIC
    # n log n の log-log 傾きは 1.1〜1.3 程度
    assert 1.0 < exponent < 1.4


def test_cubic_data_identified():
    sizes, times = _synth(lambda n: n ** 3)
    fits, exponent, _, _ = fit_measurements(sizes, times)
    assert fits[0].complexity is ComplexityClass.CUBIC
    assert abs(exponent - 3.0) < 0.1


def test_insufficient_points_returns_unknown():
    fits, exponent, stderr, notes = fit_measurements([256, 512], [0.1, 0.2])
    assert fits == []
    assert math.isnan(exponent)
    assert any("測定点" in n for n in notes)


def test_nonpositive_times_filtered():
    sizes = [256, 512, 1024, 2048, 4096]
    times = [0.0, 1e-4, 4e-4, 1.6e-3, 6.4e-3]  # 先頭が 0 → 除外して 4 点
    fits, exponent, _, _ = fit_measurements(sizes, times)
    assert fits  # 4 点残るのでフィットできる
    assert fits[0].complexity is ComplexityClass.QUADRATIC


def test_ambiguity_note_for_close_models():
    # O(n) と O(n log n) のちょうど中間 (幾何平均) のデータ → AIC 差 ≈ 0
    sizes, times = _synth(lambda n: n * math.sqrt(math.log2(n)), noise=0.0)
    _, _, _, notes = fit_measurements(sizes, times)
    assert any("不確実" in n for n in notes)


def test_fits_sorted_by_aic():
    sizes, times = _synth(lambda n: n ** 2)
    fits, _, _, _ = fit_measurements(sizes, times)
    aics = [f.aic for f in fits]
    assert aics == sorted(aics)


class TestClassesAgree:
    def test_exact_match(self):
        assert classes_agree(ComplexityClass.QUADRATIC, ComplexityClass.QUADRATIC)

    def test_mismatch(self):
        assert not classes_agree(ComplexityClass.QUADRATIC, ComplexityClass.LINEAR)

    def test_worst_case_match(self):
        # クイックソート: 理論平均 n log n・最悪 n² → 実測 n² でも整合
        assert classes_agree(
            ComplexityClass.LINEARITHMIC,
            ComplexityClass.QUADRATIC,
            worst_case=ComplexityClass.QUADRATIC,
        )

    def test_neighboring_log_classes(self):
        assert classes_agree(ComplexityClass.LINEARITHMIC, ComplexityClass.N_LOG2N)
        assert classes_agree(ComplexityClass.LINEARITHMIC, ComplexityClass.LINEAR)
