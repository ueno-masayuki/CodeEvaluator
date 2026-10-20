"""empirical (実測ベンチマーク) の E2E テスト。

実測を伴うため slow マーカーを付与し、寛容な判定でフレークを避ける。
CI では `pytest -m "not slow"` で除外できる。
"""

import math

import pytest

from codeevaluator.analyzer.empirical import BenchmarkConfig, _finalize, run_benchmark
from codeevaluator.models import ComplexityClass, Measurement
from tests.conftest import EXAMPLES_DIR


def _source(name: str) -> str:
    return (EXAMPLES_DIR / f"{name}.py").read_text(encoding="utf-8")


def _measurements(kind: str, model_fn, scale=5e-9) -> list[Measurement]:
    """決定的な合成測定値 (実行なし)。"""
    out = []
    for e in range(8, 14):
        n = 2 ** e
        t = scale * model_fn(n)
        out.append(Measurement(size=n, times=[t], median_time=t, input_kind=kind))
    return out


class TestFinalizePerKind:
    """_finalize が入力種別ごとにフィットすることの高速・決定的な検証。"""

    def test_kind_results_per_kind(self):
        # random は n²、sorted は n の合成データ
        measurements = _measurements("random", lambda n: n ** 2) + _measurements(
            "sorted", lambda n: float(n)
        )
        result = _finalize(
            "f", measurements, "random", [], timed_out=False, correctness_ok=True
        )
        assert set(result.kind_results) == {"random", "sorted"}
        assert (
            result.kind_results["random"].best_fit.complexity
            is ComplexityClass.QUADRATIC
        )
        assert (
            result.kind_results["sorted"].best_fit.complexity
            is ComplexityClass.LINEAR
        )
        # トップレベルは主要種別 (random) の値を保持する
        assert result.best_fit.complexity is ComplexityClass.QUADRATIC
        assert abs(result.exponent - 2.0) < 0.05
        assert abs(result.kind_results["sorted"].exponent - 1.0) < 0.05
        # summary_label も種別ごとに出せる
        assert "O(n²)" in result.kind_results["random"].summary_label()
        assert "O(n)" in result.kind_results["sorted"].summary_label()

    def test_insufficient_points_for_extra_kind_noted(self):
        # sorted は 2 点しかない → その種別だけ判定不能、注記に種別名が付く
        measurements = _measurements("random", lambda n: n ** 2) + [
            Measurement(size=256, times=[1e-3], median_time=1e-3, input_kind="sorted"),
            Measurement(size=512, times=[4e-3], median_time=4e-3, input_kind="sorted"),
        ]
        result = _finalize(
            "f", measurements, "random", [], timed_out=False, correctness_ok=True
        )
        assert result.kind_results["sorted"].best_fit is None
        assert math.isnan(result.kind_results["sorted"].exponent)
        assert result.best_fit is not None  # random 側は影響なし
        assert any("整列済み入力" in n for n in result.notes)


@pytest.mark.slow
def test_bubble_sort_measures_quadratic():
    config = BenchmarkConfig(min_exp=6, max_exp=10, repeats=3, per_run_timeout=5.0)
    result = run_benchmark(_source("bubble_sort"), "bubble_sort", config)
    assert result.correctness_ok
    assert result.best_fit is not None
    # 寛容な判定: O(n²) 系に入るか、指数が 1.6 を超えること
    assert (
        result.best_fit.complexity
        in (ComplexityClass.QUADRATIC, ComplexityClass.CUBIC)
        or result.exponent > 1.6
    ), f"実測: {result.summary_label()}, notes={result.notes}"


@pytest.mark.slow
def test_merge_sort_measures_linearithmic():
    config = BenchmarkConfig(min_exp=8, max_exp=12, repeats=3, per_run_timeout=5.0)
    result = run_benchmark(_source("merge_sort"), "merge_sort", config)
    assert result.correctness_ok
    assert result.best_fit is not None
    assert result.best_fit.complexity in (
        ComplexityClass.LINEAR,
        ComplexityClass.LINEARITHMIC,
        ComplexityClass.N_LOG2N,
    ), f"実測: {result.summary_label()}, notes={result.notes}"
    assert result.exponent < 1.5


@pytest.mark.slow
def test_insertion_sort_best_and_worst_cases():
    """挿入ソート: 整列済み入力は最良ケース O(n)、逆順入力は最悪ケース O(n²)。"""
    config = BenchmarkConfig(
        min_exp=7, max_exp=11, repeats=3, per_run_timeout=5.0,
        extra_kinds=("sorted", "reversed"),
    )
    result = run_benchmark(_source("insertion_sort"), "insertion_sort", config)
    assert result.correctness_ok
    best_case = result.kind_results["sorted"]
    worst_case = result.kind_results["reversed"]
    # 寛容な判定 (フレーク回避)
    assert (
        best_case.best_fit.complexity
        in (ComplexityClass.LINEAR, ComplexityClass.LINEARITHMIC)
        or best_case.exponent < 1.4
    ), f"整列済み: {best_case.summary_label()}"
    assert (
        worst_case.best_fit.complexity
        in (ComplexityClass.QUADRATIC, ComplexityClass.CUBIC)
        or worst_case.exponent > 1.6
    ), f"逆順: {worst_case.summary_label()}"


@pytest.mark.slow
def test_quick_sort_median_killer_quadratic():
    """中央ピボットのクイックソート: メディアンキラー入力で最悪 O(n²) が実測できる。"""
    config = BenchmarkConfig(
        min_exp=7, max_exp=11, repeats=3, per_run_timeout=5.0,
        extra_kinds=("median_killer",),
    )
    result = run_benchmark(_source("quick_sort"), "quick_sort", config)
    assert result.correctness_ok
    killer = result.kind_results["median_killer"]
    assert (
        killer.best_fit.complexity
        in (ComplexityClass.QUADRATIC, ComplexityClass.CUBIC)
        or killer.exponent > 1.6
    ), f"メディアンキラー: {killer.summary_label()}"
    # ランダム入力は平均 O(n log n) のまま
    assert result.exponent < 1.5


@pytest.mark.slow
def test_quick_sort_first_worst_on_sorted_input():
    """先頭ピボット版: 整列済み入力が最悪ケース O(n²) に該当する。"""
    config = BenchmarkConfig(
        min_exp=7, max_exp=11, repeats=3, per_run_timeout=5.0,
        extra_kinds=("sorted",),
    )
    result = run_benchmark(
        _source("quick_sort_first"), "quick_sort_first", config
    )
    assert result.correctness_ok
    worst = result.kind_results["sorted"]
    assert (
        worst.best_fit.complexity
        in (ComplexityClass.QUADRATIC, ComplexityClass.CUBIC)
        or worst.exponent > 1.6
    ), f"整列済み: {worst.summary_label()}"


@pytest.mark.slow
def test_non_sort_function_rejected():
    source = "def fake_sort(arr):\n    return arr\n"
    result = run_benchmark(source, "fake_sort")
    assert not result.correctness_ok
    assert result.best_fit is None


@pytest.mark.slow
def test_progress_callback_called():
    calls = []
    config = BenchmarkConfig(min_exp=6, max_exp=8, repeats=2)
    run_benchmark(
        _source("builtin_sort"),
        "builtin_sort",
        config,
        progress_callback=lambda done, total, msg: calls.append((done, total, msg)),
    )
    assert calls
    assert any("計測中" in c[2] for c in calls)
