"""empirical (実測ベンチマーク) の E2E テスト。

実測を伴うため slow マーカーを付与し、寛容な判定でフレークを避ける。
CI では `pytest -m "not slow"` で除外できる。
"""

import pytest

from codeevaluator.analyzer.empirical import BenchmarkConfig, run_benchmark
from codeevaluator.models import ComplexityClass
from tests.conftest import EXAMPLES_DIR


def _source(name: str) -> str:
    return (EXAMPLES_DIR / f"{name}.py").read_text(encoding="utf-8")


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
