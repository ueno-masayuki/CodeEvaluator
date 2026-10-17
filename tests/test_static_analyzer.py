"""静的 AST 解析 (理論計算量) のテスト。"""

import pytest

from codeevaluator.analyzer.static import analyze_function
from codeevaluator.models import ComplexityClass


@pytest.mark.parametrize(
    "name,func,expected",
    [
        ("bubble_sort", "bubble_sort", ComplexityClass.QUADRATIC),
        ("insertion_sort", "insertion_sort", ComplexityClass.QUADRATIC),
        ("selection_sort", "selection_sort", ComplexityClass.QUADRATIC),
        ("merge_sort", "merge_sort", ComplexityClass.LINEARITHMIC),
        ("quick_sort", "quick_sort", ComplexityClass.LINEARITHMIC),
        ("heap_sort", "heap_sort", ComplexityClass.LINEARITHMIC),
        ("builtin_sort", "builtin_sort", ComplexityClass.LINEARITHMIC),
    ],
)
def test_examples_theoretical_complexity(example_source, name, func, expected):
    result = analyze_function(example_source(name), func)
    assert result.complexity is expected, (
        f"{func}: 期待 {expected.label()}, 実際 {result.complexity.label()}\n"
        + "\n".join(result.reasoning)
    )
    assert result.reasoning, "判定根拠が空"
    # 根拠に日本語が含まれる
    assert any(
        any("぀" <= ch <= "ヿ" or "一" <= ch <= "鿿" for ch in r)
        for r in result.reasoning
    )


def test_quick_sort_worst_case_noted(example_source):
    result = analyze_function(example_source("quick_sort"), "quick_sort")
    assert result.recursion_kind == "divide_partition"
    assert result.worst_case is ComplexityClass.QUADRATIC
    assert any("最悪" in r for r in result.reasoning)


def test_merge_sort_recursion_detected(example_source):
    result = analyze_function(example_source("merge_sort"), "merge_sort")
    assert result.has_recursion
    assert result.recursion_kind == "divide_halving"


def test_single_loop_linear():
    source = """
def scan(arr):
    total = 0
    for x in arr:
        total += x
    return total
"""
    result = analyze_function(source, "scan")
    assert result.complexity is ComplexityClass.LINEAR
    assert result.max_loop_depth == 1


def test_triple_loop_cubic():
    source = """
def f(arr):
    n = len(arr)
    for i in range(n):
        for j in range(n):
            for k in range(n):
                arr[0] = i + j + k
    return arr
"""
    result = analyze_function(source, "f")
    assert result.complexity is ComplexityClass.CUBIC
    assert result.max_loop_depth == 3


def test_halving_while_logarithmic():
    source = """
def f(n):
    count = 0
    while n > 1:
        n //= 2
        count += 1
    return count
"""
    result = analyze_function(source, "f")
    assert result.complexity is ComplexityClass.LOGARITHMIC


def test_constant_range_loop():
    source = """
def f(arr):
    for i in range(10):
        arr[0] = i
    return arr
"""
    result = analyze_function(source, "f")
    assert result.complexity is ComplexityClass.CONSTANT


def test_exponential_recursion_flagged():
    source = """
def fib(n):
    if n <= 1:
        return n
    return fib(n - 1) + fib(n - 2)
"""
    result = analyze_function(source, "fib")
    assert result.complexity is ComplexityClass.EXPONENTIAL


def test_insertion_sort_via_insert_method():
    source = """
def insert_sort(arr):
    result = []
    for x in arr:
        for i in range(len(result)):
            if result[i] > x:
                result.insert(i, x)
                break
        else:
            result.append(x)
    return result
"""
    result = analyze_function(source, "insert_sort")
    # insert の暗黙コスト O(n) を含めて O(n²) 以上
    assert result.complexity in (ComplexityClass.QUADRATIC, ComplexityClass.CUBIC)


def test_helper_function_cost_propagates():
    source = """
def find_min_index(data, start):
    best = start
    for j in range(start, len(data)):
        if data[j] < data[best]:
            best = j
    return best

def selection(arr):
    data = list(arr)
    for i in range(len(data)):
        m = find_min_index(data, i)
        data[i], data[m] = data[m], data[i]
    return data
"""
    result = analyze_function(source, "selection")
    assert result.complexity is ComplexityClass.QUADRATIC


def test_unknown_function_raises():
    with pytest.raises(ValueError, match="見つかりません"):
        analyze_function("def f(a):\n    return a", "g")


def test_confidence_bounds(example_source):
    for name in ("bubble_sort", "merge_sort"):
        result = analyze_function(example_source(name), name)
        assert 0.3 <= result.confidence <= 0.9
