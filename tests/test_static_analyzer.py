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
        ("quick_sort_first", "quick_sort_first", ComplexityClass.LINEARITHMIC),
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


def test_quick_sort_middle_pivot_detected(example_source):
    result = analyze_function(example_source("quick_sort"), "quick_sort")
    assert result.pivot_kind == "middle"
    # 整列済み/逆順では最悪ケースが出ないことが根拠に明記される
    assert any("最悪ケースは発生しません" in r for r in result.reasoning)


def test_quick_sort_first_pivot_detected(example_source):
    result = analyze_function(example_source("quick_sort_first"), "quick_sort_first")
    assert result.pivot_kind == "first"
    assert result.worst_case is ComplexityClass.QUADRATIC
    assert any("整列済み・逆順入力が最悪ケース" in r for r in result.reasoning)


def test_random_pivot_detected():
    source = """
import random
def quick_sort_rand(arr):
    if len(arr) <= 1:
        return list(arr)
    pivot = random.choice(arr)
    less = [x for x in arr if x < pivot]
    equal = [x for x in arr if x == pivot]
    greater = [x for x in arr if x > pivot]
    return quick_sort_rand(less) + equal + quick_sort_rand(greater)
"""
    result = analyze_function(source, "quick_sort_rand")
    assert result.pivot_kind == "random"


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


def test_three_way_merge_sort_linearithmic():
    source = """
def merge3_sort(arr):
    if len(arr) <= 1:
        return list(arr)
    third = len(arr) // 3
    a = merge3_sort(arr[:third])
    b = merge3_sort(arr[third:2*third])
    c = merge3_sort(arr[2*third:])
    out = []
    import heapq
    for x in heapq.merge(a, b, c):
        out.append(x)
    return out
"""
    result = analyze_function(source, "merge3_sort")
    # T(n) = 3·T(n/3) + O(n) → O(n log n)
    assert result.complexity is ComplexityClass.LINEARITHMIC
    assert any("3 分割" in r or "T(n/3)" in r.replace(" ", "") for r in result.reasoning)


def test_master_case1_integer_exponent():
    source = """
def quad_divide(arr):
    if len(arr) <= 1:
        return list(arr)
    mid = len(arr) // 2
    a = quad_divide(arr[:mid])
    b = quad_divide(arr[:mid])
    c = quad_divide(arr[mid:])
    d = quad_divide(arr[mid:])
    for x in arr:
        pass
    return a + b + c + d
"""
    result = analyze_function(source, "quad_divide")
    # T(n) = 4·T(n/2) + O(n) → O(n²) (ケース1、log2 4 = 2)
    assert result.complexity is ComplexityClass.QUADRATIC


def test_master_case1_noninteger_exponent_unknown():
    source = """
def karatsuba_like(arr):
    if len(arr) <= 1:
        return list(arr)
    mid = len(arr) // 2
    a = karatsuba_like(arr[:mid])
    b = karatsuba_like(arr[mid:])
    c = karatsuba_like(arr[:mid])
    for x in arr:
        pass
    return a + b + c
"""
    result = analyze_function(source, "karatsuba_like")
    # T(n) = 3·T(n/2) + O(n) → O(n^1.58): 既知クラス外 → 不明 + 指数を根拠に明記
    assert result.complexity is ComplexityClass.UNKNOWN
    assert result.confidence <= 0.5
    assert any("n^1.58" in r for r in result.reasoning)


def test_binary_search_logarithmic():
    source = """
def bsearch(arr, x, lo, hi):
    if lo >= hi:
        return lo
    mid = (lo + hi) // 2
    if arr[mid] < x:
        return bsearch(arr, x, mid + 1, hi)
    return bsearch(arr, x, lo, mid)
"""
    result = analyze_function(source, "bsearch")
    # 排他分岐の 2 箇所の自己呼び出しは実行パスあたり a=1 →
    # T(n) = T(n/2) + O(1) → O(log n)
    assert result.complexity is ComplexityClass.LOGARITHMIC


def test_bisect_insort_is_quadratic():
    source = """
import bisect
def binary_insertion_sort(arr):
    result = []
    for x in arr:
        bisect.insort(result, x)
    return result
"""
    result = analyze_function(source, "binary_insertion_sort")
    # insort = 二分探索 O(log n) + シフト O(n) → 全体 O(n²)
    assert result.complexity is ComplexityClass.QUADRATIC


def test_bisect_search_only_is_linearithmic():
    source = """
from bisect import bisect_left
def count_positions(arr, queries):
    data = sorted(arr)
    out = []
    for q in queries:
        out.append(bisect_left(data, q))
    return out
"""
    result = analyze_function(source, "count_positions")
    assert result.complexity is ComplexityClass.LINEARITHMIC


def test_del_subscript_linear_cost():
    source = """
def drain(arr):
    data = list(arr)
    out = []
    for _ in range(len(data)):
        out.append(data[0])
        del data[0]
    return out
"""
    result = analyze_function(source, "drain")
    assert result.complexity is ComplexityClass.QUADRATIC


def test_unknown_result_has_low_confidence():
    # O(n² log n) は既知クラス外 → 不明 + 低信頼度
    source = """
def f(arr):
    n = len(arr)
    total = 0
    for i in range(n):
        for j in range(n):
            k = n
            while k > 1:
                k //= 2
                total += 1
    return total
"""
    result = analyze_function(source, "f")
    assert result.complexity is ComplexityClass.UNKNOWN
    assert result.confidence <= 0.5


class TestBestCase:
    def test_insertion_sort_best_case_linear(self, example_source):
        result = analyze_function(example_source("insertion_sort"), "insertion_sort")
        # 内側 while はデータ依存 (data[j] > key) → 最良ケース O(n)
        assert result.best_case is ComplexityClass.LINEAR
        assert any("最良ケース" in r for r in result.reasoning)

    def test_bubble_with_early_exit_best_case_linear(self):
        source = """
def bubble_early(arr):
    data = list(arr)
    n = len(data)
    for i in range(n):
        swapped = False
        for j in range(n - 1 - i):
            if data[j] > data[j + 1]:
                data[j], data[j + 1] = data[j + 1], data[j]
                swapped = True
        if not swapped:
            break
    return data
"""
        result = analyze_function(source, "bubble_early")
        assert result.complexity is ComplexityClass.QUADRATIC
        # 早期 break → 外側ループ 1 回 → 最良 O(n)
        assert result.best_case is ComplexityClass.LINEAR

    def test_selection_sort_has_no_better_best_case(self, example_source):
        result = analyze_function(example_source("selection_sort"), "selection_sort")
        # データ依存の打ち切りがない → 最良も O(n²) (best_case は None)
        assert result.best_case is None

    def test_merge_sort_has_no_better_best_case(self, example_source):
        result = analyze_function(example_source("merge_sort"), "merge_sort")
        assert result.best_case is None


def test_literature_complexities():
    from codeevaluator.analyzer.patterns import literature_complexities

    assert literature_complexities("挿入ソート") == ("O(n)", "O(n²)", "O(n²)")
    assert literature_complexities("クイックソート")[2] == "O(n²)"
    assert literature_complexities(None) is None
    assert literature_complexities("未知のソート") is None


def test_confidence_bounds(example_source):
    for name in ("bubble_sort", "merge_sort"):
        result = analyze_function(example_source(name), name)
        assert 0.3 <= result.confidence <= 0.9
