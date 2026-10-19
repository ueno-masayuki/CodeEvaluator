"""記号的ループ解析 (symbolic) のテスト。"""

from codeevaluator.analyzer.static import analyze_function
from codeevaluator.models import ComplexityClass

SHELL_SORT = """
def shell_sort(arr):
    data = list(arr)
    n = len(data)
    gap = n // 2
    while gap > 0:
        for i in range(gap, n):
            tmp = data[i]
            j = i
            while j >= gap and data[j - gap] > tmp:
                data[j] = data[j - gap]
                j -= gap
            data[j] = tmp
        gap //= 2
    return data
"""

COMB_SORT = """
def comb_sort(arr):
    data = list(arr)
    n = len(data)
    gap = n
    swapped = True
    while gap > 1 or swapped:
        gap = max(1, int(gap / 1.3))
        swapped = False
        for i in range(n - gap):
            if data[i] > data[i + gap]:
                data[i], data[i + gap] = data[i + gap], data[i]
                swapped = True
    return data
"""


def test_shell_sort_quadratic():
    """gap 半減列のシェルソート: Σ n·(n/gap) = O(n²) (従来は O(n² log n) → 不明)。"""
    result = analyze_function(SHELL_SORT, "shell_sort")
    assert result.complexity is ComplexityClass.QUADRATIC
    assert any("記号的ループ解析" in r for r in result.reasoning)
    # 最良ケース: データ依存の内側 while が走らなければ O(n log n)
    assert result.best_case is ComplexityClass.LINEARITHMIC


def test_comb_sort_linearithmic_with_assumption():
    """比 1.3 の幾何減少 gap: Σ (n - gap) = O(n log n)。swapped は仮定として注記。"""
    result = analyze_function(COMB_SORT, "comb_sort")
    assert result.complexity is ComplexityClass.LINEARITHMIC
    assert any("記号的ループ解析" in r for r in result.reasoning)
    assert any("swapped" in r for r in result.reasoning)
    assert result.confidence < 0.9  # 仮定を置いたぶん信頼度が下がる


def test_simple_halving_loop_still_logarithmic():
    """依存する内側ループがない単純な半減 while は従来どおり O(log n)。"""
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


def test_doubling_outer_with_dependent_inner():
    """倍化する幅 w で毎回 n/w 個のブロックを n 要素ずつ処理 → O(n log n) 型。

    ボトムアップマージソートの骨格: width を倍化しつつ全要素を走査する。
    """
    source = """
def bottom_up(arr):
    data = list(arr)
    n = len(data)
    width = 1
    while width < n:
        for i in range(0, n):
            data[i] = data[i]
        width *= 2
    return data
"""
    result = analyze_function(source, "bottom_up")
    assert result.complexity is ComplexityClass.LINEARITHMIC
