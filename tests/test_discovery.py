"""discovery (ソート関数検出) のテスト。"""

import pytest

from codeevaluator.discovery import SyntaxErrorJa, find_sort_functions
from tests.conftest import EXAMPLE_FUNCTIONS


@pytest.mark.parametrize("name,func", sorted(EXAMPLE_FUNCTIONS.items()))
def test_examples_detected_as_top_candidate(example_source, name, func):
    candidates = find_sort_functions(example_source(name))
    assert candidates, f"{name} で候補が見つからない"
    assert candidates[0].name == func
    assert candidates[0].confidence >= 0.3
    assert candidates[0].reasons  # 日本語根拠が入っている


def test_non_sort_code_scores_low():
    source = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)
"""
    candidates = find_sort_functions(source)
    # 候補ゼロ、またはスコアが低い
    assert all(c.confidence < 0.5 for c in candidates)


def test_in_place_sort_detected():
    source = """
def my_sort(arr):
    for i in range(len(arr)):
        for j in range(len(arr) - 1 - i):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
"""
    candidates = find_sort_functions(source)
    assert candidates[0].name == "my_sort"
    assert candidates[0].mutates_in_place is True


def test_syntax_error_raises_japanese_message():
    with pytest.raises(SyntaxErrorJa, match="構文エラー"):
        find_sort_functions("def broken(:\n    pass")


def test_no_arg_function_excluded():
    source = """
def sort():
    return []
"""
    assert find_sort_functions(source) == []


def test_multiple_functions_ranked():
    source = """
def helper(x):
    return x + 1

def bubble_sort(arr):
    data = list(arr)
    for i in range(len(data)):
        for j in range(len(data) - 1):
            if data[j] > data[j + 1]:
                data[j], data[j + 1] = data[j + 1], data[j]
    return data
"""
    candidates = find_sort_functions(source)
    assert candidates[0].name == "bubble_sort"
