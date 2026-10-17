"""既知アルゴリズム認識 (patterns) のテスト。"""

import ast

import pytest

from codeevaluator.analyzer import patterns

EXPECTED_NAMES = {
    "bubble_sort": "バブルソート",
    "insertion_sort": "挿入ソート",
    "selection_sort": "選択ソート",
    "merge_sort": "マージソート",
    "quick_sort": "クイックソート",
    "heap_sort": "ヒープソート",
    "builtin_sort": "組み込みソート (Timsort)",
}


def _func_node(source: str, name: str) -> ast.FunctionDef:
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} が見つからない")


@pytest.mark.parametrize("name,expected", sorted(EXPECTED_NAMES.items()))
def test_examples_recognized(example_source, name, expected):
    func = _func_node(example_source(name), name)
    assert patterns.recognize(func) == expected


def test_unknown_structure_returns_none():
    source = """
def mystery(arr):
    return [x * 2 for x in arr]
"""
    func = _func_node(source, "mystery")
    assert patterns.recognize(func) is None
