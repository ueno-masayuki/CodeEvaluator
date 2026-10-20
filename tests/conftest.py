"""テスト共通フィクスチャ: examples/ のソースコード読み込み。"""

from pathlib import Path

import pytest

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"

#: (ファイル名ステム, 期待される関数名)
EXAMPLE_FUNCTIONS = {
    "bubble_sort": "bubble_sort",
    "insertion_sort": "insertion_sort",
    "selection_sort": "selection_sort",
    "merge_sort": "merge_sort",
    "quick_sort": "quick_sort",
    "quick_sort_first": "quick_sort_first",
    "heap_sort": "heap_sort",
    "builtin_sort": "builtin_sort",
}


@pytest.fixture
def example_source():
    """examples/<name>.py のソース文字列を返すローダー。"""

    def _load(name: str) -> str:
        return (EXAMPLES_DIR / f"{name}.py").read_text(encoding="utf-8")

    return _load
