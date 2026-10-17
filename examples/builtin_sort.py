"""組み込みソート: Python の sorted() (Timsort, O(n log n)) のラッパー。"""


def builtin_sort(arr):
    """リストを昇順にソートして返す(非破壊)。"""
    return sorted(arr)
