"""バブルソート: 隣接要素の交換を繰り返す O(n²) のソート。"""


def bubble_sort(arr):
    """リストを昇順にソートして返す(非破壊)。"""
    data = list(arr)
    n = len(data)
    for i in range(n):
        for j in range(n - 1 - i):
            if data[j] > data[j + 1]:
                data[j], data[j + 1] = data[j + 1], data[j]
    return data
