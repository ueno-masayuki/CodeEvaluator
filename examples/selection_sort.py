"""選択ソート: 最小値を選んで先頭と交換する O(n²) のソート。"""


def selection_sort(arr):
    """リストを昇順にソートして返す(非破壊)。"""
    data = list(arr)
    n = len(data)
    for i in range(n):
        min_idx = i
        for j in range(i + 1, n):
            if data[j] < data[min_idx]:
                min_idx = j
        data[i], data[min_idx] = data[min_idx], data[i]
    return data
