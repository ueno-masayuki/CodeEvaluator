"""クイックソート: ピボットによる分割統治。平均 O(n log n)、最悪 O(n²)。"""


def quick_sort(arr):
    """リストを昇順にソートして返す(非破壊)。"""
    if len(arr) <= 1:
        return list(arr)
    pivot = arr[len(arr) // 2]
    less = [x for x in arr if x < pivot]
    equal = [x for x in arr if x == pivot]
    greater = [x for x in arr if x > pivot]
    return quick_sort(less) + equal + quick_sort(greater)
