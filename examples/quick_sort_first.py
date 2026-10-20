"""クイックソート (先頭要素ピボット版)。

平均 O(n log n)、最悪 O(n²)。ピボットに先頭要素を使うため、
**整列済み・逆順の入力がそのまま最悪ケースに該当**し、
ベンチマークの「整列済み」「逆順」入力で実測 O(n²) を観察できる。
(中央要素ピボット版 quick_sort.py では、これらの入力で最悪ケースは発生しない)
"""


def quick_sort_first(arr):
    """リストを昇順にソートして返す(非破壊)。"""
    if len(arr) <= 1:
        return list(arr)
    pivot = arr[0]
    less = [x for x in arr[1:] if x < pivot]
    equal = [x for x in arr if x == pivot]
    greater = [x for x in arr[1:] if x > pivot]
    return quick_sort_first(less) + equal + quick_sort_first(greater)
