"""挿入ソート: 整列済み部分に要素を挿入していく O(n²) のソート。"""


def insertion_sort(arr):
    """リストを昇順にソートして返す(非破壊)。"""
    data = list(arr)
    for i in range(1, len(data)):
        key = data[i]
        j = i - 1
        while j >= 0 and data[j] > key:
            data[j + 1] = data[j]
            j -= 1
        data[j + 1] = key
    return data
