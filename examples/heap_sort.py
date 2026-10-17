"""ヒープソート: heapq を利用した O(n log n) のソート。"""

import heapq


def heap_sort(arr):
    """リストを昇順にソートして返す(非破壊)。"""
    heap = list(arr)
    heapq.heapify(heap)
    result = []
    while heap:
        result.append(heapq.heappop(heap))
    return result
