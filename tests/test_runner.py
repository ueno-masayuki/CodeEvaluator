"""runner (隔離実行・タイムアウト・正当性検査) のテスト。"""

import pytest

from codeevaluator.runner import SortRunner, WorkerError, generate_input


class TestGenerateInput:
    def test_reproducible(self):
        assert generate_input(64, "random") == generate_input(64, "random")

    def test_kinds(self):
        data = generate_input(64, "sorted")
        assert data == sorted(data)
        data = generate_input(64, "reversed")
        assert data == sorted(data, reverse=True)

    def test_unknown_kind(self):
        with pytest.raises(ValueError, match="不明な入力種別"):
            generate_input(8, "zigzag")

    def test_median_killer_properties(self):
        data = generate_input(64, "median_killer")
        # 0..n-1 の順列で決定的
        assert sorted(data) == list(range(64))
        assert data == generate_input(64, "median_killer")
        # どの再帰段でも中央要素が部分列の最大値になっている
        # (中央ピボットの 3-way クイックソートで毎回 greater 側が空になる)
        part = data
        while len(part) > 1:
            mid = part[len(part) // 2]
            assert mid == max(part)
            part = [x for x in part if x < mid]


SORT_OK = """
def my_sort(arr):
    return sorted(arr)
"""

SORT_IN_PLACE = """
def my_sort(arr):
    arr.sort()
"""

NOT_A_SORT = """
def my_sort(arr):
    return arr
"""

INFINITE_LOOP = """
def my_sort(arr):
    while True:
        pass
"""

BROKEN_LOAD = """
import nonexistent_module_xyz
def my_sort(arr):
    return sorted(arr)
"""


class TestSortRunner:
    def test_correct_sort_passes(self):
        with SortRunner(SORT_OK, "my_sort") as runner:
            ok, err = runner.check_sorts_correctly()
            assert ok, err

    def test_in_place_sort_passes(self):
        with SortRunner(SORT_IN_PLACE, "my_sort") as runner:
            ok, err = runner.check_sorts_correctly()
            assert ok, err

    def test_identity_function_fails(self):
        with SortRunner(NOT_A_SORT, "my_sort") as runner:
            ok, err = runner.check_sorts_correctly()
            assert not ok
            assert "一致しません" in err

    def test_timing_returns_positive(self):
        with SortRunner(SORT_OK, "my_sort") as runner:
            elapsed = runner.time_once(256, "random", timeout=10.0)
            assert elapsed > 0

    def test_infinite_loop_times_out(self):
        with SortRunner(INFINITE_LOOP, "my_sort") as runner:
            with pytest.raises(TimeoutError):
                runner.time_once(16, "random", timeout=1.0)

    def test_broken_code_raises_japanese_error(self):
        runner = SortRunner(BROKEN_LOAD, "my_sort")
        with pytest.raises(WorkerError, match="読み込みに失敗"):
            runner.start()

    def test_source_size_cap(self):
        big = "# x\n" * 20000  # > 50KB
        with pytest.raises(WorkerError, match="大きすぎます"):
            SortRunner(big, "f")
