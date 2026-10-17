"""ユーザーコードの隔離実行基盤。

ユーザーが入力した Python コードを multiprocessing (spawn) の
子プロセスで exec し、Pipe 経由で計測コマンドをやり取りする。

- タイムアウト: join(timeout) 超過で terminate (確実な打ち切り)
- 正当性検査: 計測前に小入力で sorted() と結果を比較
- ワーカーは使い回し、タイムアウト時のみ作り直す

注意: これは暴走・無限ループからの保護であり、サンドボックスではない。
悪意あるコードの実行を防ぐ仕組みは持たないため、
信頼できるコードのみを解析すること。
"""

from __future__ import annotations

import multiprocessing as mp
import random
import time
from dataclasses import dataclass

#: 入力コードの上限 (バイト)
MAX_SOURCE_BYTES = 50 * 1024
#: 入力サイズのハードキャップ
MAX_INPUT_SIZE = 2 ** 16

_SEED = 42


def generate_input(size: int, kind: str, seed: int = _SEED) -> list[int]:
    """再現性のある固定シードでベンチマーク入力を生成する。"""
    rng = random.Random(seed + size)
    data = [rng.randint(0, size * 10) for _ in range(size)]
    if kind == "sorted":
        data.sort()
    elif kind == "reversed":
        data.sort(reverse=True)
    elif kind != "random":
        raise ValueError(f"不明な入力種別: {kind}")
    return data


class WorkerError(RuntimeError):
    """子プロセス内でユーザーコードが失敗したときの日本語例外。"""


@dataclass
class TimingReply:
    """1 回の計測コマンドに対する子プロセスの応答。"""

    elapsed: float | None = None  # 秒。None はエラー
    sorted_ok: bool | None = None  # 正当性検査コマンドの結果
    error: str | None = None


def _worker_main(conn, source: str, func_name: str) -> None:
    """子プロセスのメインループ (spawn のためトップレベル関数)。

    プロトコル: 親から ("check", size, kind) / ("time", size, kind) /
    ("quit",) を受け、TimingReply を返す。
    """
    import gc
    import sys

    sys.setrecursionlimit(max(1100, 2 * MAX_INPUT_SIZE + 100))

    namespace: dict = {}
    try:
        exec(compile(source, "<user_code>", "exec"), namespace)
        func = namespace[func_name]
    except Exception as e:  # noqa: BLE001 - ユーザーコードの任意の失敗を報告する
        conn.send(TimingReply(error=f"コードの読み込みに失敗: {type(e).__name__}: {e}"))
        conn.close()
        return
    conn.send(TimingReply())  # ロード成功の合図

    while True:
        try:
            msg = conn.recv()
        except (EOFError, OSError):
            break
        cmd = msg[0]
        if cmd == "quit":
            break
        _, size, kind = msg
        data = generate_input(size, kind)
        expected = sorted(data)
        arg = list(data)  # 毎回コピーして渡す (インプレース対策)
        try:
            if cmd == "check":
                returned = func(arg)
                result = returned if returned is not None else arg
                conn.send(TimingReply(sorted_ok=(list(result) == expected)))
            elif cmd == "time":
                gc.disable()
                try:
                    start = time.perf_counter()
                    func(arg)
                    elapsed = time.perf_counter() - start
                finally:
                    gc.enable()
                conn.send(TimingReply(elapsed=elapsed))
        except RecursionError:
            conn.send(TimingReply(error="再帰の深さ上限を超えました (RecursionError)"))
        except Exception as e:  # noqa: BLE001
            conn.send(TimingReply(error=f"実行時エラー: {type(e).__name__}: {e}"))
    conn.close()


class SortRunner:
    """子プロセスのワーカーを管理し、計測・検査コマンドを送るフロントエンド。"""

    def __init__(self, source: str, func_name: str):
        if len(source.encode("utf-8")) > MAX_SOURCE_BYTES:
            raise WorkerError(
                f"コードが大きすぎます (上限 {MAX_SOURCE_BYTES // 1024} KB)"
            )
        self.source = source
        self.func_name = func_name
        self._ctx = mp.get_context("spawn")
        self._proc = None
        self._conn = None

    # ---------- ライフサイクル ----------

    def start(self, timeout: float = 30.0) -> None:
        """ワーカーを起動してユーザーコードをロードする。"""
        parent_conn, child_conn = self._ctx.Pipe()
        self._proc = self._ctx.Process(
            target=_worker_main,
            args=(child_conn, self.source, self.func_name),
            daemon=True,
        )
        self._proc.start()
        child_conn.close()
        self._conn = parent_conn
        reply = self._recv(timeout)
        if reply.error:
            self.close()
            raise WorkerError(reply.error)

    def close(self) -> None:
        """ワーカーを終了する。"""
        if self._conn is not None:
            try:
                self._conn.send(("quit",))
            except (BrokenPipeError, OSError):
                pass
            self._conn.close()
            self._conn = None
        if self._proc is not None:
            self._proc.join(timeout=1.0)
            if self._proc.is_alive():
                self._proc.terminate()
                self._proc.join(timeout=1.0)
            self._proc = None

    def _kill(self) -> None:
        """タイムアウトしたワーカーを強制終了する。"""
        if self._proc is not None:
            self._proc.terminate()
            self._proc.join(timeout=1.0)
            self._proc = None
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> "SortRunner":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ---------- コマンド ----------

    def _recv(self, timeout: float) -> TimingReply:
        if self._conn is None:
            raise WorkerError("ワーカーが起動していません")
        if not self._conn.poll(timeout):
            self._kill()
            raise TimeoutError
        try:
            return self._conn.recv()
        except (EOFError, OSError) as e:
            self._kill()
            raise WorkerError(f"ワーカーが異常終了しました: {e}") from e

    def _ensure_alive(self) -> None:
        if self._proc is None or not self._proc.is_alive():
            self.start()

    def check_sorts_correctly(
        self, size: int = 32, kinds: tuple[str, ...] = ("random", "sorted", "reversed"),
        timeout: float = 10.0,
    ) -> tuple[bool, str | None]:
        """小入力でソートの正当性を検査する。戻り値: (OK, エラーメッセージ)。"""
        self._ensure_alive()
        for kind in kinds:
            try:
                self._conn.send(("check", size, kind))
                reply = self._recv(timeout)
            except TimeoutError:
                return False, f"正当性検査がタイムアウトしました (入力種別: {kind})"
            if reply.error:
                return False, reply.error
            if not reply.sorted_ok:
                return False, (
                    f"関数 '{self.func_name}' の出力がソート結果と一致しません "
                    f"(入力種別: {kind})"
                )
        return True, None

    def time_once(self, size: int, kind: str, timeout: float) -> float:
        """1 回の実行時間 (秒) を計測する。超過時は TimeoutError。"""
        if size > MAX_INPUT_SIZE:
            raise WorkerError(f"入力サイズが上限 {MAX_INPUT_SIZE} を超えています")
        self._ensure_alive()
        self._conn.send(("time", size, kind))
        reply = self._recv(timeout)
        if reply.error:
            raise WorkerError(reply.error)
        return reply.elapsed
