"""実測ベンチマークによる実際的な計算量オーダーの推定。

入力サイズを幾何級数で変えながらユーザー関数の実行時間を計測し、
fitting.fit_measurements で候補モデルへのフィットを行う。
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Callable

from codeevaluator.fitting import fit_measurements
from codeevaluator.models import EmpiricalResult, KindFitSummary, Measurement
from codeevaluator.runner import SortRunner, WorkerError

#: 進捗コールバック型: (完了数, 全体数, 日本語メッセージ)
ProgressCallback = Callable[[int, int, str], None]


@dataclass
class BenchmarkConfig:
    """ベンチマーク設定。"""

    min_exp: int = 8                 # 最小サイズ 2^min_exp
    max_exp: int = 13                # 最大サイズ 2^max_exp
    repeats: int = 5                 # サイズごとの反復回数
    per_run_timeout: float = 5.0     # 1 回の実行のタイムアウト (秒)
    input_kinds: tuple[str, ...] = ("random",)  # 先頭の種別を総合判定のフィットに使う
    extra_kinds: tuple[str, ...] = ()           # 追加種別 (sorted/reversed)。種別ごとにフィットも行う

    def sizes(self) -> list[int]:
        return [2 ** e for e in range(self.min_exp, self.max_exp + 1)]


def run_benchmark(
    source: str,
    func_name: str,
    config: BenchmarkConfig | None = None,
    progress_callback: ProgressCallback | None = None,
) -> EmpiricalResult:
    """ユーザー関数のベンチマークを実行し、実測計算量を推定する。"""
    config = config or BenchmarkConfig()
    notes: list[str] = []
    measurements: list[Measurement] = []
    timed_out = False

    all_kinds = tuple(dict.fromkeys(config.input_kinds + config.extra_kinds))
    fit_kind = all_kinds[0]
    sizes = config.sizes()
    total_steps = len(sizes) * len(all_kinds)
    done = 0

    def _progress(message: str) -> None:
        if progress_callback is not None:
            progress_callback(done, total_steps, message)

    runner = SortRunner(source, func_name)
    try:
        _progress("ワーカープロセスを起動中…")
        runner.start()

        # 正当性検査: ソートしない関数の計測は無意味
        _progress("ソートの正当性を検査中…")
        ok, err = runner.check_sorts_correctly()
        if not ok:
            notes.append(err or "正当性検査に失敗しました")
            return _finalize(
                func_name, measurements, fit_kind, notes,
                timed_out=False, correctness_ok=False,
            )

        for kind in all_kinds:
            prev_median: float | None = None
            for size in sizes:
                # 予算則: 前サイズの中央値から ×4 (傾き 2 の外挿) で予測し、
                # タイムアウト超過が見込まれるなら事前に打ち切る
                if prev_median is not None and prev_median * 4 > config.per_run_timeout:
                    notes.append(
                        f"サイズ {size} 以降は予測実行時間がタイムアウト "
                        f"({config.per_run_timeout:.0f} 秒) を超えるため打ち切りました "
                        f"(入力種別: {kind})"
                    )
                    timed_out = True
                    break
                _progress(f"n={size} ({_kind_ja(kind)}) を計測中…")
                times: list[float] = []
                aborted = False
                for _ in range(config.repeats):
                    try:
                        times.append(
                            runner.time_once(size, kind, config.per_run_timeout)
                        )
                    except TimeoutError:
                        notes.append(
                            f"n={size} ({_kind_ja(kind)}) の実行が "
                            f"{config.per_run_timeout:.0f} 秒を超えたため打ち切りました"
                        )
                        timed_out = True
                        aborted = True
                        break
                if times:
                    median = statistics.median(times)
                    measurements.append(
                        Measurement(
                            size=size,
                            times=times,
                            median_time=median,
                            input_kind=kind,
                        )
                    )
                    prev_median = median
                done += 1
                _progress(f"n={size} ({_kind_ja(kind)}) 完了")
                if aborted:
                    break
    except WorkerError as e:
        notes.append(str(e))
        return _finalize(
            func_name, measurements, fit_kind, notes,
            timed_out=timed_out, correctness_ok=False,
        )
    finally:
        runner.close()

    _progress("曲線フィットを計算中…")
    result = _finalize(
        func_name, measurements, fit_kind, notes,
        timed_out=timed_out, correctness_ok=True,
    )
    done = total_steps
    _progress("完了")
    return result


def _kind_ja(kind: str) -> str:
    return {"random": "ランダム", "sorted": "整列済み", "reversed": "逆順"}.get(
        kind, kind
    )


def _finalize(
    func_name: str,
    measurements: list[Measurement],
    fit_kind: str,
    notes: list[str],
    timed_out: bool,
    correctness_ok: bool,
) -> EmpiricalResult:
    """入力種別ごとにフィットを実行して EmpiricalResult に詰める。"""
    notes = list(notes)
    kind_results: dict[str, KindFitSummary] = {}
    # 測定に現れた順で種別ごとにフィットする (fit_kind を先頭に)
    kinds = list(dict.fromkeys(
        [fit_kind] + [m.input_kind for m in measurements]
    ))
    for kind in kinds:
        points = [m for m in measurements if m.input_kind == kind]
        if not points:
            continue
        fits, exponent, stderr, fit_notes = fit_measurements(
            [m.size for m in points],
            [m.median_time for m in points],
        )
        kind_results[kind] = KindFitSummary(
            input_kind=kind,
            fits=fits,
            best_fit=fits[0] if fits else None,
            exponent=exponent,
            exponent_stderr=stderr,
        )
        suffix = "" if kind == fit_kind else f" ({_kind_ja(kind)}入力)"
        notes.extend(n + suffix for n in fit_notes)

    primary = kind_results.get(
        fit_kind,
        KindFitSummary(fit_kind, [], None, math.nan, math.nan),
    )
    _append_adaptivity_note(measurements, fit_kind, notes)
    return EmpiricalResult(
        function_name=func_name,
        measurements=measurements,
        fits=primary.fits,
        best_fit=primary.best_fit,
        exponent=primary.exponent,
        exponent_stderr=primary.exponent_stderr,
        timed_out=timed_out,
        correctness_ok=correctness_ok,
        notes=notes,
        kind_results=kind_results,
    )


def _append_adaptivity_note(
    measurements: list[Measurement], fit_kind: str, notes: list[str]
) -> None:
    """整列済み入力が大幅に速い場合、適応的アルゴリズムの注記を加える。"""
    by_kind: dict[str, dict[int, float]] = {}
    for m in measurements:
        by_kind.setdefault(m.input_kind, {})[m.size] = m.median_time
    base = by_kind.get(fit_kind, {})
    for kind, label in (("sorted", "整列済み"), ("reversed", "逆順")):
        other = by_kind.get(kind, {})
        common = [s for s in base if s in other and base[s] > 0]
        if len(common) < 2:
            continue
        ratios = [other[s] / base[s] for s in common]
        geo_mean = math.exp(sum(math.log(r) for r in ratios) / len(ratios))
        if geo_mean < 0.3:
            notes.append(
                f"{label}入力はランダム入力より大幅に速い (平均 {geo_mean:.2f} 倍) — "
                "適応的なアルゴリズムの可能性があります (例: 挿入ソートの最良ケース O(n))"
            )
        elif geo_mean > 3.0:
            notes.append(
                f"{label}入力はランダム入力より大幅に遅い (平均 {geo_mean:.1f} 倍) — "
                "入力順に敏感なアルゴリズムの可能性があります (例: クイックソートの最悪ケース)"
            )
