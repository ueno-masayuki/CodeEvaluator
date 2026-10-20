"""計算量クラスと解析結果のデータ構造。

本モジュールは全モジュールが依存する基盤であり、
- ComplexityClass: 計算量クラスの列挙型(表示ラベルとモデル関数を持つ)
- ComplexityTerm: 静的解析内部で使う (多項式次数, log次数) の合成表現
- 各種解析結果の dataclass
を提供する。
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass, field
from typing import ClassVar


class ComplexityClass(enum.Enum):
    """計算量クラス。

    各メンバは (キー, 表示ラベル) を値に持ち、model() で
    フィッティング・プロットに使うモデル関数 f(n) を返す。
    """

    CONSTANT = ("constant", "O(1)")
    LOGARITHMIC = ("logarithmic", "O(log n)")
    LINEAR = ("linear", "O(n)")
    LINEARITHMIC = ("linearithmic", "O(n log n)")
    N_LOG2N = ("n_log2n", "O(n log²n)")
    QUADRATIC = ("quadratic", "O(n²)")
    CUBIC = ("cubic", "O(n³)")
    EXPONENTIAL = ("exponential", "O(2^n)")
    UNKNOWN = ("unknown", "不明")

    def label(self) -> str:
        """表示用ラベル (例: "O(n log n)") を返す。"""
        return self.value[1]

    def model(self, n):
        """モデル関数 f(n) を numpy 配列対応で評価する。

        フィッティングでは t ≈ c * f(n) の f にあたる。
        UNKNOWN / EXPONENTIAL はフィット対象外のため呼び出しは誤り。
        """
        import numpy as np

        n = np.asarray(n, dtype=float)
        if self is ComplexityClass.CONSTANT:
            return np.ones_like(n)
        if self is ComplexityClass.LOGARITHMIC:
            return np.log2(n)
        if self is ComplexityClass.LINEAR:
            return n
        if self is ComplexityClass.LINEARITHMIC:
            return n * np.log2(n)
        if self is ComplexityClass.N_LOG2N:
            return n * np.log2(n) ** 2
        if self is ComplexityClass.QUADRATIC:
            return n ** 2
        if self is ComplexityClass.CUBIC:
            return n ** 3
        raise ValueError(f"{self} はモデル関数を持ちません")


#: 実測フィッティングで比較する候補クラス(ソートアルゴリズムで現実的な範囲)
FIT_CANDIDATES: tuple[ComplexityClass, ...] = (
    ComplexityClass.LINEAR,
    ComplexityClass.LINEARITHMIC,
    ComplexityClass.N_LOG2N,
    ComplexityClass.QUADRATIC,
    ComplexityClass.CUBIC,
)


@dataclass(frozen=True, order=True)
class ComplexityTerm:
    """静的解析内部の計算量表現: n^poly * (log n)^log を表す。

    直列合成は max、入れ子(ループ内)は積(次数の和)で合成する。
    order=True により (poly, log) の辞書式比較で大小を比べられる。
    """

    poly: int = 0
    log: int = 0

    # クラス定数 ZERO / N / LOG_N / N_LOG_N はモジュール末尾で設定する
    ZERO: ClassVar["ComplexityTerm"]
    N: ClassVar["ComplexityTerm"]
    LOG_N: ClassVar["ComplexityTerm"]
    N_LOG_N: ClassVar["ComplexityTerm"]

    def times(self, other: "ComplexityTerm") -> "ComplexityTerm":
        """入れ子合成(積): 次数を加算する。"""
        return ComplexityTerm(self.poly + other.poly, self.log + other.log)

    def to_class(self) -> ComplexityClass:
        """enum の計算量クラスへ写像する。該当なしは UNKNOWN。"""
        mapping = {
            (0, 0): ComplexityClass.CONSTANT,
            (0, 1): ComplexityClass.LOGARITHMIC,
            (1, 0): ComplexityClass.LINEAR,
            (1, 1): ComplexityClass.LINEARITHMIC,
            (1, 2): ComplexityClass.N_LOG2N,
            (2, 0): ComplexityClass.QUADRATIC,
            (3, 0): ComplexityClass.CUBIC,
        }
        if (self.poly, self.log) in mapping:
            return mapping[(self.poly, self.log)]
        # 該当しない高次項は最も近い既知クラスへ丸めず UNKNOWN とする
        return ComplexityClass.UNKNOWN

    def label(self) -> str:
        """内部表現の表示ラベル (例: "O(n² log n)")。"""
        known = self.to_class()
        if known is not ComplexityClass.UNKNOWN:
            return known.label()
        parts = []
        if self.poly == 1:
            parts.append("n")
        elif self.poly >= 2:
            sup = {2: "²", 3: "³"}.get(self.poly, f"^{self.poly}")
            parts.append(f"n{sup}")
        if self.log == 1:
            parts.append("log n")
        elif self.log >= 2:
            parts.append(f"log^{self.log} n")
        return "O(" + " ".join(parts) + ")" if parts else "O(1)"


# 定数ショートカット(クラス変数の後付け。frozen はインスタンスにのみ効く)
ComplexityTerm.ZERO = ComplexityTerm(0, 0)
ComplexityTerm.N = ComplexityTerm(1, 0)
ComplexityTerm.LOG_N = ComplexityTerm(0, 1)
ComplexityTerm.N_LOG_N = ComplexityTerm(1, 1)


@dataclass(frozen=True)
class SortFunctionInfo:
    """コード中から検出した「ソート関数らしい」関数の情報。"""

    name: str
    source: str
    lineno: int
    confidence: float
    reasons: list[str] = field(default_factory=list)
    mutates_in_place: bool = False


@dataclass(frozen=True)
class StaticAnalysisResult:
    """AST 静的解析による理論計算量の判定結果。"""

    function_name: str
    complexity: ComplexityClass
    confidence: float
    reasoning: list[str] = field(default_factory=list)
    max_loop_depth: int = 0
    has_recursion: bool = False
    recursion_kind: str | None = None  # "linear" | "divide_halving" | "divide_partition"
    recognized_algorithm: str | None = None
    worst_case: ComplexityClass | None = None  # 平均と最悪が異なる場合のみ(クイックソート等)
    best_case: ComplexityClass | None = None   # 平均と最良が異なる場合のみ(挿入ソート等)
    pivot_kind: str | None = None  # "first" | "last" | "middle" | "random" (パーティション分割時)


@dataclass(frozen=True)
class Measurement:
    """1つの入力サイズ・入力種別に対する実測結果。"""

    size: int
    times: list[float]
    median_time: float
    input_kind: str  # "random" | "sorted" | "reversed"


@dataclass(frozen=True)
class FitResult:
    """1つの候補計算量クラスへのフィット結果。"""

    complexity: ComplexityClass
    scale: float
    r_squared: float
    aic: float


def _format_summary(best_fit: FitResult | None, exponent: float) -> str:
    """「O(n^1.98) ≈ O(n²)」形式の表示文字列を作る。"""
    if best_fit is None:
        return "判定不能"
    if math.isnan(exponent):
        return best_fit.complexity.label()
    return f"O(n^{exponent:.2f}) ≈ {best_fit.complexity.label()}"


@dataclass(frozen=True)
class KindFitSummary:
    """1つの入力種別 (random / sorted / reversed) に対するフィット結果一式。"""

    input_kind: str
    fits: list[FitResult]            # AIC 昇順
    best_fit: FitResult | None
    exponent: float
    exponent_stderr: float

    def summary_label(self) -> str:
        """「O(n^1.98) ≈ O(n²)」形式の表示文字列を返す。"""
        return _format_summary(self.best_fit, self.exponent)


@dataclass(frozen=True)
class EmpiricalResult:
    """実測ベンチマークと曲線フィットの総合結果。"""

    function_name: str
    measurements: list[Measurement]
    fits: list[FitResult]
    best_fit: FitResult | None
    exponent: float
    exponent_stderr: float
    timed_out: bool = False
    correctness_ok: bool = True
    notes: list[str] = field(default_factory=list)
    # 入力種別ごとのフィット結果。トップレベルの fits / best_fit / exponent は
    # 主要種別 (通常 random) の値を保持する (後方互換・総合判定用)。
    kind_results: dict[str, KindFitSummary] = field(default_factory=dict)

    def summary_label(self) -> str:
        """「O(n^1.98) ≈ O(n²)」形式の表示文字列を返す。"""
        return _format_summary(self.best_fit, self.exponent)
