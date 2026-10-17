"""実測データの候補計算量モデルへのフィットとモデル選択。

各候補クラス C について t ≈ c · f_C(n) を log 空間で解く:
    log t = log c + log f_C(n)
log c は閉形式 mean(log t − log f_C(n)) で求まる (1 パラメータ)。
log 空間の残差から R² と AIC を計算し、AIC 最小のモデルを採用する。

併せて log-log 回帰 (np.polyfit) の傾きから実測指数 p を求め、
「O(n^1.98) ≈ O(n²)」のような表示に使う。
"""

from __future__ import annotations

import math

import numpy as np

from codeevaluator.models import FIT_CANDIDATES, ComplexityClass, FitResult

#: フィットに最低限必要な測定点数
MIN_POINTS = 4

#: AIC 差がこの値未満なら「判別が不確実」と注記する
AIC_AMBIGUITY_THRESHOLD = 2.0


def fit_measurements(
    sizes: list[int], times: list[float]
) -> tuple[list[FitResult], float, float, list[str]]:
    """測定値を候補モデルにフィットする。

    戻り値: (全候補のフィット結果 (AIC 昇順), 実測指数, 指数の標準誤差, 日本語注記)
    """
    notes: list[str] = []
    pairs = [
        (s, t) for s, t in zip(sizes, times) if t > 0 and s > 1
    ]
    if len(pairs) < MIN_POINTS:
        notes.append(
            f"有効な測定点が {len(pairs)} 点しかないため判定できません "
            f"(最低 {MIN_POINTS} 点必要)"
        )
        return [], math.nan, math.nan, notes

    n = np.array([p[0] for p in pairs], dtype=float)
    t = np.array([p[1] for p in pairs], dtype=float)
    log_t = np.log(t)
    k = len(pairs)

    fits: list[FitResult] = []
    tss = float(np.sum((log_t - log_t.mean()) ** 2))
    for cls in FIT_CANDIDATES:
        log_f = np.log(cls.model(n))
        log_c = float(np.mean(log_t - log_f))
        residuals = log_t - (log_c + log_f)
        rss = float(np.sum(residuals ** 2))
        r_squared = 1.0 - rss / tss if tss > 0 else 0.0
        # AIC = k·ln(RSS/k) + 2·params (params=1 で全モデル共通)
        aic = k * math.log(max(rss, 1e-300) / k) + 2.0
        fits.append(
            FitResult(
                complexity=cls,
                scale=math.exp(log_c),
                r_squared=r_squared,
                aic=aic,
            )
        )

    fits.sort(key=lambda f: f.aic)
    if len(fits) >= 2 and fits[1].aic - fits[0].aic < AIC_AMBIGUITY_THRESHOLD:
        notes.append(
            f"{fits[0].complexity.label()} と {fits[1].complexity.label()} の判別が"
            "不確実です (AIC 差 < 2)。測定点を増やすかサイズ範囲を拡大してください。"
        )

    # log-log 回帰による実測指数
    try:
        coeffs, cov = np.polyfit(np.log(n), log_t, 1, cov=True)
        exponent = float(coeffs[0])
        stderr = float(np.sqrt(cov[0, 0]))
    except (np.linalg.LinAlgError, ValueError):
        exponent, stderr = math.nan, math.nan
        notes.append("log-log 回帰による指数推定に失敗しました")

    return fits, exponent, stderr, notes


def classes_agree(
    theoretical: ComplexityClass,
    empirical: ComplexityClass,
    worst_case: ComplexityClass | None = None,
) -> bool:
    """理論と実測の計算量クラスが整合しているとみなせるか。

    - 完全一致は整合
    - n log n と n log²n のような隣接 log 次数も整合扱い
    - worst_case がある場合 (クイックソート等) は平均/最悪のどちらかと一致で整合
    """
    if theoretical is empirical:
        return True
    if worst_case is not None and worst_case is empirical:
        return True
    near = {
        ComplexityClass.LINEARITHMIC: {ComplexityClass.N_LOG2N, ComplexityClass.LINEAR},
        ComplexityClass.N_LOG2N: {ComplexityClass.LINEARITHMIC},
        ComplexityClass.LINEAR: {ComplexityClass.LINEARITHMIC},
    }
    return empirical in near.get(theoretical, set())
