"""実測結果の log-log グラフ描画 (matplotlib)。

- 実測点 (中央値) を入力種別ごとに散布し、min/max をエラーバーで示す
- 最良フィット曲線を実線、次点 2 モデルを破線 (無彩色) で重ねる
- 日本語フォントがない環境では軸ラベルを英語に自動フォールバックする
"""

from __future__ import annotations

import math

import matplotlib
import numpy as np

matplotlib.use("Agg")  # GUI 不要のバックエンド (Streamlit は Figure を受け取る)

from matplotlib import font_manager
from matplotlib.figure import Figure

from codeevaluator.models import EmpiricalResult, Measurement

#: 日本語表示に使えるフォントの候補 (優先順)
_CJK_FONTS = (
    "Noto Sans CJK JP",
    "Noto Sans JP",
    "IPAexGothic",
    "IPAGothic",
    "TakaoGothic",
    "Yu Gothic",
    "Hiragino Sans",
    "MS Gothic",
)

# チャート配色 (検証済みリファレンスパレット・ライトモード)
_SURFACE = "#fcfcfb"
_INK = "#0b0b0b"
_MUTED = "#898781"
_GRID = "#e1e0d9"
_BASELINE = "#c3c2b7"
_SERIES = {  # 入力種別 → カテゴリカルスロット (固定順)
    "random": "#2a78d6",   # blue
    "sorted": "#1baf7a",   # aqua
    "reversed": "#eb6834", # orange
}
_KIND_JA = {"random": "ランダム入力", "sorted": "整列済み入力", "reversed": "逆順入力"}
_KIND_EN = {"random": "random input", "sorted": "sorted input", "reversed": "reversed input"}


def _find_cjk_font() -> str | None:
    """利用可能な日本語フォント名を返す。なければ None。"""
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in _CJK_FONTS:
        if name in available:
            return name
    return None


def make_loglog_figure(result: EmpiricalResult, fit_kind: str = "random") -> Figure:
    """入力サイズ対実行時間の log-log グラフを作る。"""
    cjk = _find_cjk_font()
    ja = cjk is not None
    font_family = [cjk, "DejaVu Sans"] if cjk else ["DejaVu Sans"]

    fig = Figure(figsize=(7.5, 5.0), dpi=110, facecolor=_SURFACE)
    ax = fig.add_subplot(111, facecolor=_SURFACE)

    # 入力種別ごとの実測点
    by_kind: dict[str, list[Measurement]] = {}
    for m in result.measurements:
        by_kind.setdefault(m.input_kind, []).append(m)

    for kind, ms in by_kind.items():
        ms = sorted(ms, key=lambda m: m.size)
        sizes = np.array([m.size for m in ms], dtype=float)
        medians = np.array([m.median_time for m in ms], dtype=float)
        lower = medians - np.array([min(m.times) for m in ms])
        upper = np.array([max(m.times) for m in ms]) - medians
        color = _SERIES.get(kind, _MUTED)
        label = (_KIND_JA if ja else _KIND_EN).get(kind, kind)
        ax.errorbar(
            sizes, medians, yerr=[lower, upper],
            fmt="o", markersize=7, color=color, ecolor=color,
            elinewidth=1.2, capsize=3, label=label, zorder=3,
            markeredgecolor=_SURFACE, markeredgewidth=0.8,
        )

    # フィット曲線: 種別ごとの最良フィットを種別色の実線で描く
    kind_results = result.kind_results or (
        {fit_kind: None} if result.fits else {}
    )
    for kind in kind_results:
        summary = result.kind_results.get(kind)
        fits = summary.fits if summary is not None else result.fits
        best = fits[0] if fits else None
        if best is None:
            continue
        points = sorted(
            (m for m in result.measurements if m.input_kind == kind),
            key=lambda m: m.size,
        )
        if not points:
            continue
        xs = np.geomspace(points[0].size, points[-1].size, 128)
        kind_label = (_KIND_JA if ja else _KIND_EN).get(kind, kind)
        ax.plot(
            xs, best.scale * best.complexity.model(xs),
            color=_SERIES.get(kind, _INK), linewidth=2.0, zorder=2,
            label=(
                f"{kind_label}: {best.complexity.label()} (R²={best.r_squared:.3f})"
            ),
        )
        # 次点 2 モデル: 主要種別のみ、無彩色の破線 (文脈情報)
        if kind == fit_kind:
            for alt, style in zip(fits[1:3], ("--", ":")):
                ax.plot(
                    xs, alt.scale * alt.complexity.model(xs),
                    color=_MUTED, linewidth=1.4, linestyle=style, zorder=1,
                    label=(
                        f"次点 {alt.complexity.label()}" if ja
                        else f"runner-up {alt.complexity.label()}"
                    ),
                )

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    if ja:
        ax.set_xlabel("入力サイズ n", fontfamily=font_family, color=_INK)
        ax.set_ylabel("実行時間 [秒]", fontfamily=font_family, color=_INK)
        title = f"実測: {result.summary_label()}"
    else:
        ax.set_xlabel("input size n", color=_INK)
        ax.set_ylabel("time [s]", color=_INK)
        title = f"measured: {result.summary_label()}"
    if not math.isnan(result.exponent):
        ax.set_title(title, fontfamily=font_family, color=_INK, fontsize=11)

    # 控えめなグリッドと軸 (ヘアライン)
    ax.grid(True, which="major", color=_GRID, linewidth=0.7)
    ax.grid(False, which="minor")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(_BASELINE)
    ax.tick_params(colors=_MUTED, labelsize=9)

    legend = ax.legend(
        loc="upper left", fontsize=9, frameon=False,
        prop={"family": font_family, "size": 9},
    )
    for text in legend.get_texts():
        text.set_color(_INK)

    fig.tight_layout()
    return fig
