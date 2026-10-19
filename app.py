"""CodeEvaluator — ソートアルゴリズム計算量判定の Streamlit アプリ。

起動: streamlit run app.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# editable install なしでも動くフォールバック
_SRC = Path(__file__).resolve().parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from codeevaluator.analyzer.empirical import BenchmarkConfig, run_benchmark
from codeevaluator.analyzer.patterns import literature_complexities
from codeevaluator.analyzer.static import analyze_function
from codeevaluator.discovery import SyntaxErrorJa, find_sort_functions
from codeevaluator.fitting import classes_agree
from codeevaluator.models import ComplexityClass
from codeevaluator.plotting import make_loglog_figure

EXAMPLES_DIR = Path(__file__).resolve().parent / "examples"


def main() -> None:
    """アプリ本体 (spawn の子プロセスでの再実行を避けるため関数化)。"""

    st.set_page_config(page_title="CodeEvaluator — ソート計算量判定", layout="wide")

    st.title("CodeEvaluator — ソートアルゴリズム計算量判定")
    st.caption(
        "Python のソートプログラムを解析し、理論的な計算量オーダー (AST 静的解析) と "
        "実際的な計算量オーダー (実測 + 曲線フィット) を判定します。"
    )

    # ---------- サイドバー ----------

    with st.sidebar:
        st.header("入力方法")
        input_mode = st.radio(
            "ソースコードの入力方法",
            ("コード貼り付け", ".py ファイルをアップロード", "サンプルから選択"),
            label_visibility="collapsed",
        )

        st.header("ベンチマーク設定")
        exp_range = st.select_slider(
            "入力サイズの範囲 (2 の冪)",
            options=list(range(6, 15)),
            value=(8, 12),
            format_func=lambda e: f"2^{e} = {2**e:,}",
        )
        repeats = st.slider("サイズごとの反復回数", 2, 10, 5)
        timeout_s = st.slider("1 回の実行のタイムアウト (秒)", 1, 30, 5)
        extra_kinds = st.multiselect(
            "追加の入力種別",
            options=["sorted", "reversed"],
            default=["sorted", "reversed"],
            format_func=lambda k: {"sorted": "整列済み", "reversed": "逆順"}[k],
            help=(
                "ランダム入力に加えて計測する入力種別。種別ごとに計算量オーダーを"
                "判定します (最良/最悪ケースの実測確認用)。総合判定はランダム入力で行います。"
            ),
        )

        st.warning(
            "⚠️ 本アプリは入力されたコードを**実際に実行**します。"
            "タイムアウト等の基本的な保護はありますが、サンドボックスではありません。"
            "信頼できるコードのみを解析してください。"
        )

    # ---------- ソース取得 ----------

    source: str | None = None
    if input_mode == "コード貼り付け":
        source = st.text_area(
            "ソートアルゴリズムの Python コード",
            height=280,
            placeholder="def my_sort(arr):\n    ...",
            key="pasted_source",
        )
    elif input_mode == ".py ファイルをアップロード":
        uploaded = st.file_uploader("Python ファイル (.py)", type=["py"])
        if uploaded is not None:
            source = uploaded.read().decode("utf-8")
            with st.expander("アップロードされたコード", expanded=False):
                st.code(source, language="python")
    else:
        example_names = sorted(p.stem for p in EXAMPLES_DIR.glob("*.py"))
        chosen = st.selectbox("サンプルを選択", example_names)
        if chosen:
            source = (EXAMPLES_DIR / f"{chosen}.py").read_text(encoding="utf-8")
            with st.expander("サンプルコード", expanded=True):
                st.code(source, language="python")

    if not source or not source.strip():
        st.info("ソートアルゴリズムの Python コードを入力してください。")
        st.stop()

    # ---------- ソート関数の検出 ----------

    try:
        candidates = find_sort_functions(source)
    except SyntaxErrorJa as e:
        st.error(f"❌ {e}")
        st.stop()

    if not candidates:
        st.error(
            "❌ ソート関数らしい関数が見つかりませんでした。"
            "リストを引数に取るトップレベル関数を定義してください。"
        )
        st.stop()

    if len(candidates) > 1:
        func_name = st.selectbox(
            "解析対象の関数 (ソート関数らしい順)",
            [c.name for c in candidates],
            format_func=lambda n: (
                f"{n} (信頼度 {next(c.confidence for c in candidates if c.name == n):.2f})"
            ),
        )
    else:
        func_name = candidates[0].name
    chosen_info = next(c for c in candidates if c.name == func_name)

    st.subheader(f"解析対象: `{func_name}()`")
    with st.expander("関数検出の根拠"):
        for reason in chosen_info.reasons:
            st.markdown(f"- {reason}")

    # ベンチマーク結果はソース・関数・設定が変わったら無効化する
    _bench_key = (source, func_name, exp_range, repeats, timeout_s, tuple(extra_kinds))
    if st.session_state.get("bench_key") != _bench_key:
        st.session_state.pop("bench_result", None)

    col_theory, col_empirical = st.columns(2)

    # ---------- 理論解析 (即時) ----------

    with col_theory:
        st.header("理論解析 (静的)")
        try:
            static_result = analyze_function(source, func_name)
        except (ValueError, SyntaxError, RecursionError) as e:
            st.error(f"静的解析に失敗しました: {e}")
            static_result = None

        if static_result is not None:
            has_cases = (
                static_result.best_case is not None
                or static_result.worst_case is not None
            )
            if has_cases:
                # 最良 / 平均 / 最悪 の 3 ケースを横並びで表示
                best = static_result.best_case or static_result.complexity
                worst = static_result.worst_case or static_result.complexity
                for col, (label, cls) in zip(
                    st.columns(3),
                    [("最良", best), ("平均", static_result.complexity), ("最悪", worst)],
                ):
                    with col:
                        st.metric(f"理論計算量 ({label})", cls.label())
            else:
                st.metric("理論計算量", static_result.complexity.label())
            st.progress(
                static_result.confidence,
                text=f"判定の信頼度: {static_result.confidence:.2f}",
            )
            if static_result.recognized_algorithm:
                st.info(f"💡 **{static_result.recognized_algorithm}** の可能性が高い構造です。")
                literature = literature_complexities(static_result.recognized_algorithm)
                if literature:
                    st.caption(
                        f"文献値: 最良 {literature[0]} / 平均 {literature[1]} / "
                        f"最悪 {literature[2]}"
                    )
            with st.expander("判定根拠", expanded=True):
                for reason in static_result.reasoning:
                    st.markdown(f"- {reason}")
            detail = []
            if static_result.max_loop_depth:
                detail.append(f"ループの最大入れ子深さ: {static_result.max_loop_depth}")
            if static_result.has_recursion:
                kind_ja = {
                    "linear": "線形再帰",
                    "divide_halving": "半分割の分割統治",
                    "divide_partition": "パーティション分割の分割統治",
                    "unknown": "分類不能の再帰",
                }.get(static_result.recursion_kind or "unknown", "再帰")
                detail.append(f"再帰: あり ({kind_ja})")
            if detail:
                st.caption(" / ".join(detail))

    # ---------- 実測解析 (ボタンで実行) ----------

    with col_empirical:
        st.header("実測解析 (ベンチマーク)")
        if st.button("▶ ベンチマーク実行", type="primary", use_container_width=True):
            config = BenchmarkConfig(
                min_exp=exp_range[0],
                max_exp=exp_range[1],
                repeats=repeats,
                per_run_timeout=float(timeout_s),
                extra_kinds=tuple(extra_kinds),
            )
            progress_bar = st.progress(0.0, text="準備中…")

            def _on_progress(done: int, total: int, message: str) -> None:
                progress_bar.progress(min(done / max(total, 1), 1.0), text=message)

            with st.spinner("計測中…"):
                result = run_benchmark(source, func_name, config, _on_progress)
            progress_bar.empty()
            st.session_state["bench_result"] = result
            st.session_state["bench_key"] = _bench_key

        result = st.session_state.get("bench_result")
        if result is None:
            st.caption("「ベンチマーク実行」を押すと実測を開始します。")
        elif not result.correctness_ok:
            st.error("❌ " + (result.notes[0] if result.notes else "正当性検査に失敗しました"))
        else:
            kind_ja_map = {"random": "ランダム", "sorted": "整列済み", "reversed": "逆順"}
            kind_summaries = [
                result.kind_results[k]
                for k in ("random", "sorted", "reversed")
                if k in result.kind_results
            ]
            if len(kind_summaries) >= 2:
                # 種別ごとの実測計算量を横並びで表示 (値はクラス名、詳細は caption)
                for col, summary in zip(st.columns(len(kind_summaries)), kind_summaries):
                    with col:
                        st.metric(
                            f"実測計算量 ({kind_ja_map.get(summary.input_kind, summary.input_kind)})",
                            summary.best_fit.complexity.label()
                            if summary.best_fit
                            else "判定不能",
                        )
                        if not math.isnan(summary.exponent):
                            st.caption(
                                f"O(n^{summary.exponent:.2f})・"
                                f"指数 {summary.exponent:.2f} ± {summary.exponent_stderr:.2f}"
                            )
            else:
                st.metric("実測計算量", result.summary_label())
                if not math.isnan(result.exponent):
                    st.caption(
                        f"log-log 回帰の実測指数: {result.exponent:.2f} ± {result.exponent_stderr:.2f}"
                    )
            if result.timed_out:
                st.warning("一部のサイズはタイムアウトにより打ち切られました。")
            for note in result.notes:
                st.info(note)

            st.pyplot(make_loglog_figure(result), use_container_width=True)

            # 測定値とフィット結果の表
            meas_df = pd.DataFrame(
                {
                    "サイズ n": [m.size for m in result.measurements],
                    "入力種別": [
                        kind_ja_map.get(m.input_kind, m.input_kind)
                        for m in result.measurements
                    ],
                    "中央値 [秒]": [m.median_time for m in result.measurements],
                    "試行回数": [len(m.times) for m in result.measurements],
                }
            )
            fit_rows = [
                (kind_ja_map.get(s.input_kind, s.input_kind), f)
                for s in kind_summaries
                for f in s.fits
            ] or [("ランダム", f) for f in result.fits]
            fit_df = pd.DataFrame(
                {
                    "入力種別": [kind for kind, _ in fit_rows],
                    "モデル": [f.complexity.label() for _, f in fit_rows],
                    "R² (log 空間)": [round(f.r_squared, 4) for _, f in fit_rows],
                    "AIC": [round(f.aic, 2) for _, f in fit_rows],
                }
            )
            tab_meas, tab_fit = st.tabs(["測定値", "フィット比較"])
            with tab_meas:
                st.dataframe(meas_df, use_container_width=True, hide_index=True)
            with tab_fit:
                st.dataframe(fit_df, use_container_width=True, hide_index=True)

    # ---------- 総合判定 ----------

    result = st.session_state.get("bench_result")
    if (
        static_result is not None
        and result is not None
        and result.correctness_ok
        and result.best_fit is not None
    ):
        st.divider()
        st.header("総合判定")
        theoretical = static_result.complexity
        empirical = result.best_fit.complexity
        if theoretical in (ComplexityClass.UNKNOWN, ComplexityClass.EXPONENTIAL):
            st.warning(
                f"理論 {theoretical.label()}・実測 {result.summary_label()} (ランダム入力) — "
                "理論オーダーを特定できなかったため比較できません。"
            )
        elif classes_agree(theoretical, empirical, static_result.worst_case):
            st.success(
                f"✅ 理論 {theoretical.label()}・実測 {result.summary_label()} (ランダム入力) — "
                "**一致しています**"
            )
        else:
            st.warning(
                f"⚠️ 理論 {theoretical.label()}・実測 {result.summary_label()} (ランダム入力) — "
                "**一致していません**。測定範囲が狭い、定数項が支配的、"
                "または静的解析の仮定が実装と合っていない可能性があります。"
            )

        # 最良/最悪ケースと整列済み/逆順入力の実測の対応 (参考情報)
        if static_result.best_case is not None and "sorted" in result.kind_results:
            summary = result.kind_results["sorted"]
            if summary.best_fit is not None:
                st.caption(
                    f"参考: 最良ケース理論 {static_result.best_case.label()} ⇔ "
                    f"整列済み入力の実測 {summary.summary_label()}"
                )
        if static_result.worst_case is not None and "reversed" in result.kind_results:
            summary = result.kind_results["reversed"]
            if summary.best_fit is not None:
                st.caption(
                    f"参考: 最悪ケース理論 {static_result.worst_case.label()} ⇔ "
                    f"逆順入力の実測 {summary.summary_label()} "
                    "(最悪ケースの入力が逆順とは限りません)"
                )

if __name__ == "__main__":
    # Streamlit は __name__ == "__main__" でスクリプトを実行する。
    # multiprocessing (spawn) の子プロセスは __mp_main__ として本ファイルを
    # 再読み込みするため、このガードで UI の再実行を防ぐ。
    main()
