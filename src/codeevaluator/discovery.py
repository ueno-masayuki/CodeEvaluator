"""ソースコード中から「ソート関数」を検出するモジュール。

トップレベルの関数定義をヒューリスティックにスコアリングし、
ソート関数らしい候補を信頼度の降順で返す。

静的スコアリングのみを行う(実行による動的検証は runner.py が担う)。
"""

from __future__ import annotations

import ast

from codeevaluator.models import SortFunctionInfo

#: 第1引数名としてよくあるリスト系の名前
_LIST_PARAM_NAMES = {
    "arr", "array", "lst", "list_", "data", "a", "xs", "nums",
    "numbers", "items", "l", "seq", "values", "elements",
}

#: 候補として採用する最低スコア
SCORE_THRESHOLD = 0.3


class SyntaxErrorJa(ValueError):
    """構文エラーを日本語メッセージで伝える例外。"""

    def __init__(self, err: SyntaxError):
        self.original = err
        super().__init__(
            f"Python コードの構文エラーです (行 {err.lineno}): {err.msg}"
        )


def find_sort_functions(source: str) -> list[SortFunctionInfo]:
    """ソース文字列からソート関数の候補を信頼度降順で返す。

    構文エラー時は SyntaxErrorJa を送出する。
    """
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        raise SyntaxErrorJa(e) from e

    candidates: list[SortFunctionInfo] = []
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        info = _score_function(node, source)
        if info is not None and info.confidence >= SCORE_THRESHOLD:
            candidates.append(info)

    candidates.sort(key=lambda c: c.confidence, reverse=True)
    return candidates


def _score_function(node: ast.FunctionDef, source: str) -> SortFunctionInfo | None:
    """1つの関数定義をスコアリングする。明らかに対象外なら None。"""
    reasons: list[str] = []
    score = 0.0

    args = node.args.posonlyargs + node.args.args
    if not args:
        # 引数なしの関数はソート対象を受け取れない
        return None
    if node.name.startswith("_") or node.name == "main":
        score -= 0.2
        reasons.append(f"関数名 '{node.name}' は補助関数・エントリポイントの可能性")

    # 関数名に sort を含む
    if "sort" in node.name.lower():
        score += 0.4
        reasons.append(f"関数名 '{node.name}' に 'sort' を含む")

    # 第1引数名がリスト系
    first = args[0].arg
    if first.lower() in _LIST_PARAM_NAMES:
        score += 0.3
        reasons.append(f"第1引数名 '{first}' がリストを示唆")

    # 本体の構造: 比較演算と並べ替え操作
    has_compare = False
    has_rearrange = False
    returns_value = False
    calls_sorted = False
    for sub in ast.walk(node):
        if isinstance(sub, ast.Compare) and any(
            isinstance(op, (ast.Lt, ast.Gt, ast.LtE, ast.GtE)) for op in sub.ops
        ):
            has_compare = True
        # スワップ (タプル代入) / insert / append は並べ替え操作とみなす
        if isinstance(sub, ast.Assign) and any(
            isinstance(t, ast.Tuple) for t in sub.targets
        ):
            has_rearrange = True
        if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute):
            if sub.func.attr in ("insert", "append", "sort"):
                has_rearrange = True
        if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name):
            if sub.func.id == "sorted":
                calls_sorted = True
                has_rearrange = True
        if isinstance(sub, ast.Return) and sub.value is not None:
            returns_value = True

    if has_compare and has_rearrange:
        score += 0.2
        reasons.append("比較演算と並べ替え操作 (スワップ/insert/append) を含む")
    elif calls_sorted:
        score += 0.2
        reasons.append("組み込み sorted() を呼び出している")

    # リストを返す、または戻り値なし (インプレース変更の可能性)
    if returns_value:
        score += 0.1
        reasons.append("値を return している")
        mutates = False
    else:
        score += 0.1
        reasons.append("return なし (インプレース変更の可能性)")
        mutates = True

    score = max(0.0, min(1.0, score))
    segment = ast.get_source_segment(source, node) or ""
    return SortFunctionInfo(
        name=node.name,
        source=segment,
        lineno=node.lineno,
        confidence=round(score, 2),
        reasons=reasons,
        mutates_in_place=mutates,
    )
