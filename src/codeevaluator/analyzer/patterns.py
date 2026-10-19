"""既知のソートアルゴリズムの構造パターン認識 (補助情報)。

オーダー判定の主体は static.py の AST 合成であり、
本モジュールは「バブルソートの可能性が高い」等の表示補足にのみ使う。
"""

from __future__ import annotations

import ast

#: 認識済みアルゴリズムの文献上の計算量 (最良, 平均, 最悪)。表示の補足専用。
LITERATURE_COMPLEXITIES: dict[str, tuple[str, str, str]] = {
    "バブルソート": ("O(n) (早期終了付き実装)", "O(n²)", "O(n²)"),
    "挿入ソート": ("O(n)", "O(n²)", "O(n²)"),
    "選択ソート": ("O(n²)", "O(n²)", "O(n²)"),
    "マージソート": ("O(n log n)", "O(n log n)", "O(n log n)"),
    "クイックソート": ("O(n log n)", "O(n log n)", "O(n²)"),
    "ヒープソート": ("O(n log n)", "O(n log n)", "O(n log n)"),
    "組み込みソート (Timsort)": ("O(n)", "O(n log n)", "O(n log n)"),
}


def literature_complexities(name: str | None) -> tuple[str, str, str] | None:
    """認識済みアルゴリズム名に対する文献上の (最良, 平均, 最悪) を返す。"""
    if name is None:
        return None
    return LITERATURE_COMPLEXITIES.get(name)


def recognize(
    func: ast.FunctionDef, module_funcs: dict[str, ast.FunctionDef] | None = None
) -> str | None:
    """関数の構造シグネチャから既知アルゴリズムの日本語名を返す。不明なら None。"""
    module_funcs = module_funcs or {}

    if _is_builtin_wrapper(func):
        return "組み込みソート (Timsort)"
    if _uses_heapq(func):
        return "ヒープソート"

    has_self_recursion = _has_self_call(func)
    if has_self_recursion:
        if _has_partition_comprehension(func) or _mentions_pivot(func):
            return "クイックソート"
        if _has_halving_slices(func):
            return "マージソート"
        return None

    # 非再帰: ループ構造で判別
    if _is_bubble(func):
        return "バブルソート"
    if _is_insertion(func):
        return "挿入ソート"
    if _is_selection(func):
        return "選択ソート"
    return None


# ---------- 個別パターン ----------


def _is_builtin_wrapper(func: ast.FunctionDef) -> bool:
    """本体が実質 sorted() / .sort() の呼び出しだけか。"""
    stmts = [
        s
        for s in func.body
        if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant))
    ]  # docstring を除く
    if len(stmts) > 3:
        return False
    uses = False
    for s in stmts:
        for sub in ast.walk(s):
            if isinstance(sub, ast.Call):
                if isinstance(sub.func, ast.Name) and sub.func.id == "sorted":
                    uses = True
                elif isinstance(sub.func, ast.Attribute) and sub.func.attr == "sort":
                    uses = True
            # ループや再帰があればラッパーではない
        if isinstance(s, (ast.For, ast.While)):
            return False
    return uses


def _uses_heapq(func: ast.FunctionDef) -> bool:
    for sub in ast.walk(func):
        if isinstance(sub, ast.Attribute) and sub.attr in (
            "heapify", "heappush", "heappop", "heapreplace", "heappushpop",
        ):
            return True
        if isinstance(sub, ast.Name) and sub.id in ("heapify", "heappush", "heappop"):
            return True
    return False


def _has_self_call(func: ast.FunctionDef) -> bool:
    for sub in ast.walk(func):
        if (
            isinstance(sub, ast.Call)
            and isinstance(sub.func, ast.Name)
            and sub.func.id == func.name
        ):
            return True
    return False


def _has_partition_comprehension(func: ast.FunctionDef) -> bool:
    for sub in ast.walk(func):
        if isinstance(sub, (ast.ListComp, ast.GeneratorExp)) and any(
            gen.ifs for gen in sub.generators
        ):
            return True
    return False


def _mentions_pivot(func: ast.FunctionDef) -> bool:
    for sub in ast.walk(func):
        if isinstance(sub, ast.Name) and "pivot" in sub.id.lower():
            return True
        if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name) and (
            "partition" in sub.func.id.lower()
        ):
            return True
    return False


def _has_halving_slices(func: ast.FunctionDef) -> bool:
    """mid = ...//2 を使ったスライス分割があるか (マージソートの形)。"""
    halving_names: set[str] = set()
    for sub in ast.walk(func):
        if isinstance(sub, ast.Assign) and len(sub.targets) == 1 and isinstance(
            sub.targets[0], ast.Name
        ):
            for inner in ast.walk(sub.value):
                if isinstance(inner, ast.BinOp) and isinstance(
                    inner.op, (ast.FloorDiv, ast.RShift)
                ):
                    halving_names.add(sub.targets[0].id)
    if not halving_names:
        return False
    for sub in ast.walk(func):
        if isinstance(sub, ast.Subscript) and isinstance(sub.slice, ast.Slice):
            names = {n.id for n in ast.walk(sub.slice) if isinstance(n, ast.Name)}
            if names & halving_names:
                return True
    return False


def _adjacent_swap(stmt: ast.stmt) -> bool:
    """data[j], data[j+1] = data[j+1], data[j] 形の隣接スワップか。"""
    if not (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1):
        return False
    target = stmt.targets[0]
    if not (isinstance(target, ast.Tuple) and len(target.elts) == 2):
        return False
    a, b = target.elts
    if not (isinstance(a, ast.Subscript) and isinstance(b, ast.Subscript)):
        return False
    ia, ib = a.slice, b.slice
    # 片方の添字が「もう片方 + 1」になっているか
    def plus_one(base: ast.expr, other: ast.expr) -> bool:
        return (
            isinstance(other, ast.BinOp)
            and isinstance(other.op, ast.Add)
            and isinstance(other.right, ast.Constant)
            and other.right.value == 1
            and ast.dump(other.left) == ast.dump(base)
        )

    return plus_one(ia, ib) or plus_one(ib, ia)


def _swap_of_subscripts(stmt: ast.stmt) -> bool:
    """x[i], x[k] = x[k], x[i] 形のスワップか (隣接とは限らない)。"""
    if not (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1):
        return False
    target = stmt.targets[0]
    return (
        isinstance(target, ast.Tuple)
        and len(target.elts) == 2
        and all(isinstance(e, ast.Subscript) for e in target.elts)
    )


def _is_bubble(func: ast.FunctionDef) -> bool:
    """2重 for ループの内側に隣接スワップ → バブルソート。"""
    for outer in ast.walk(func):
        if not isinstance(outer, ast.For):
            continue
        for inner in outer.body:
            if not isinstance(inner, ast.For):
                continue
            for sub in ast.walk(inner):
                if isinstance(sub, ast.stmt) and _adjacent_swap(sub):
                    return True
    return False


def _is_insertion(func: ast.FunctionDef) -> bool:
    """for の中に後方シフトの while、または .insert() → 挿入ソート。"""
    for outer in ast.walk(func):
        if not isinstance(outer, ast.For):
            continue
        for sub in ast.walk(outer):
            if isinstance(sub, ast.While):
                # while 内で添字を減らしながら要素をシフト
                for s in ast.walk(sub):
                    if isinstance(s, ast.AugAssign) and isinstance(s.op, ast.Sub):
                        return True
            if (
                isinstance(sub, ast.Call)
                and isinstance(sub.func, ast.Attribute)
                and sub.func.attr == "insert"
            ):
                return True
    return False


def _is_selection(func: ast.FunctionDef) -> bool:
    """内側 for で最小/最大位置を探し、外側でスワップ → 選択ソート。"""
    for outer in ast.walk(func):
        if not isinstance(outer, ast.For):
            continue
        has_inner_argmin = False
        has_outer_swap = False
        for stmt in outer.body:
            if isinstance(stmt, ast.For):
                # 内側ループで If の下に添字変数の更新があるか
                for sub in ast.walk(stmt):
                    if isinstance(sub, ast.If):
                        for s in ast.walk(sub):
                            if isinstance(s, ast.Assign) and any(
                                isinstance(t, ast.Name) for t in s.targets
                            ):
                                has_inner_argmin = True
            elif _swap_of_subscripts(stmt):
                has_outer_swap = True
        if has_inner_argmin and has_outer_swap:
            return True
    return False
