"""AST 静的解析による理論計算量オーダーの導出。

方針:
- 文のリストは直列合成 = 各文の max
- ループは「反復回数 × 本体」の積
- while はループ変数の更新パターンから反復回数を分類 (半減 → O(log n) 等)
- sorted() / list.insert / heapq 等の暗黙コストを加味
- 再帰は呼び出し数とサイズ削減方法を分類し、簡易マスター定理で解く
- 判定根拠 (reasoning) は日本語で積み、置いた仮定ごとに信頼度を下げる

任意コードの厳密なオーダー決定は原理的に不可能なため、
判別できない構造には明示的な仮定を置き、根拠欄に残す。
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field

from codeevaluator.analyzer import patterns
from codeevaluator.models import (
    ComplexityClass,
    ComplexityTerm,
    StaticAnalysisResult,
)

#: よく使う定数コスト
_COST_SORTED = ComplexityTerm(1, 1)   # sorted() / .sort() — Timsort
_COST_LINEAR = ComplexityTerm.N
_COST_LOG = ComplexityTerm.LOG_N


def analyze_function(source: str, func_name: str) -> StaticAnalysisResult:
    """ソース中の指定関数を解析し、理論計算量の判定結果を返す。"""
    tree = ast.parse(source)
    funcs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    if func_name not in funcs:
        raise ValueError(f"関数 '{func_name}' が見つかりません")

    module = _ModuleAnalyzer(funcs)
    res = module.analyze(func_name)

    if res.exponential:
        complexity = ComplexityClass.EXPONENTIAL
    else:
        complexity = res.term.to_class()
        if complexity is ComplexityClass.UNKNOWN:
            res.reasons.append(
                f"合成結果 {res.term.label()} は既知のクラスに該当しないため「不明」とします"
            )

    confidence = max(0.3, round(0.9 - 0.15 * res.assumptions, 2))
    recognized = patterns.recognize(funcs[func_name], funcs)

    worst = None
    if res.worst_term is not None and res.worst_term != res.term:
        worst = res.worst_term.to_class()

    return StaticAnalysisResult(
        function_name=func_name,
        complexity=complexity,
        confidence=confidence,
        reasoning=res.reasons,
        max_loop_depth=res.max_loop_depth,
        has_recursion=res.has_recursion,
        recursion_kind=res.recursion_kind,
        recognized_algorithm=recognized,
        worst_case=worst,
    )


@dataclass
class _FuncResult:
    """関数 1 つ分の内部解析結果。"""

    term: ComplexityTerm = ComplexityTerm.ZERO
    worst_term: ComplexityTerm | None = None
    reasons: list[str] = field(default_factory=list)
    assumptions: int = 0
    has_recursion: bool = False
    recursion_kind: str | None = None
    max_loop_depth: int = 0
    exponential: bool = False


class _ModuleAnalyzer:
    """モジュール内の関数間呼び出しを解決しながら解析する (メモ化・循環ガード)。"""

    def __init__(self, funcs: dict[str, ast.FunctionDef]):
        self.funcs = funcs
        self._memo: dict[str, _FuncResult] = {}
        self._stack: set[str] = set()

    def analyze(self, name: str) -> _FuncResult:
        if name in self._memo:
            return self._memo[name]
        self._stack.add(name)
        try:
            result = _FunctionAnalyzer(self.funcs[name], self).run()
        finally:
            self._stack.discard(name)
        self._memo[name] = result
        return result

    def cost_of_call(self, name: str) -> tuple[ComplexityTerm, str | None, int]:
        """同一モジュール内のユーザー関数呼び出しのコストを返す。

        戻り値: (コスト項, 日本語根拠 or None, 追加の仮定数)
        """
        if name in self._stack:
            # 相互再帰は簡易解析の対象外: 仮定を置いて線形とみなす
            return (
                _COST_LINEAR,
                f"相互再帰 '{name}' は解析対象外のため O(n) と仮定",
                1,
            )
        sub = self.analyze(name)
        return (
            sub.term,
            f"関数 '{name}' の呼び出し: {sub.term.label()}",
            0,
        )


class _FunctionAnalyzer:
    """1 つの関数本体の計算量を合成する。"""

    def __init__(self, func: ast.FunctionDef, module: _ModuleAnalyzer):
        self.func = func
        self.module = module
        self.result = _FuncResult()
        self._seen_reasons: set[str] = set()
        # 自己再帰呼び出しの記録 (ノードのリスト)
        self.self_calls: list[ast.Call] = []
        # 代入の事前走査: 半分割・パーティション変数の検出に使う
        self._halving_names: set[str] = set()
        self._partition_names: set[str] = set()
        self._prescan_assignments()

    # ---------- 公開 ----------

    def run(self) -> _FuncResult:
        body_term = self._stmts_term(self.func.body, depth=0)
        if self.self_calls:
            self._resolve_recursion(body_term)
        else:
            self.result.term = body_term
            if self.result.max_loop_depth > 0:
                self._add_reason(
                    f"ループ構造の合成結果: {body_term.label()}"
                    f" (最大入れ子深さ {self.result.max_loop_depth})"
                )
        return self.result

    # ---------- 補助 ----------

    def _add_reason(self, text: str, assumption: bool = False) -> None:
        if text not in self._seen_reasons:
            self._seen_reasons.add(text)
            self.result.reasons.append(text)
            if assumption:
                self.result.assumptions += 1

    def _prescan_assignments(self) -> None:
        """mid = len(a)//2 (半分割) や less = [x for x in a if ...] (分割) を記録。"""
        for sub in ast.walk(self.func):
            if not (isinstance(sub, ast.Assign) and len(sub.targets) == 1):
                continue
            target = sub.targets[0]
            if not isinstance(target, ast.Name):
                continue
            value = sub.value
            # 値の式に // 2 や >> 1 を含む → 半分割インデックス
            for inner in ast.walk(value):
                if isinstance(inner, ast.BinOp) and isinstance(
                    inner.op, (ast.FloorDiv, ast.RShift)
                ):
                    self._halving_names.add(target.id)
            # 条件付き内包表記 → パーティション結果
            if isinstance(value, (ast.ListComp, ast.GeneratorExp)) and any(
                gen.ifs for gen in value.generators
            ):
                self._partition_names.add(target.id)

    # ---------- 文の合成 ----------

    def _stmts_term(self, stmts: list[ast.stmt], depth: int) -> ComplexityTerm:
        term = ComplexityTerm.ZERO
        for s in stmts:
            term = max(term, self._stmt_term(s, depth))
        return term

    def _stmt_term(self, stmt: ast.stmt, depth: int) -> ComplexityTerm:
        if isinstance(stmt, ast.For):
            return self._for_term(stmt, depth)
        if isinstance(stmt, ast.While):
            return self._while_term(stmt, depth)
        if isinstance(stmt, ast.If):
            branches = max(
                self._stmts_term(stmt.body, depth),
                self._stmts_term(stmt.orelse, depth),
            )
            return max(self._expr_term(stmt.test), branches)
        if isinstance(stmt, (ast.Try,)):
            inner = max(
                self._stmts_term(stmt.body, depth),
                self._stmts_term(stmt.orelse, depth),
                self._stmts_term(stmt.finalbody, depth),
            )
            for handler in stmt.handlers:
                inner = max(inner, self._stmts_term(handler.body, depth))
            return inner
        if isinstance(stmt, (ast.With,)):
            return self._stmts_term(stmt.body, depth)
        if isinstance(stmt, ast.FunctionDef):
            # 入れ子関数の定義自体はコスト 0 (呼び出し時に評価)
            return ComplexityTerm.ZERO
        # 代入・式・return 等: 式中の呼び出しコストを拾う
        term = ComplexityTerm.ZERO
        for child in ast.iter_child_nodes(stmt):
            if isinstance(child, ast.expr):
                term = max(term, self._expr_term(child))
        return term

    def _for_term(self, stmt: ast.For, depth: int) -> ComplexityTerm:
        self.result.max_loop_depth = max(self.result.max_loop_depth, depth + 1)
        iters, label = self._for_iterations(stmt.iter)
        body = max(
            self._stmts_term(stmt.body, depth + 1),
            self._stmts_term(stmt.orelse, depth),
            self._expr_term(stmt.iter),
        )
        total = iters.times(max(body, ComplexityTerm.ZERO))
        self._add_reason(
            f"for ループ (深さ {depth + 1}): 反復回数 {label}"
            + (f" × 本体 {body.label()}" if body > ComplexityTerm.ZERO else "")
        )
        return total

    def _for_iterations(self, iter_expr: ast.expr) -> tuple[ComplexityTerm, str]:
        """for の反復回数を分類する。戻り値: (項, 日本語ラベル)。"""
        if isinstance(iter_expr, ast.Call) and isinstance(iter_expr.func, ast.Name):
            if iter_expr.func.id == "range":
                if iter_expr.args and all(
                    isinstance(a, ast.Constant) for a in iter_expr.args
                ):
                    return ComplexityTerm.ZERO, "定数回"
                return _COST_LINEAR, "n 回 (range)"
        if isinstance(iter_expr, (ast.Name, ast.Attribute, ast.Call, ast.Subscript)):
            return _COST_LINEAR, "n 回 (シーケンス走査)"
        self._add_reason("for の反復回数を判別できないため n 回と仮定", assumption=True)
        return _COST_LINEAR, "n 回 (仮定)"

    def _while_term(self, stmt: ast.While, depth: int) -> ComplexityTerm:
        self.result.max_loop_depth = max(self.result.max_loop_depth, depth + 1)
        iters, label, assumed = self._while_iterations(stmt)
        body = max(
            self._stmts_term(stmt.body, depth + 1),
            self._stmts_term(stmt.orelse, depth),
        )
        self._add_reason(
            f"while ループ (深さ {depth + 1}): 反復回数 {label}"
            + (f" × 本体 {body.label()}" if body > ComplexityTerm.ZERO else ""),
            assumption=assumed,
        )
        return iters.times(max(body, ComplexityTerm.ZERO))

    def _while_iterations(
        self, stmt: ast.While
    ) -> tuple[ComplexityTerm, str, bool]:
        """while の反復回数を更新パターンから分類する。

        戻り値: (項, 日本語ラベル, 仮定を置いたか)
        """
        test_names = {
            n.id for n in ast.walk(stmt.test) if isinstance(n, ast.Name)
        }
        log_update = False
        linear_update = False

        for sub in ast.walk(stmt):
            # i //= 2, i >>= 1, i *= 2, i <<= 1 → 対数回
            if isinstance(sub, ast.AugAssign) and isinstance(sub.target, ast.Name):
                if sub.target.id in test_names:
                    if isinstance(sub.op, (ast.FloorDiv, ast.RShift, ast.Mult, ast.LShift)):
                        log_update = True
                    elif isinstance(sub.op, (ast.Add, ast.Sub)):
                        linear_update = True
            # i = i // 2, i = 2 * i + 1 (ヒープの sift) → 対数回 / i = i + 1 → 線形
            if (
                isinstance(sub, ast.Assign)
                and len(sub.targets) == 1
                and isinstance(sub.targets[0], ast.Name)
                and sub.targets[0].id in test_names
            ):
                ops = [
                    type(inner.op)
                    for inner in ast.walk(sub.value)
                    if isinstance(inner, ast.BinOp)
                ]
                if any(op in (ast.FloorDiv, ast.RShift, ast.Mult, ast.LShift) for op in ops):
                    log_update = True
                elif any(op in (ast.Add, ast.Sub) for op in ops):
                    linear_update = True
            # 条件変数のリストから pop / heappop / del → 要素数ぶん線形
            if isinstance(sub, ast.Call):
                if (
                    isinstance(sub.func, ast.Attribute)
                    and sub.func.attr == "pop"
                    and isinstance(sub.func.value, ast.Name)
                    and sub.func.value.id in test_names
                ):
                    linear_update = True
                if (
                    isinstance(sub.func, ast.Attribute)
                    and sub.func.attr in ("heappop",)
                    and sub.args
                    and isinstance(sub.args[0], ast.Name)
                    and sub.args[0].id in test_names
                ):
                    linear_update = True
            if isinstance(sub, ast.Delete):
                linear_update = True

        if log_update and not linear_update:
            return _COST_LOG, "O(log n) 回 (ループ変数を毎回半減/倍化)", False
        if linear_update:
            return _COST_LINEAR, "O(n) 回 (ループ変数を毎回 ±1 / 要素を 1 つずつ除去)", False
        return _COST_LINEAR, "O(n) 回 (更新パターン不明のため仮定)", True

    # ---------- 式のコスト ----------

    def _expr_term(self, node: ast.expr | None) -> ComplexityTerm:
        if node is None:
            return ComplexityTerm.ZERO
        if isinstance(node, ast.Call):
            cost = self._call_cost(node)
            for child in ast.iter_child_nodes(node):
                if isinstance(child, ast.expr):
                    cost = max(cost, self._expr_term(child))
            return cost
        if isinstance(node, (ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp)):
            gens = ComplexityTerm(len(node.generators), 0)
            inner = ComplexityTerm.ZERO
            if isinstance(node, ast.DictComp):
                inner = max(self._expr_term(node.key), self._expr_term(node.value))
            else:
                inner = self._expr_term(node.elt)
            for gen in node.generators:
                for cond in gen.ifs:
                    inner = max(inner, self._expr_term(cond))
            self._add_reason("内包表記によるシーケンス走査: O(n)")
            return gens.times(inner)
        if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Slice):
            # スライスコピーは O(n)
            inner = ComplexityTerm.ZERO
            for child in ast.iter_child_nodes(node):
                if isinstance(child, ast.expr):
                    inner = max(inner, self._expr_term(child))
            return max(_COST_LINEAR, inner)
        if isinstance(node, ast.Compare) and any(
            isinstance(op, (ast.In, ast.NotIn)) for op in node.ops
        ):
            self._add_reason("in 演算子によるシーケンス探索: O(n) と仮定", assumption=True)
            return _COST_LINEAR
        # その他: 子ノードの max
        term = ComplexityTerm.ZERO
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.expr):
                term = max(term, self._expr_term(child))
        return term

    def _call_cost(self, call: ast.Call) -> ComplexityTerm:
        func = call.func
        # 組み込み関数
        if isinstance(func, ast.Name):
            name = func.id
            if name == "sorted":
                self._add_reason("組み込み sorted() (Timsort) を使用: O(n log n)")
                return _COST_SORTED
            if name in ("min", "max", "sum", "list", "tuple", "set", "reversed"):
                return _COST_LINEAR
            if name in ("len", "range", "abs", "int", "float", "print", "enumerate", "zip"):
                return ComplexityTerm.ZERO
            if name == self.func.name:
                # 自己再帰: ここではコスト 0 とし、後で再帰解析に回す
                self.self_calls.append(call)
                return ComplexityTerm.ZERO
            if name in self.module.funcs:
                cost, reason, extra = self.module.cost_of_call(name)
                if reason:
                    self._add_reason(reason, assumption=extra > 0)
                return cost
            return ComplexityTerm.ZERO
        # メソッド・モジュール関数
        if isinstance(func, ast.Attribute):
            attr = func.attr
            if attr == "sort":
                self._add_reason("list.sort() (Timsort) を使用: O(n log n)")
                return _COST_SORTED
            if attr in ("insert", "remove", "index", "count", "extend", "copy"):
                self._add_reason(f"list.{attr}() は内部で線形走査/シフト: O(n)")
                return _COST_LINEAR
            if attr == "pop":
                # pop() は O(1)、pop(i) は O(n)
                if call.args:
                    self._add_reason("list.pop(i) は要素シフトを伴う: O(n)")
                    return _COST_LINEAR
                return ComplexityTerm.ZERO
            if attr in ("heappush", "heappop", "heappushpop", "heapreplace"):
                self._add_reason(f"heapq.{attr}() : O(log n)")
                return _COST_LOG
            if attr == "heapify":
                self._add_reason("heapq.heapify() : O(n)")
                return _COST_LINEAR
            if attr in ("append", "add"):
                return ComplexityTerm.ZERO
        return ComplexityTerm.ZERO

    # ---------- 再帰の解析 ----------

    def _resolve_recursion(self, local_work: ComplexityTerm) -> None:
        """自己再帰を分類し、簡易マスター定理で全体の計算量を求める。"""
        res = self.result
        res.has_recursion = True
        a = len(self.self_calls)
        kind = self._classify_reduction()
        res.recursion_kind = kind
        f = local_work

        if kind in ("divide_halving", "divide_partition"):
            term = self._master_divide(a, f)
            if kind == "divide_halving":
                self._add_reason(
                    f"半分割の分割統治 (自己呼び出し {a} 箇所) + 各段 {f.label()} "
                    f"→ {term.label()}"
                )
            else:
                self._add_reason(
                    f"パーティション分割の分割統治 (自己呼び出し {a} 箇所) + 各段 {f.label()} "
                    f"→ 平均 {term.label()}"
                )
                # 最悪ケース: 分割が毎回偏ると線形再帰相当
                worst = ComplexityTerm(f.poly + 1, f.log)
                res.worst_term = worst
                self._add_reason(
                    f"分割が毎回偏る最悪ケースでは {worst.label()} "
                    "(例: ピボットが常に最小値/最大値)"
                )
            res.term = term
        elif kind == "linear":
            if a >= 2:
                res.exponential = True
                self._add_reason(
                    f"サイズ −1 の再帰を {a} 分岐で呼び出し → 指数的 O(2^n) の可能性"
                )
                res.term = ComplexityTerm(99, 0)  # 表示は EXPONENTIAL 側で処理
            else:
                term = ComplexityTerm(f.poly + 1, f.log)
                self._add_reason(
                    f"サイズを 1 ずつ減らす線形再帰 + 各段 {f.label()} → {term.label()}"
                )
                res.term = term
        else:  # unknown
            self._add_reason(
                "再帰のサイズ削減方法を判別できないため、サイズ −1 の線形再帰と仮定",
                assumption=True,
            )
            if a >= 2:
                res.exponential = True
                res.term = ComplexityTerm(99, 0)
            else:
                res.term = ComplexityTerm(f.poly + 1, f.log)

    def _master_divide(self, a: int, f: ComplexityTerm) -> ComplexityTerm:
        """T(n) = a·T(n/2) + f(n) の簡易解 (a は 1 または 2 を想定)。"""
        if a <= 1:
            # T(n) = T(n/2) + f(n): f が定数なら log、そうでなければ f が支配
            if f == ComplexityTerm.ZERO:
                return ComplexityTerm.LOG_N
            return f
        # a >= 2 (2 分割とみなす)
        if f.poly >= 2:
            return f  # 各段の仕事が支配 (マスター定理ケース 3)
        if f.poly == 1:
            return ComplexityTerm(1, f.log + 1)  # ケース 2: n log n 系
        # f.poly == 0 (定数/対数) → 葉の数が支配: O(n)
        return ComplexityTerm.N

    def _classify_reduction(self) -> str:
        """自己呼び出しの引数からサイズ削減方法を分類する。"""
        found_halving = False
        found_partition = False
        found_linear = False

        for call in self.self_calls:
            for arg in call.args:
                # スライス引数
                if isinstance(arg, ast.Subscript) and isinstance(arg.slice, ast.Slice):
                    lower, upper = arg.slice.lower, arg.slice.upper
                    names = {
                        n.id
                        for n in ast.walk(arg.slice)
                        if isinstance(n, ast.Name)
                    }
                    if names & self._halving_names:
                        found_halving = True
                    elif _is_const_offset(lower) or _is_const_offset(upper):
                        found_linear = True
                    else:
                        found_halving = True  # 範囲分割とみなす
                    continue
                # パーティション結果のリスト
                if isinstance(arg, ast.Name) and arg.id in self._partition_names:
                    found_partition = True
                    continue
                # 内包表記を直接渡すケース
                if isinstance(arg, (ast.ListComp, ast.GeneratorExp)) and any(
                    gen.ifs for gen in arg.generators
                ):
                    found_partition = True
                    continue
                # mid / mid+1 等の半分割インデックス
                names = {
                    n.id for n in ast.walk(arg) if isinstance(n, ast.Name)
                }
                if names & self._halving_names:
                    found_halving = True
                    continue
                # n - 1 型
                if (
                    isinstance(arg, ast.BinOp)
                    and isinstance(arg.op, (ast.Sub, ast.Add))
                    and isinstance(arg.right, ast.Constant)
                ):
                    found_linear = True

        if found_partition:
            return "divide_partition"
        if found_halving:
            return "divide_halving"
        if found_linear:
            return "linear"
        return "unknown"


def _is_const_offset(node: ast.expr | None) -> bool:
    """arr[1:] / arr[:-1] のような定数オフセットのスライス境界か。"""
    if node is None:
        return False
    if isinstance(node, ast.Constant):
        return True
    if isinstance(node, ast.UnaryOp) and isinstance(node.operand, ast.Constant):
        return True
    return False
