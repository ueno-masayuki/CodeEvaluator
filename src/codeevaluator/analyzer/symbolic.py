"""記号的ループ解析: 幾何的に変化するループ変数を持つループ入れ子の総コスト評価。

対象は「gap //= 2」「gap = int(gap / 1.3)」のように幾何的に減少 (または倍化で
増加) するループ変数 g を持つ while ループで、その本体に g へ依存する
内側ループ (range の境界や while のステップが g を参照) を含むもの。
シェルソートの「j -= gap」のような依存ステップは、単純な入れ子合成
(反復回数の積) では O(n) と過大評価されるため、ここで正確に扱う。

手法:
1. 外側 1 反復あたりのコスト C(n, g) を sympy 式として構築する
   (内側 for は Sum の閉形式、依存ステップの while は 上界/ステップ)。
2. g の幾何スケジュール (n → n/r → n/r² → …) に沿って C を数値的に総和し、
   複数の n について厳密な反復回数カーブを得る (実行時間ではなく回数なので
   ノイズゼロ)。
3. 候補クラス n^p log^k n (p=0..3, k=0..2) への log 空間フィットで分類する。

構築できない構造 (未知の式・副作用など) では None を返し、
呼び出し側は従来のヒューリスティック合成にフォールバックする。
"""

from __future__ import annotations

import ast
import math
from dataclasses import dataclass

import numpy as np
import sympy as sp

from codeevaluator.models import ComplexityTerm

#: 評価に使う入力サイズ (厳密カウントなので大きくできる)
_EVAL_SIZES = [2 ** e for e in range(8, 22, 2)]

#: 外側ループの反復回数の安全上限 (無限ループ防止)
_MAX_OUTER_ITERS = 10_000


@dataclass
class GeometricVar:
    """幾何的に更新されるループ変数。"""

    name: str
    ratio: float          # 1 反復あたりの縮小/拡大率 (>1)
    growing: bool         # True: 倍化で増加、False: 除算で減少
    start: sp.Expr        # 初期値 (n の式)


@dataclass
class SymbolicResult:
    """記号的解析の結果。"""

    term: ComplexityTerm | None   # None は既知クラス外
    exponent: float               # log-log 勾配 (参考値)
    reasons: list[str]
    assumptions: int


_N = sp.Symbol("n", positive=True)
_G = sp.Symbol("g", positive=True)


def analyze_geometric_while(
    stmt: ast.While,
    size_names: set[str],
    init_assigns: dict[str, ast.expr],
    stmt_cost_term,
) -> SymbolicResult | None:
    """幾何変数を持つ while ループの総コストを分類する。

    Args:
        stmt: 対象の while ループ
        size_names: 「n 相当」とみなす変数名の集合 (len() 由来など)
        init_assigns: 変数名 → ループ前の最後の代入式 (初期値推定用)
        stmt_cost_term: 文の固定コストを ComplexityTerm で返すコールバック
            (sorted() 等の暗黙コストを含める。ループは除く)

    Returns:
        SymbolicResult、または対象外/構築不能なら None。
    """
    gvar = _find_geometric_var(stmt, size_names, init_assigns)
    if gvar is None:
        return None
    if not _body_depends_on(stmt.body, gvar.name):
        return None  # 依存する内側ループがなければ従来の log n × 本体で十分

    env = {gvar.name: _G}
    for name in size_names:
        env[name] = _N
    assumptions = 0

    try:
        cost = _stmts_cost(stmt.body, env, gvar, stmt_cost_term)
    except _Unsupported:
        return None
    if cost is None:
        return None

    # g のスケジュールに沿って数値的に総和する
    try:
        cost_fn = sp.lambdify((_N, _G), cost, modules=["math"])
        start_fn = sp.lambdify((_N,), gvar.start, modules=["math"])
    except Exception:  # noqa: BLE001 - lambdify 失敗は対象外として扱う
        return None

    totals = []
    for n_val in _EVAL_SIZES:
        g_val = float(start_fn(n_val))
        total = 0.0
        iters = 0
        while 1 <= g_val <= n_val * 2 and iters < _MAX_OUTER_ITERS:
            try:
                total += max(0.0, float(cost_fn(n_val, g_val)))
            except (ValueError, ZeroDivisionError, OverflowError):
                return None
            g_val = g_val * gvar.ratio if gvar.growing else g_val / gvar.ratio
            if not gvar.growing and g_val < 1:
                break
            if gvar.growing and g_val > n_val:
                break
            iters += 1
        totals.append(total + 1.0)

    term, exponent = _classify_counts(_EVAL_SIZES, totals)
    direction = "倍化で増加" if gvar.growing else "幾何的に減少"
    label = term.label() if term is not None else f"O(n^{exponent:.2f})"
    reasons = [
        f"記号的ループ解析: ループ変数 '{gvar.name}' は比 {gvar.ratio:g} で{direction}。"
        f"依存する内側ループを含む総反復回数を評価 → {label}"
    ]
    # swapped フラグ等、幾何変数以外の継続条件は無視している
    extra_names = {
        n.id for n in ast.walk(stmt.test) if isinstance(n, ast.Name)
    } - {gvar.name} - size_names
    if extra_names:
        reasons.append(
            f"継続条件の変数 {sorted(extra_names)} による追加反復は無視しました (仮定)"
        )
        assumptions += 1

    return SymbolicResult(
        term=term, exponent=exponent, reasons=reasons, assumptions=assumptions
    )


class _Unsupported(Exception):
    """記号化できない構造に遭遇した。"""


# ---------- 幾何変数の検出 ----------


def _find_geometric_var(
    stmt: ast.While,
    size_names: set[str],
    init_assigns: dict[str, ast.expr],
) -> GeometricVar | None:
    """while の条件変数のうち幾何更新されるものを探す。"""
    test_names = {n.id for n in ast.walk(stmt.test) if isinstance(n, ast.Name)}
    for sub in stmt.body:
        found = _geometric_update(sub, test_names)
        if found is None:
            continue
        name, ratio, growing = found
        start = _initial_value(name, init_assigns, size_names)
        if start is None:
            start = _N  # 初期値不明は n と仮定 (呼び出し側で仮定計上済みの範囲)
        return GeometricVar(name=name, ratio=ratio, growing=growing, start=start)
    return None


def _geometric_update(
    stmt: ast.stmt, test_names: set[str]
) -> tuple[str, float, bool] | None:
    """文が「v //= c / v = int(v/c) / v *= c」型の幾何更新なら (名前, 比, 増加) を返す。"""
    # v //= c, v >>= k, v *= c, v <<= k
    if isinstance(stmt, ast.AugAssign) and isinstance(stmt.target, ast.Name):
        name = stmt.target.id
        if name not in test_names or not isinstance(stmt.value, ast.Constant):
            return None
        c = stmt.value.value
        if not isinstance(c, (int, float)) or c <= 0:
            return None
        if isinstance(stmt.op, ast.FloorDiv) and c > 1:
            return name, float(c), False
        if isinstance(stmt.op, ast.RShift):
            return name, float(2 ** c), False
        if isinstance(stmt.op, ast.Mult) and c > 1:
            return name, float(c), True
        if isinstance(stmt.op, ast.LShift):
            return name, float(2 ** c), True
        return None
    # v = <expr> 型: 式の中の v // c, v / c, v * c を探す (max(1, int(v/1.3)) 等)
    if (
        isinstance(stmt, ast.Assign)
        and len(stmt.targets) == 1
        and isinstance(stmt.targets[0], ast.Name)
    ):
        name = stmt.targets[0].id
        if name not in test_names:
            return None
        for inner in ast.walk(stmt.value):
            if not isinstance(inner, ast.BinOp):
                continue
            left_names = {
                n.id for n in ast.walk(inner.left) if isinstance(n, ast.Name)
            }
            if name not in left_names or not isinstance(inner.right, ast.Constant):
                continue
            c = inner.right.value
            if not isinstance(c, (int, float)) or c <= 1:
                continue
            if isinstance(inner.op, (ast.FloorDiv, ast.Div)):
                return name, float(c), False
            if isinstance(inner.op, ast.Mult):
                return name, float(c), True
            if isinstance(inner.op, ast.RShift):
                return name, float(2 ** c), False
    return None


def _initial_value(
    name: str, init_assigns: dict[str, ast.expr], size_names: set[str]
) -> sp.Expr | None:
    """ループ前の代入から初期値の n 式を推定する。"""
    expr = init_assigns.get(name)
    if expr is None:
        return None
    env = {s: _N for s in size_names}
    try:
        return _to_expr(expr, env)
    except _Unsupported:
        return None


def _body_depends_on(body: list[ast.stmt], name: str) -> bool:
    """本体の内側ループが変数 name に依存するか (range 境界・while 条件/更新)。"""
    for sub in ast.walk(ast.Module(body=body, type_ignores=[])):
        if isinstance(sub, ast.For):
            if any(
                isinstance(n, ast.Name) and n.id == name
                for n in ast.walk(sub.iter)
            ):
                return True
        if isinstance(sub, ast.While):
            if any(
                isinstance(n, ast.Name) and n.id == name
                for n in ast.walk(sub.test)
            ):
                return True
        if isinstance(sub, ast.AugAssign) and any(
            isinstance(n, ast.Name) and n.id == name
            for n in ast.walk(sub.value)
        ):
            return True
    return False


# ---------- コスト式の構築 ----------


def _stmts_cost(
    stmts: list[ast.stmt],
    env: dict[str, sp.Symbol],
    gvar: GeometricVar,
    stmt_cost_term,
) -> sp.Expr:
    """文リストの 1 実行あたりコストを sympy 式で返す (直列は和)。

    j = i のような単純な代入は環境に取り込み、後続の while の
    反復回数上界 (bound/step) を精密化する。
    """
    env = dict(env)
    total = sp.Integer(0)
    for s in stmts:
        total += _stmt_cost(s, env, gvar, stmt_cost_term)
        if (
            isinstance(s, ast.Assign)
            and len(s.targets) == 1
            and isinstance(s.targets[0], ast.Name)
        ):
            try:
                env[s.targets[0].id] = _to_expr(s.value, env)
            except _Unsupported:
                env.pop(s.targets[0].id, None)
    return total


def _stmt_cost(
    stmt: ast.stmt,
    env: dict[str, sp.Symbol],
    gvar: GeometricVar,
    stmt_cost_term,
) -> sp.Expr:
    if isinstance(stmt, ast.For):
        return _for_cost(stmt, env, gvar, stmt_cost_term)
    if isinstance(stmt, ast.While):
        return _while_cost(stmt, env, gvar, stmt_cost_term)
    if isinstance(stmt, ast.If):
        # 上界として両分岐の和 (big-O では max と同じオーダー)
        return (
            _stmts_cost(stmt.body, env, gvar, stmt_cost_term)
            + _stmts_cost(stmt.orelse, env, gvar, stmt_cost_term)
            + 1
        )
        # 条件式のコストは 1 に吸収
    # 幾何変数自身の更新・通常の文: 固定コスト (暗黙コールコスト込み)
    term = stmt_cost_term(stmt)
    return _term_to_expr(term) + 1


def _for_cost(
    stmt: ast.For,
    env: dict[str, sp.Symbol],
    gvar: GeometricVar,
    stmt_cost_term,
) -> sp.Expr:
    """for i in range(...) のコスト: Sum の閉形式。"""
    if not (
        isinstance(stmt.iter, ast.Call)
        and isinstance(stmt.iter.func, ast.Name)
        and stmt.iter.func.id == "range"
        and isinstance(stmt.target, ast.Name)
    ):
        raise _Unsupported
    args = stmt.iter.args
    if len(args) == 1:
        lo, hi, step = sp.Integer(0), _to_expr(args[0], env), sp.Integer(1)
    elif len(args) == 2:
        lo, hi, step = _to_expr(args[0], env), _to_expr(args[1], env), sp.Integer(1)
    elif len(args) == 3:
        lo, hi = _to_expr(args[0], env), _to_expr(args[1], env)
        step = _to_expr(args[2], env)
    else:
        raise _Unsupported

    ivar = sp.Symbol(stmt.target.id, positive=True)
    inner_env = dict(env)
    inner_env[stmt.target.id] = ivar
    body = _stmts_cost(stmt.body, inner_env, gvar, stmt_cost_term)

    # Max は Sum の閉形式化を妨げるため使わない (負値は数値評価時に 0 に切る)
    count = (hi - lo) / step
    if ivar not in body.free_symbols:
        return count * body
    # 本体が i に依存 → Sum の閉形式 (step は等間隔とみなし i を lo..hi で総和/step)
    try:
        summed = sp.Sum(body, (ivar, lo, hi - 1)).doit() / step
    except Exception as e:  # noqa: BLE001
        raise _Unsupported from e
    if summed.has(sp.Sum):
        raise _Unsupported
    return summed


def _while_cost(
    stmt: ast.While,
    env: dict[str, sp.Symbol],
    gvar: GeometricVar,
    stmt_cost_term,
) -> sp.Expr:
    """内側 while のコスト: ステップ幅から反復回数の上界を推定。"""
    test_names = {
        n.id for n in ast.walk(stmt.test) if isinstance(n, ast.Name)
    }
    # ループ変数 v と加減ステップ s を探す (v -= s / v += s / v = v - s)
    for sub in ast.walk(ast.Module(body=stmt.body, type_ignores=[])):
        step_expr = None
        if (
            isinstance(sub, ast.AugAssign)
            and isinstance(sub.target, ast.Name)
            and sub.target.id in test_names
            and isinstance(sub.op, (ast.Add, ast.Sub))
        ):
            step_expr = sub.value
        elif (
            isinstance(sub, ast.Assign)
            and len(sub.targets) == 1
            and isinstance(sub.targets[0], ast.Name)
            and sub.targets[0].id in test_names
            and isinstance(sub.value, ast.BinOp)
            and isinstance(sub.value.op, (ast.Add, ast.Sub))
        ):
            step_expr = sub.value.right
        if step_expr is None:
            continue
        step = _to_expr(step_expr, env)
        # 反復範囲の上界: ループ変数の初期値 (j = i なら i)、不明なら n
        bound = _N
        for name in test_names:
            if name in env and env[name] not in (_N, _G):
                bound = env[name]
                break
        body = _stmts_cost(
            [s for s in stmt.body if s is not sub], env, gvar, stmt_cost_term
        )
        # Max を避け、+1 で正値を保つ (Sum の閉形式化を妨げないため)
        count = bound / step + 1
        return count * (body + 1)
    raise _Unsupported


def _term_to_expr(term: ComplexityTerm) -> sp.Expr:
    """ComplexityTerm → sympy 式 (n^p · log^k n)。"""
    return _N ** term.poly * sp.log(_N + 1) ** term.log


def _to_expr(node: ast.expr, env: dict[str, sp.Symbol]) -> sp.Expr:
    """制限付きの式変換: 定数・既知変数・len()・四則演算のみ。"""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return sp.Float(node.value) if isinstance(node.value, float) else sp.Integer(
            node.value
        )
    if isinstance(node, ast.Name):
        if node.id in env:
            return env[node.id]
        raise _Unsupported
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        if node.func.id == "len" and node.args:
            return _N  # リスト長は n とみなす
        if node.func.id == "int" and len(node.args) == 1:
            return _to_expr(node.args[0], env)
        if node.func.id in ("max", "min") and node.args:
            parts = [_to_expr(a, env) for a in node.args]
            return sp.Max(*parts) if node.func.id == "max" else sp.Min(*parts)
        raise _Unsupported
    if isinstance(node, ast.BinOp):
        left = _to_expr(node.left, env)
        right = _to_expr(node.right, env)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, (ast.Div, ast.FloorDiv)):
            return left / right
        raise _Unsupported
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_to_expr(node.operand, env)
    raise _Unsupported


# ---------- 分類 ----------

#: 分類候補 (poly, log)
_CANDIDATES = [
    (p, k) for p in range(0, 4) for k in range(0, 3) if (p, k) != (0, 0)
] + [(0, 0)]


def _classify_counts(
    sizes: list[int], totals: list[float]
) -> tuple[ComplexityTerm | None, float]:
    """厳密な反復回数カーブを n^p log^k n 候補へフィットして分類する。"""
    n = np.array(sizes, dtype=float)
    t = np.array([max(v, 1.0) for v in totals], dtype=float)
    log_t = np.log(t)

    best_pair: tuple[int, int] | None = None
    best_rss = math.inf
    for p, k in _CANDIDATES:
        log_f = p * np.log(n) + k * np.log(np.log(n))
        log_c = float(np.mean(log_t - log_f))
        rss = float(np.sum((log_t - (log_c + log_f)) ** 2))
        if rss < best_rss:
            best_rss = rss
            best_pair = (p, k)

    slope = float(np.polyfit(np.log(n), log_t, 1)[0])
    # 残差が大きい (非整数指数など既知形に合わない) 場合は不明扱い
    k_pts = len(sizes)
    if best_rss / k_pts > 0.01:
        return None, slope
    term = ComplexityTerm(*best_pair)
    if term.to_class().name == "UNKNOWN":
        return None, slope
    return term, slope
