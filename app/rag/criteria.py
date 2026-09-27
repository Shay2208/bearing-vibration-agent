"""确定性判据核对引擎：把知识条目的具名判据条款逐条与本次振动事实核对。

分工（对应「规则能判的规则判」这一层）：
  - 检索负责「找条款」：条目级打分给出候选，条款级 BM25 挑出与本次查询最相关的判据文本；
  - 核对负责「验条款」：按条款自带的 metric / op / value 与结构化事实逐条判定，只输出三态。

三态：`hit`（成立）/ `miss`（不成立）/ `unknown`（指标缺失，不下判，不算不成立）。
因子：`factor = 0.4 + 0.6 × 命中 / 已核对数`；全部命中为 1.0，全部已核对条款都与数据矛盾时降到 0.4；
      没有任何可核对指标时不罚分（factor=1.0，state=unverified），只标记「未核对」。
排序：`alignment_score = 检索分 × factor`，让「检索相关但判据与数据矛盾」的条目自然下沉。

条款 value 以字符串存放在知识库中，按实测值类型在此转换（数值 / 布尔），
不使用 eval，metric 必须是这里的受控词表，op 必须是受控运算符。
"""

from __future__ import annotations

import math
import operator
from typing import Any, Callable

# 判据因子：全部命中 = 1.0，全部已核对条款均不成立 = 0.4
_FACTOR_BASE = 0.4
_FACTOR_SPAN = 0.6

# 转频倍次定位容差（主频落在 n×fr 的 ±10% 内视为对齐）
_SHAFT_TOLERANCE = 0.1


def _num(value) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


# ---------------------------------------------------------------------------
# 指标解析器：metric → 取值函数，返回 None 表示该指标本次不可核对（记 unknown）
# ---------------------------------------------------------------------------
def _comparison(ctx: dict) -> dict:
    return ctx.get("comparison") or {}


def _ratio(ctx: dict, name: str) -> float | None:
    block = (_comparison(ctx).get("feature_changes") or {}).get(name) or {}
    return _num(block.get("ratio"))


def _changed_feature_count(ctx: dict, _spec: dict) -> float | None:
    comparison = _comparison(ctx)
    if "changed_features" not in comparison:
        return None
    return float(len(comparison.get("changed_features") or []))


def _dominant_shift_magnitude(ctx: dict, _spec: dict) -> float | None:
    block = _comparison(ctx).get("frequency_change") or {}
    value = _num(block.get("shift_hz"))
    return None if value is None else abs(value)


def _shaft_frequency(ctx: dict) -> float | None:
    return _num((_comparison(ctx).get("fault_frequencies") or {}).get("shaft_frequency"))


def _dominant_frequency(ctx: dict) -> float | None:
    value = _num((ctx.get("features") or {}).get("dominant_frequency"))
    if value is None:
        value = _num((_comparison(ctx).get("frequency_change") or {}).get("abnormal_dominant"))
    return value


def _dominant_near_shaft(ctx: dict, spec: dict) -> bool | None:
    order = _num(spec.get("order")) or 1.0
    shaft = _shaft_frequency(ctx)
    dominant = _dominant_frequency(ctx)
    if shaft is None or dominant is None or shaft <= 0:
        return None
    target = order * shaft
    return abs(dominant - target) / target <= _SHAFT_TOLERANCE


def _envelope_block(ctx: dict, spec: dict) -> dict | None:
    family = str(spec.get("family") or "").strip().lower()
    changes = _comparison(ctx).get("characteristic_energy_changes") or {}
    if not family or family not in changes:
        return None
    return changes.get(family) or None


def _envelope_energy(ctx: dict, spec: dict) -> float | None:
    block = _envelope_block(ctx, spec)
    return None if block is None else _num(block.get("abnormal"))


def _envelope_ratio(ctx: dict, spec: dict) -> float | None:
    block = _envelope_block(ctx, spec)
    return None if block is None else _num(block.get("ratio"))


def _envelope_rank(ctx: dict, spec: dict) -> float | None:
    if _envelope_block(ctx, spec) is None:
        return None
    changes = _comparison(ctx).get("characteristic_energy_changes") or {}
    family = str(spec.get("family") or "").strip().lower()
    ranked = sorted(changes, key=lambda key: _num((changes[key] or {}).get("abnormal")) or 0.0, reverse=True)
    return float(ranked.index(family) + 1)


def _envelope_margin(ctx: dict, _spec: dict) -> float | None:
    dominant = _comparison(ctx).get("dominant_characteristic") or {}
    return _num(dominant.get("margin_ratio"))


def _dominant_ambiguous(ctx: dict, _spec: dict) -> bool | None:
    dominant = _comparison(ctx).get("dominant_characteristic")
    if not isinstance(dominant, dict):
        return None
    return bool(dominant.get("ambiguous"))


def _peak_match_orders(ctx: dict, spec: dict) -> float | None:
    matches = _comparison(ctx).get("fault_frequency_matches")
    family = str(spec.get("family") or "").strip().upper()
    if matches is None or not family:
        return None
    return float(
        sum(
            1
            for item in matches
            if str(item.get("label") or "").strip().upper() == family and item.get("harmonic_order")
        )
    )


def _tied_family(ctx: dict, spec: dict) -> bool | None:
    dominant = _comparison(ctx).get("dominant_characteristic") or {}
    tied = dominant.get("tied_candidates") or []
    family = str(spec.get("family") or "").strip().upper()
    if not dominant.get("ambiguous") or not tied or not family:
        return None
    return any(str(item.get("label") or "").strip().upper() == family for item in tied)


METRICS: dict[str, Callable[[dict, dict], Any]] = {
    "rms_ratio": lambda ctx, _spec: _ratio(ctx, "rms"),
    "peak_ratio": lambda ctx, _spec: _ratio(ctx, "peak"),
    "kurtosis_ratio": lambda ctx, _spec: _ratio(ctx, "kurtosis"),
    "crest_factor_ratio": lambda ctx, _spec: _ratio(ctx, "crest_factor"),
    "dominant_amplitude_ratio": lambda ctx, _spec: _ratio(ctx, "dominant_amplitude"),
    "changed_feature_count": _changed_feature_count,
    "dominant_shift_hz": _dominant_shift_magnitude,
    "dominant_near_shaft": _dominant_near_shaft,
    "envelope_energy": _envelope_energy,
    "envelope_ratio": _envelope_ratio,
    "envelope_rank": _envelope_rank,
    "envelope_margin": _envelope_margin,
    "dominant_ambiguous": _dominant_ambiguous,
    "peak_match_orders": _peak_match_orders,
    "tied_family": _tied_family,
}

OPS: dict[str, Callable[[Any, Any], bool]] = {
    ">=": operator.ge,
    "<=": operator.le,
    ">": operator.gt,
    "<": operator.lt,
    "==": operator.eq,
    "!=": operator.ne,
}

_TRUE_WORDS = {"true", "1", "yes", "是", "成立"}
_FALSE_WORDS = {"false", "0", "no", "否", "不成立"}


def _coerce(expected: str, actual: Any):
    """按实测值类型转换条款声明值；无法转换时返回 None。"""
    text = str(expected).strip()
    if isinstance(actual, bool):
        lowered = text.lower()
        if lowered in _TRUE_WORDS:
            return True
        if lowered in _FALSE_WORDS:
            return False
        return None
    value = _num(text)
    if value is None:
        return text if isinstance(actual, str) else None
    return value


def _expected_text(spec: dict) -> str:
    metric = spec.get("metric")
    family = spec.get("family")
    target = f"{metric}({family})" if family else str(metric)
    return f"{target} {spec.get('op')} {spec.get('value')}"


def check_criterion(criterion: dict, ctx: dict) -> dict:
    """核对单条判据，返回带 state / actual 的条款行（metric 或 op 非法记 unknown）。"""
    spec = dict(criterion or {})
    metric = str(spec.get("metric") or "")
    op_name = str(spec.get("op") or "")
    row = {
        "id": spec.get("id"),
        "claim": spec.get("claim", ""),
        "metric": metric,
        "family": spec.get("family"),
        "op": op_name,
        "value": spec.get("value"),
        "expect": _expected_text(spec),
        "actual": None,
        "state": "unknown",
    }
    resolver = METRICS.get(metric)
    compare = OPS.get(op_name)
    if resolver is None or compare is None:
        return row
    actual = resolver(ctx, spec)
    row["actual"] = actual
    if actual is None:
        return row
    expected = _coerce(spec.get("value"), actual)
    if expected is None:
        return row
    try:
        row["state"] = "hit" if compare(actual, expected) else "miss"
    except TypeError:
        row["state"] = "unknown"
    return row


def summarize_state(hits: int, misses: int, unknowns: int, total: int) -> str:
    if total == 0:
        return "none"
    if hits == 0 and misses == 0:
        return "unverified"
    if misses == 0:
        return "consistent"
    if hits == 0:
        return "conflict"
    return "partial" if misses < hits else "conflict"


def evaluate(criteria: list[dict] | None, ctx: dict) -> dict:
    """核对一个条目的全部判据条款，返回三态计数、状态与判据因子。"""
    items = [check_criterion(item, ctx) for item in (criteria or [])]
    hits = sum(1 for item in items if item["state"] == "hit")
    misses = sum(1 for item in items if item["state"] == "miss")
    unknowns = sum(1 for item in items if item["state"] == "unknown")
    checked = hits + misses
    if not items:
        factor = 1.0
    elif checked == 0:
        # 本次没有任何可核对指标：不罚分，只标记「未核对」
        factor = 1.0
    else:
        factor = _FACTOR_BASE + _FACTOR_SPAN * (hits / checked)
    return {
        "state": summarize_state(hits, misses, unknowns, len(items)),
        "hits": hits,
        "misses": misses,
        "unknowns": unknowns,
        "checked": checked,
        "total": len(items),
        "factor": round(factor, 4),
        "items": items,
    }


def evaluate_candidates(
    candidates: list[dict], comparison: dict | None, features: dict | None = None
) -> list[dict]:
    """给候选附加判据核对结果与 alignment_score，并按 alignment_score 重排。

    没有任何对比事实（comparison 为空，例如纯文本检索）时原样返回，不做核对。
    """
    candidates = list(candidates or [])
    if not comparison:
        return candidates
    ctx = {"comparison": comparison, "features": features or {}}
    enriched: list[dict] = []
    for candidate in candidates:
        item = dict(candidate)
        block = evaluate(candidate.get("criteria") or [], ctx)
        item["criteria_check"] = block
        score = _num(candidate.get("score")) or 0.0
        item["alignment_score"] = round(score * block["factor"], 4)
        enriched.append(item)
    enriched.sort(key=lambda item: item.get("alignment_score") or 0.0, reverse=True)
    return enriched