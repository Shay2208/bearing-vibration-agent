"""模板化诊断报告：完全确定性生成，不依赖任何模型。

职责：
  - 「数据事实」来自 MCP 对比结果（comparison.evidence）；
  - 「知识依据」来自 RAG 候选（evidence + source + locator），每条主要判断都能追溯到来源；
  - 「推断结论」在候选充分时给出最可能故障类型，证据不足时输出「无法确认」；
  - 字段结构与接真实模型时逐字段一致（模型只影响 conclusion / inferences 的措辞）。
"""

from __future__ import annotations

# 报告 markdown 的 11 个小节标题（顺序即渲染顺序）
MARKDOWN_SECTIONS = [
    "分析对象与输入信息",
    "数据质量检查",
    "正常与异常特征对比表",
    "频谱或频带变化说明",
    "RAG 检索到的故障候选",
    "最终结论或无法确认说明",
    "结论对应的振动证据",
    "结论对应的知识来源",
    "置信说明",
    "建议复核的数据或现场检查项目",
    "项目原型的适用边界",
]

# 特征键 → 中文标签（顺序固定）
FEATURE_ORDER = [
    ("rms", "RMS"),
    ("peak", "峰值"),
    ("kurtosis", "峭度"),
    ("crest_factor", "波峰因子"),
    ("dominant_frequency", "主频(Hz)"),
    ("dominant_amplitude", "主频幅值"),
]

# 条款核对三态的中文标签（状态枚举与 app/rag/criteria.py 一一对应）
_CRITERIA_STATE_LABEL = {
    "consistent": "条款全部命中",
    "partial": "条款部分命中",
    "conflict": "条款与数据冲突",
    "unverified": "条款缺少可核对指标",
    "none": "条目未声明判据条款",
}

# 各故障类型建议核对的特征频率
_FAULT_FREQUENCY_HINT = {
    "inner_race_fault": "BPFI 及其 2×BPFI、3×BPFI，并检查谱峰两侧是否存在转频 fr 间隔边带",
    "outer_race_fault": "BPFO 及其 2 倍、3 倍谐波，并核对峰值偏差是否在工程容许范围内",
    "ball_fault": "BSF 与 2×BSF（滚动体故障通常以 2×BSF 为主），并留意峰幅波动",
    "cage_fault": "FTF 及其谐波，但 FTF 能量通常很弱，需借助包络解调确认",
}

PROTOTYPE_LIMITS = [
    "仅适配单列深沟球轴承（CWRU 6205-2RS JEM SKF 驱动端）与单点加速度信号，其他轴承型号、测点或传感器类型未验证。",
    "只覆盖 0 hp / 1797 rpm 一种转速负载工况，变转速、变负载下特征频率换算与判定阈值均未验证。",
    "每个样本仅 1 秒（12000 点）片段，样本量不足以支撑统计意义上可靠的诊断或模型训练。",
    "知识库条目有限，RAG 候选排序能力受限于知识覆盖范围；检索为空时本原型只输出「无法确认」。",
    "包络解调采用按奈奎斯特频率固定比例（0.33~0.83 倍）选定的共振带，未使用谱峭度/kurtogram 自适应选带；不做故障严重程度分级与剩余寿命预测。",
    "输出为工程辅助建议，不能替代现场复测、拆检与专业人员判断。",
    "样本真实故障标签仅用于离线评测，不进入在线诊断链路。",
]


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------
def _num(value, digits: int = 4) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _pct(ratio) -> str:
    """ratio 是 abnormal/normal 的倍数，这里换算成相对变化百分比。"""
    if ratio is None:
        return "-"
    try:
        return f"{(float(ratio) - 1.0) * 100:+.0f}%"
    except (TypeError, ValueError):
        return str(ratio)


def _times(ratio) -> str:
    """ratio 是倍数 abnormal/normal，按量级选择可读精度。"""
    if ratio is None:
        return "-"
    try:
        value = float(ratio)
    except (TypeError, ValueError):
        return str(ratio)
    if value >= 100:
        return f"{value:.0f} 倍"
    if value >= 10:
        return f"{value:.1f} 倍"
    return f"{value:.2f} 倍"


def _cell(value) -> str:
    return "-" if value is None else str(value)


def _actual(value) -> str:
    """条款实测值：布尔按「是/否」写，其余按数值格式化。"""
    if isinstance(value, bool):
        return "是" if value else "否"
    return _num(value, 4)


def _bullets(items) -> str:
    items = [str(i) for i in (items or []) if str(i).strip()]
    return "\n".join(f"- {i}" for i in items) if items else "- （无）"


def _table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 各段落生成
# ---------------------------------------------------------------------------
def _analysis_target(device: dict) -> str:
    return (
        f"{device.get('device_type') or '旋转设备'} · {device.get('sensor_position') or '未指定测点'} · "
        f"转速 {device.get('rotation_speed')} rpm · 采样率 {device.get('sampling_rate')} Hz · "
        f"异常样本 {device.get('abnormal_file')}（基线 {device.get('normal_file')}）"
    )


def _data_quality_summary(data_quality: dict | None) -> str:
    if not data_quality:
        return "未执行数据校验。"
    checks = data_quality.get("checks") or {}
    failed = [name for name, item in checks.items() if not (item or {}).get("passed")]
    counts = f"正常样本 {((data_quality.get('normal') or {}).get('n_samples'))} 点，异常样本 {((data_quality.get('abnormal') or {}).get('n_samples'))} 点"
    verdict = "校验通过" if data_quality.get("valid") else "校验未通过"
    comparable = "两样本可比" if data_quality.get("comparable") else "两样本不可比"
    parts = [f"{verdict}、{comparable}（{counts}）。"]
    if failed:
        parts.append("未通过的检查项：" + "、".join(failed) + "。")
    errors = data_quality.get("errors") or []
    if errors:
        parts.append("错误：" + "；".join(f"{e.get('code')}（{e.get('message')}）" for e in errors[:4]) + "。")
    warnings = data_quality.get("warnings") or []
    if warnings:
        parts.append("提示：" + "；".join(str(w.get("message")) for w in warnings[:3]) + "。")
    return "".join(parts)


def _change_note(key: str, block: dict) -> str:
    delta = block.get("delta")
    ratio = block.get("ratio")
    if key == "dominant_frequency":
        if delta is None:
            return "无数据"
        if abs(float(delta)) < 1e-9:
            return "主频未迁移"
        return f"主频{'上移' if float(delta) > 0 else '下移'} {abs(float(delta)):.1f}Hz"
    if ratio is None:
        return "缺少基准值，未计算变化倍数"
    ratio = float(ratio)
    if abs(ratio - 1.0) < 0.3:
        return "变化不显著（<30%）"
    trend = "上升" if ratio > 1.0 else "下降"
    return f"{trend} {abs(ratio - 1.0) * 100:.0f}%（约 {_times(ratio)}）"


def _feature_comparison(comparison: dict | None) -> list[dict]:
    changes = (comparison or {}).get("feature_changes") or {}
    rows = []
    for key, label in FEATURE_ORDER:
        block = changes.get(key) or {}
        rows.append({
            "feature": label,
            "normal": block.get("normal"),
            "abnormal": block.get("abnormal"),
            "delta": block.get("delta"),
            "ratio": block.get("ratio"),
            "note": _change_note(key, block),
        })
    return rows


def _band_energy_comparison(comparison: dict | None) -> list[dict]:
    changes = (comparison or {}).get("band_energy_changes") or {}
    return [
        {"band": band, "normal": (b or {}).get("normal"), "abnormal": (b or {}).get("abnormal"), "ratio": (b or {}).get("ratio")}
        for band, b in changes.items()
    ]


def _envelope_spectrum_note(comparison: dict) -> str | None:
    """包络谱说明；comparison 未做包络分析（无 characteristic_energy_changes）时返回 None。

    能量值统一用科学计数法（2 位有效数字）书写：包络能量跨越 1e-7 ~ 1e-1 量级，
    定点小数会把小量级写成 0.000000，看不出量级差异。
    """
    changes = comparison.get("characteristic_energy_changes")
    if not changes:
        return None

    def energy(value) -> str:
        try:
            return f"{float(value):.2e}"
        except (TypeError, ValueError):
            return "-"

    def baseline_suffix(block: dict) -> str:
        # 基线未检出该成分时倍数无意义，直接标注「基线未检出」，不编造倍数
        if not block.get("detected_in_baseline"):
            return "基线未检出"
        return f"基线 {energy(block.get('normal'))}，约 {_times(block.get('ratio'))}"

    segments: list[str] = []
    dominant = comparison.get("dominant_characteristic")
    tied = (dominant or {}).get("tied_candidates") or []
    if tied:
        # 并列时不说"能量最高"：逐族列出能量，让复核自行判断
        items = []
        for item in tied:
            name = str(item.get("label") or "").lower()
            block = changes.get(name) or {}
            items.append(
                f"{name.upper()}（{_num(item.get('frequency_hz'), 1)}Hz）"
                f"{energy(block.get('abnormal', item.get('abnormal_energy')))}（{baseline_suffix(block)}）"
            )
        segments.append("包络谱特征频率族能量接近、无单一主导：" + "；".join(items))
    elif dominant:
        label = str(dominant.get("label") or "").upper()
        block = changes.get(label.lower()) or {}
        # 优先用逐 label 的 abnormal 原值：dominant.abnormal_energy 被算法四舍五入，
        # 能量极小时会显示为 0.00e+00。
        abnormal_energy = block.get("abnormal", dominant.get("abnormal_energy"))
        segments.append(
            f"包络谱中 {label}（{_num(dominant.get('frequency_hz'), 1)}Hz）及其 2、3 阶谐波频带能量最高"
            f"（{energy(abnormal_energy)}，{baseline_suffix(block)}）"
        )

    ranked = sorted(
        changes.items(),
        key=lambda kv: float((kv[1] or {}).get("abnormal") or 0.0),
        reverse=True,
    )[:3]
    if ranked:
        segments.append(
            "各特征频率包络能量："
            + "、".join(f"{name.upper()} {energy((block or {}).get('abnormal'))}" for name, block in ranked)
        )

    if not dominant:
        segments.append("包络谱中未出现与 BPFO/BPFI/BSF/FTF 对齐的显著成分")
    return "；".join(segments)


def _spectrum_notes(comparison: dict | None) -> str:
    if not comparison:
        return "无对比数据，无法说明频谱与频带变化。"
    parts: list[str] = []

    freq_change = comparison.get("frequency_change") or {}
    if freq_change:
        normal, abnormal = freq_change.get("normal_dominant"), freq_change.get("abnormal_dominant")
        if normal == abnormal:
            parts.append(f"主频保持 {_num(normal, 1)}Hz 未迁移")
        else:
            shift = freq_change.get("shift_hz")
            parts.append(f"主频由 {_num(normal, 1)}Hz 迁移至 {_num(abnormal, 1)}Hz（偏移 {shift:+.1f}Hz）")

    ranked = sorted(
        _band_energy_comparison(comparison),
        key=lambda b: abs(float(b.get("abnormal") or 0.0) - float(b.get("normal") or 0.0)),
        reverse=True,
    )[:2]
    if ranked:
        parts.append(
            "频带能量变化最明显的是 "
            + "，".join(f"{b['band']} 由 {_num(b['normal'], 6)} 变为 {_num(b['abnormal'], 6)}（约 {_times(b['ratio'])}）" for b in ranked)
        )

    fault_freqs = comparison.get("fault_frequencies") or {}
    if fault_freqs:
        parts.append(
            f"按转速 {_num(fault_freqs.get('rotation_speed'), 1)}rpm 换算的理论特征频率："
            f"转频 {_num(fault_freqs.get('shaft_frequency'), 2)}Hz、BPFO {_num(fault_freqs.get('bpfo'), 2)}Hz、"
            f"BPFI {_num(fault_freqs.get('bpfi'), 2)}Hz、BSF {_num(fault_freqs.get('bsf'), 2)}Hz、FTF {_num(fault_freqs.get('ftf'), 2)}Hz"
        )

    matches = comparison.get("fault_frequency_matches") or []
    if matches:
        segs = []
        for match in matches:
            order = match.get("harmonic_order") or 1
            base = fault_freqs.get(str(match.get("label", "")).lower())
            segs.append(
                f"{match.get('label')}{'×' + str(order) if order > 1 else ''} "
                f"{_num(base or match.get('frequency'), 2)}Hz 附近存在谱峰 {_num(match.get('nearest_peak'), 2)}Hz"
                f"（偏差 {_num(match.get('deviation_pct'), 2)}%，幅值 {_num(match.get('peak_amplitude'))}）"
            )
        parts.append("谱峰与特征频率对齐情况：" + "；".join(segs))
    elif fault_freqs:
        parts.append("异常样本谱峰中未检测到与 BPFO/BPFI/BSF/FTF（含 2、3 次谐波）对齐的显著峰值")

    envelope_note = _envelope_spectrum_note(comparison)
    if envelope_note:
        parts.append(envelope_note)

    changed = comparison.get("changed_features") or []
    if changed:
        labels = dict(FEATURE_ORDER)
        parts.append("发生显著变化的特征：" + "、".join(labels.get(k, k) for k in changed))

    return "；".join(parts) + "。" if parts else "频谱未出现可说明的变化。"


def _criteria_rows(candidate: dict) -> list[dict]:
    """把候选的条款核对结果展开成报告行（条款 ID / 判据 / 三态 / 期望 / 实测）。"""
    items = ((candidate or {}).get("criteria_check") or {}).get("items") or []
    return [
        {
            "clause_id": item.get("id"),
            "claim": item.get("claim"),
            "family": item.get("family"),
            "state": item.get("state"),
            "expect": item.get("expect"),
            "actual": item.get("actual"),
        }
        for item in items
    ]


def _missed_clauses(candidate: dict) -> list[str]:
    """候选里未命中的条款 ID（与数据事实矛盾的条款）。"""
    return [
        str(item.get("id"))
        for item in ((candidate or {}).get("criteria_check") or {}).get("items") or []
        if item.get("state") == "miss"
    ]


def _criteria_note(candidate: dict) -> str | None:
    """条款核对结论的中文说明；候选没有条款核对结果时返回 None。"""
    check = (candidate or {}).get("criteria_check") or {}
    if not check.get("items"):
        return None
    label = _CRITERIA_STATE_LABEL.get(check.get("state"), str(check.get("state")))
    note = (
        f"判据条款核对（{label}）：共 {check.get('total')} 条，命中 {check.get('hits')} 条、"
        f"未命中 {check.get('misses')} 条、未核对 {check.get('unknowns')} 条，判据因子 {check.get('factor')}。"
    )
    missed = _missed_clauses(candidate)
    if missed:
        note += "与数据事实矛盾的条款：" + "、".join(missed) + "。"
    return note


def _knowledge_candidates(candidates: list[dict]) -> list[dict]:
    rows = []
    for c in candidates or []:
        check = c.get("criteria_check") or {}
        rows.append({
            "fault_type": c.get("fault_type"),
            "evidence": c.get("evidence"),
            "source": c.get("source"),
            "locator": c.get("locator"),
            "score": c.get("score"),
            "alignment_score": c.get("alignment_score"),
            "criteria_state": check.get("state"),
            "criteria_hits": check.get("hits"),
            "criteria_misses": check.get("misses"),
            "criteria_unknowns": check.get("unknowns"),
            "criteria_factor": check.get("factor"),
            "criteria": _criteria_rows(c),
        })
    return rows


def _tied_note(comparison: dict | None) -> str | None:
    """并列族的中文说明；未并列时返回 None。"""
    dominant = (comparison or {}).get("dominant_characteristic") or {}
    if not dominant.get("ambiguous"):
        return None
    names = "、".join(
        str(item.get("label")) for item in dominant.get("tied_candidates") or []
    ) or str(dominant.get("label") or "")
    return (
        f"包络谱特征频率族能量并列（{names}，最高/次高能量比 {_num(dominant.get('margin_ratio'), 4)} "
        "低于量级余量阈值），主导特征频率族不可判定"
    )


def _retrieval_confidence(candidates: list[dict], comparison: dict | None = None) -> dict:
    tied = _tied_note(comparison)
    if tied:
        if not candidates:
            return {"level": "low", "reason": f"{tied}，且无候选项与并列族匹配，按证据不足处理，无法形成置信判断。"}
        top = candidates[0]
        return {
            "level": "low",
            "reason": (
                f"{tied}；本次结论由检索首位候选 {top.get('fault_type')} 声明的特征频率族 "
                f"{top.get('frequency_family')} 落在并列族内推出，属并列证据下的采信，置信度下调为低。"
            ),
        }
    if not candidates:
        return {"level": "low", "reason": "未检索到任何候选故障类型，证据不足，无法形成置信判断。"}

    top = candidates[0]
    score = float(top.get("score") or 0.0)
    count = len(candidates)
    if count < 2:
        return {
            "level": "low",
            "reason": f"仅检索到 1 条候选（{top.get('fault_type')}，score={score:.4f}），缺少交叉印证，证据强度不足。",
        }
    second = candidates[1]
    second_score = float(second.get("score") or 0.0)
    if score < 0.5:
        return {
            "level": "low",
            "reason": f"top 候选 {top.get('fault_type')} 的 score={score:.4f} 低于 0.5，检索匹配偏弱，仅作参考。",
        }
    if score >= 0.6:
        gap = score - second_score
        if gap >= 0.08:
            return {
                "level": "high",
                "reason": (
                    f"top 候选 {top.get('fault_type')} score={score:.4f} ≥ 0.6，且领先次候选 "
                    f"{second.get('fault_type')}（{second_score:.4f}）{gap:.4f}，共 {count} 条候选，证据相对集中。"
                ),
            }
        return {
            "level": "medium",
            "reason": (
                f"top 候选 {top.get('fault_type')} score={score:.4f} ≥ 0.6，但次候选 "
                f"{second.get('fault_type')}（{second_score:.4f}）接近，候选间区分度有限，共 {count} 条候选。"
            ),
        }
    return {
        "level": "medium",
        "reason": f"top 候选 {top.get('fault_type')} score={score:.4f} 处于 0.5~0.6 区间，匹配中等，共 {count} 条候选。",
    }


def _confidence(candidates: list[dict], comparison: dict | None = None) -> dict:
    """检索分给出的置信判断，再用条款核对结果修正：条款与数据冲突即下调为低并交人工确认。"""
    result = _retrieval_confidence(candidates, comparison)
    top = candidates[0] if candidates else {}
    if ((top.get("criteria_check") or {}).get("state")) != "conflict":
        note = _criteria_note(top)
        if note:
            result = {**result, "reason": f"{result['reason']} {note}"}
        return result
    return {
        "level": "low",
        "reason": (
            f"{result['reason']} {_criteria_note(top)}"
            "首位候选的判据条款与本次数据事实矛盾，规则层不采信该候选，置信度下调为低，需人工确认后再定论。"
        ),
    }


def _conclusion_sources(candidates: list[dict], rag: dict | None, chosen: str | None) -> list[dict]:
    sources = (rag or {}).get("sources") or []
    if not chosen:
        return []
    picked = [
        {"source": s.get("name"), "url": s.get("url"), "locator": s.get("locator")}
        for s in sources
        if s.get("fault_type") == chosen
    ]
    if picked:
        return picked
    for candidate in candidates or []:
        if candidate.get("fault_type") == chosen:
            return [{"source": candidate.get("source"), "url": candidate.get("source_url"), "locator": candidate.get("locator")}]
    return []


def _review_suggestions(data_quality: dict | None, candidates: list[dict], chosen: str | None, insufficient: bool) -> list[str]:
    suggestions: list[str] = []
    suggestions.append(
        f"数据层面：当前异常样本仅 {(data_quality or {}).get('abnormal', {}).get('n_samples', '未知')} 点"
        f"（约 1 秒），建议延长采集时长并在同工况下多次采集，确认特征是否可重复。"
    )
    suggestions.append("数据层面：核对采样率设置与时间戳是否一致，并确认测点位置、传感器安装方向在两次采集中保持一致。")
    if insufficient or not candidates:
        suggestions.append("数据层面：补充与异常样本严格同转速、同负载的健康基线样本，否则统计量对比的意义有限。")
        suggestions.append("现场层面：确认轴承型号与几何参数、实际转速，以便正确换算 BPFO/BPFI/BSF/FTF。")
    suggestions.append("现场层面：在驱动端径向（水平/垂直）与轴向多点复测，排除传感器安装松动、结构共振造成的伪峰。")
    if chosen:
        hint = _FAULT_FREQUENCY_HINT.get(chosen)
        if hint:
            suggestions.append(f"现场层面：按实际转速核算 {hint}，确认谱峰偏差是否在工程容许范围内。")
    suggestions.append("现场层面：结合包络解调或谱峭度选择最优解调频带后重建信号，确认冲击成分是否具有稳定周期性。")
    suggestions.append("现场层面：检查润滑状态、异物污染、轴承温度与运行时长，作为严重程度判断的辅助依据。")
    return suggestions


def _prototype_limits() -> list[str]:
    return list(PROTOTYPE_LIMITS)


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------
def build_report(
    *,
    device: dict,
    data_quality: dict | None,
    comparison: dict | None,
    rag: dict | None,
    candidates: list[dict] | None = None,
    llm_result: dict | None = None,
) -> dict:
    """确定性生成报告。llm_result 只用于改写 conclusion / inferences 的措辞。"""
    candidates = list(candidates or [])
    insufficient = not candidates
    top = candidates[0] if candidates else None
    chosen = None if insufficient else ((llm_result or {}).get("selected_fault_type") or top.get("fault_type"))

    evidence = [str(e) for e in ((comparison or {}).get("evidence") or [])]
    data_facts = evidence or ["未获得可用的对比证据。"]

    knowledge_basis: list[str] = []
    for candidate in candidates:
        state = (candidate.get("criteria_check") or {}).get("state")
        suffix = f"；判据核对 {_CRITERIA_STATE_LABEL.get(state, state)}" if state else ""
        knowledge_basis.append(
            f"{candidate.get('fault_type')}：{candidate.get('evidence')}"
            f"〔来源：{candidate.get('source')} · {candidate.get('locator')}；相似度 {candidate.get('score')}{suffix}〕"
        )
    if not knowledge_basis:
        knowledge_basis = ["检索阈值内未命中任何知识条目，本报告不做故障类型判断。"]

    # 结论
    if insufficient:
        observed = "；".join(data_facts[:3])
        conclusion = (
            f"无法确认：当前数据与知识库证据不足以判定具体故障类型。"
            f"已观察到的振动现象为：{observed or '数据未显示明确异常特征'}。"
            f"建议按下方复核项补充数据与现场信息后重新诊断。"
        )
        inferences = [
            "证据不足，故不输出任何具体故障类型（避免编造结论）。",
            "现有数据只能说明信号水平或结构发生变化，无法定位到具体轴承部件。",
            "需要同工况健康基线、更多测点与更长的采集时长，才可能形成可靠判断。",
        ]
    else:
        conclusion = (
            f"最可能故障类型：{top.get('title')}（{top.get('fault_type')}）。"
            f"判断依据：知识库候选给出的特征为「{top.get('evidence')}」，"
            f"与数据事实（{'；'.join(evidence[:2]) or '见对比证据'}）方向一致。"
            "该结论来自知识库候选检索，仍需按复核项在现场确认。"
        )
        if llm_result and (llm_result.get("conclusion") or "").strip():
            conclusion = str(llm_result["conclusion"]).strip()
        inferences = [
            f"检索候选指向 {top.get('title')}（{top.get('fault_type')}），其典型特征与本次观测到的变化方向一致。",
            (
                f"候选按 alignment_score（检索分 × 判据因子）排序：首位 {top.get('fault_type')} 检索分 "
                f"{top.get('score')}、alignment_score {top.get('alignment_score')}，共 {len(candidates)} 条候选；"
                "该排序反映候选与查询语句及本次数据事实的吻合程度，不等同于确诊。"
            ),
        ]
        if len(candidates) < 2:
            inferences.append("候选单一，缺少交叉印证，结论可靠性受限。")
        else:
            inferences.append(f"次候选为 {candidates[1].get('title')}（{candidates[1].get('fault_type')}），未完全排除复合故障的可能。")
        if llm_result and (llm_result.get("reason") or "").strip():
            inferences.append(f"模型补充理由：{str(llm_result['reason']).strip()}")
        tied = _tied_note(comparison)
        if tied:
            inferences.append(
                f"{tied}；本次结论由检索首位候选（{top.get('fault_type')}，声明特征频率族 "
                f"{top.get('frequency_family')}）与并列族一致推出，属并列证据下的采信，置信度已下调为低，"
                "但仍需人工复核各并列族的谐波能量与转频边带。"
            )
        note = _criteria_note(top)
        if note:
            inferences.append(note)

    confidence = _confidence(candidates, comparison)
    conclusion_evidence = evidence[:4] or ["未获得可用的对比证据。"]
    conclusion_sources = _conclusion_sources(candidates, rag, chosen)

    # 复核建议：以确定性建议为骨架，模型成功返回时追加其结构化 review_suggestions（去重）
    suggestions = _review_suggestions(data_quality, candidates, chosen, insufficient)
    if not insufficient and ((top.get("criteria_check") or {}).get("state") == "conflict"):
        suggestions.append(
            f"现场层面：首位候选（{top.get('fault_type')}）的判据条款 {'、'.join(_missed_clauses(top))} "
            "与本次数据事实不符，规则层不采信该候选，请按第 5 节条款核对表逐条人工确认（必要时补采数据）后再定论。"
        )
    tied = _tied_note(comparison)
    if not insufficient and tied:
        names = "、".join(
            str(item.get("label")) for item in ((comparison or {}).get("dominant_characteristic") or {}).get("tied_candidates") or []
        )
        suggestions.append(
            f"现场层面：包络谱中 {names} 特征频率族能量并列、无单一主导族，请分别核对各并列族"
            "（含 2×BSF 与转频边带）的谐波能量与周期稳定性，确认是否属能量分散型损伤表现后再决定是否拆检。"
        )
    for item in (llm_result or {}).get("review_suggestions") or []:
        text = str(item).strip()
        if text and text not in suggestions:
            suggestions.append(text)

    return {
        "analysis_target": _analysis_target(device),
        "data_quality_summary": _data_quality_summary(data_quality),
        "feature_comparison": _feature_comparison(comparison),
        "band_energy_comparison": _band_energy_comparison(comparison),
        "spectrum_notes": _spectrum_notes(comparison),
        "knowledge_candidates": _knowledge_candidates(candidates),
        "conclusion": conclusion,
        "conclusion_evidence": conclusion_evidence,
        "conclusion_sources": conclusion_sources,
        "confidence": confidence,
        "data_facts": data_facts,
        "knowledge_basis": knowledge_basis,
        "inferences": inferences,
        "review_suggestions": suggestions,
        "prototype_limits": _prototype_limits(),
    }


# ---------------------------------------------------------------------------
# markdown 渲染
# ---------------------------------------------------------------------------
def render_markdown(report: dict, context: dict) -> str:
    report = report or {}
    context = context or {}
    device = context.get("device") or {}
    data_quality = context.get("data_quality") or {}
    rag = context.get("rag") or {}
    trace = context.get("trace") or {}
    candidates = context.get("candidates") or []

    lines: list[str] = ["# 轴承振动诊断报告", ""]

    # 1
    lines += [f"## 1. {MARKDOWN_SECTIONS[0]}", "", report.get("analysis_target", ""), ""]
    lines += [
        _table(
            ["项目", "取值"],
            [
                ["设备类型", device.get("device_type") or "-"],
                ["采样率 (Hz)", _cell(device.get("sampling_rate"))],
                ["转速 (rpm)", _cell(device.get("rotation_speed"))],
                ["测点", device.get("sensor_position") or "-"],
                ["健康基线文件", device.get("normal_file") or "-"],
                ["异常样本文件", device.get("abnormal_file") or "-"],
                ["工具后端", trace.get("backend") or "-"],
                ["工具调用次数", f"{trace.get('tool_calls')}/{trace.get('max_tool_calls')}"],
                ["知识检索方式", (rag.get("retrieval") or trace.get("retrieval") or "-")],
            ],
        ),
        "",
    ]

    # 2
    lines += [f"## 2. {MARKDOWN_SECTIONS[1]}", "", report.get("data_quality_summary", ""), ""]
    checks = data_quality.get("checks") or {}
    if checks:
        lines += [
            _table(
                ["检查项", "结果", "说明"],
                [[name, "通过" if (item or {}).get("passed") else "未通过", (item or {}).get("detail", "")] for name, item in checks.items()],
            ),
            "",
        ]

    # 3
    lines += [f"## 3. {MARKDOWN_SECTIONS[2]}", ""]
    feature_rows = report.get("feature_comparison") or []
    lines += [
        _table(
            ["特征", "正常", "异常", "变化量", "相对变化", "说明"],
            [
                [r.get("feature"), _num(r.get("normal")), _num(r.get("abnormal")), _num(r.get("delta")), _pct(r.get("ratio")), r.get("note")]
                for r in feature_rows
            ],
        ),
        "",
    ]
    band_rows = report.get("band_energy_comparison") or []
    if band_rows:
        lines += [
            "频带能量（单位：幅值平方和）：",
            "",
            _table(
                ["频带", "正常", "异常", "倍数(异常/正常)"],
                [[b.get("band"), _num(b.get("normal"), 6), _num(b.get("abnormal"), 6), _times(b.get("ratio"))] for b in band_rows],
            ),
            "",
        ]

    # 4
    lines += [f"## 4. {MARKDOWN_SECTIONS[3]}", "", report.get("spectrum_notes", ""), ""]

    # 5
    lines += [f"## 5. {MARKDOWN_SECTIONS[4]}", ""]
    if candidates:
        lines += [
            _table(
                ["故障类型", "标题", "相似度", "alignment_score", "来源", "定位", "命中特征"],
                [
                    [
                        c.get("fault_type"),
                        c.get("title"),
                        _num(c.get("score"), 4),
                        _num(c.get("alignment_score"), 4),
                        c.get("source"),
                        c.get("locator"),
                        c.get("evidence"),
                    ]
                    for c in candidates
                ],
            ),
            "",
        ]
        clause_rows = [
            [
                c.get("fault_type"),
                row.get("clause_id"),
                row.get("claim"),
                _CRITERIA_STATE_LABEL.get(row.get("state"), row.get("state")),
                _actual(row.get("actual")),
                row.get("expect"),
                c.get("locator"),
            ]
            for c in report.get("knowledge_candidates") or []
            for row in c.get("criteria") or []
        ]
        if clause_rows:
            lines += [
                "判据条款核对（hit=成立、miss=与数据矛盾、unknown=指标缺失未核对）：",
                "",
                _table(["故障类型", "条款 ID", "判据", "核对结果", "实测值", "期望", "定位"], clause_rows),
                "",
            ]
    else:
        lines += ["未检索到满足阈值的知识条目（证据不足）。", ""]

    # 6
    lines += [f"## 6. {MARKDOWN_SECTIONS[5]}", "", report.get("conclusion", ""), ""]

    # 7
    lines += [f"## 7. {MARKDOWN_SECTIONS[6]}", "", _bullets(report.get("conclusion_evidence")), ""]

    # 8
    lines += [f"## 8. {MARKDOWN_SECTIONS[7]}", ""]
    sources = report.get("conclusion_sources") or []
    if sources:
        lines += [
            _table(
                ["来源", "URL", "定位"],
                [[s.get("source"), s.get("url"), s.get("locator")] for s in sources],
            ),
            "",
        ]
    else:
        lines += ["本次没有可追溯的知识来源。", ""]

    # 9
    confidence = report.get("confidence") or {}
    lines += [f"## 9. {MARKDOWN_SECTIONS[8]}", "", f"置信等级：{confidence.get('level', '-')}", "", confidence.get("reason", ""), ""]

    # 10
    lines += [f"## 10. {MARKDOWN_SECTIONS[9]}", "", _bullets(report.get("review_suggestions")), ""]

    # 11
    lines += [f"## 11. {MARKDOWN_SECTIONS[10]}", "", _bullets(report.get("prototype_limits")), ""]

    return "\n".join(lines).rstrip() + "\n"