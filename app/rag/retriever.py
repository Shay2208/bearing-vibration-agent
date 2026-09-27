"""检索查询组装与候选约束：把结构化证据拼成中文查询、做阈值过滤与判据核对。"""

from __future__ import annotations

from app.config import settings
from app.rag import criteria as criteria_engine
from app.rag.store import KnowledgeStore

_store: KnowledgeStore | None = None


def get_store() -> KnowledgeStore:
    global _store
    if _store is None:
        _store = KnowledgeStore()
    return _store


def _fmt(value) -> str:
    if value is None:
        return "?"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def build_query(device_info: dict, comparison: dict, features: dict | None = None) -> str:
    """把设备信息与对比证据拼成一段中文检索语句。"""
    device_info = device_info or {}
    comparison = comparison or {}
    parts: list[str] = []

    # 设备与工况
    device_type = device_info.get("device_type", "旋转设备")
    sensor = device_info.get("sensor_position", "未知测点")
    head = f"{sensor} 位置 {device_type} 振动信号"
    speed = device_info.get("rotation_speed")
    if speed is not None:
        head += f"，转速约 {_fmt(speed)} rpm"
        try:
            head += f"（转频约 {float(speed) / 60:.2f} Hz）"
        except (TypeError, ValueError):
            pass
    if device_info.get("sampling_rate") is not None:
        head += f"，采样率 {_fmt(device_info['sampling_rate'])} Hz"
    parts.append(head)

    # 主要特征变化
    feature_changes = comparison.get("feature_changes") or {}
    names = comparison.get("changed_features") or list(feature_changes.keys())
    changed = []
    for name in names[:6]:
        info = feature_changes.get(name) or {}
        normal, abnormal, ratio = info.get("normal"), info.get("abnormal"), info.get("ratio")
        if normal is None or abnormal is None:
            changed.append(str(name))
            continue
        seg = f"{name} 由 {_fmt(normal)} 变为 {_fmt(abnormal)}"
        if ratio is not None:
            seg += f"（约 {_fmt(ratio)} 倍）"
        changed.append(seg)
    if changed:
        parts.append("主要特征变化：" + "，".join(changed))

    # 主频迁移
    freq_change = comparison.get("frequency_change") or {}
    if freq_change:
        seg = (
            f"主频由 {_fmt(freq_change.get('normal_dominant'))} Hz "
            f"迁移至 {_fmt(freq_change.get('abnormal_dominant'))} Hz"
        )
        if freq_change.get("shift_hz") is not None:
            seg += f"，偏移 {_fmt(freq_change['shift_hz'])} Hz"
        parts.append(seg)

    # 谱峰与特征频率谐波对齐：同一特征频率族的 1/2/3 阶合并成一条，
    # 逐阶罗列会让单个族在查询文本里反复出现、把检索压向单一条目（实测会把外圈样本压向内圈）。
    matches = comparison.get("fault_frequency_matches") or []
    fault_freqs = comparison.get("fault_frequencies") or {}
    by_family: dict[str, list[dict]] = {}
    for match in matches:
        if match.get("label"):
            by_family.setdefault(str(match["label"]), []).append(match)
    aligned = []
    for label, items in by_family.items():
        orders = sorted(int(item.get("harmonic_order") or 1) for item in items)
        seg = f"{label} 族特征频率 {_fmt(fault_freqs.get(label.lower()))} Hz"
        if len(orders) > 1:
            seg += f"，{ '、'.join(str(order) for order in orders) } 阶谐波处均有谱峰"
        elif orders[0] > 1:
            seg += f"，第 {orders[0]} 阶谐波处有谱峰"
        deviations = [item.get("deviation_pct") for item in items if item.get("deviation_pct") is not None]
        if deviations:
            seg += f"（最大偏差 {_fmt(max(deviations))}%）"
        aligned.append(seg)
    if aligned:
        parts.append("谱峰与 " + "、".join(aligned) + " 对齐")

    # 包络谱特征频率能量（只出现 BPFO/BPFI/BSF/FTF 频率代号与数值，不引入故障中文名）
    energy_changes = comparison.get("characteristic_energy_changes") or {}
    if energy_changes:
        dominant = comparison.get("dominant_characteristic")
        tied = (dominant or {}).get("tied_candidates") or []
        if tied:
            # 并列时不写"最突出"：把两个族一起写进查询，让检索自行权衡
            parts.append(
                "包络谱中 "
                + " 与 ".join(
                    f"{item.get('label')}（{_fmt(item.get('frequency_hz'))} Hz）能量 {_fmt(item.get('abnormal_energy'))}"
                    for item in tied
                )
                + "，两者能量相当、不足以区分主次"
            )
        elif dominant:
            seg = (
                f"包络谱中 {dominant.get('label')}（{_fmt(dominant.get('frequency_hz'))} Hz）成分最突出，"
                f"能量 {_fmt(dominant.get('abnormal_energy'))}"
            )
            seg += (
                f"，为基线的 {_fmt(dominant.get('ratio'))} 倍"
                if dominant.get("detected_in_baseline")
                else "，基线未检出"
            )
            parts.append(seg)

        ranked = sorted(
            energy_changes.items(), key=lambda kv: (kv[1] or {}).get("abnormal") or 0, reverse=True
        )[:2]
        segs = []
        for name, block in ranked:
            block = block or {}
            seg = f"{str(name).upper()}（含 2、3 阶谐波）包络能量 {_fmt(block.get('abnormal'))}"
            seg += (
                f"，为基线的 {_fmt(block.get('ratio'))} 倍"
                if block.get("detected_in_baseline")
                else "，基线未检出"
            )
            segs.append(seg)
        if segs:
            parts.append("；".join(segs))

    # 转频与工况（只引用与证据相关的特征频率，避免无关故障标签干扰检索）
    if fault_freqs.get("shaft_frequency") is not None:
        parts.append(f"理论转频 {_fmt(fault_freqs['shaft_frequency'])} Hz")

    # 频带能量变化（取 ratio 最大的 2~3 个频带）
    band_changes = comparison.get("band_energy_changes") or {}
    ranked = sorted(
        band_changes.items(), key=lambda kv: (kv[1] or {}).get("ratio") or 0, reverse=True
    )[:3]
    bands = [f"{name} 能量约 {_fmt((info or {}).get('ratio'))} 倍" for name, info in ranked]
    if bands:
        parts.append("频带能量变化：" + "，".join(bands))

    # 证据摘要
    evidence = comparison.get("evidence") or []
    if evidence:
        parts.append("证据：" + "；".join(str(e) for e in evidence[:3]))

    # 附加特征
    if features:
        items = [
            f"{k}={_fmt(v)}"
            for k, v in features.items()
            if not isinstance(v, (dict, list))
        ]
        if items:
            parts.append("附加特征：" + "，".join(items))

    return "；".join(parts)


def retrieve_candidates(
    query: str,
    top_k: int | None = None,
    threshold: float | None = None,
    comparison: dict | None = None,
    features: dict | None = None,
) -> dict:
    """按阈值过滤候选，再用条款判据核对结果重排；过滤后为空则标记证据不足，绝不编造 fault_type。

    阈值过滤仍按检索分 `score`（候选集合不变），`alignment_score` 只决定候选之间的先后，
    因此纯文本检索（不传 comparison）与判据核对接入前后的候选集合完全一致。
    """
    store = get_store()
    top_k = top_k or settings.rag_top_k
    threshold = settings.rag_score_threshold if threshold is None else threshold

    results = store.search(query, top_k=top_k)
    candidates = [r for r in results if r["score"] >= threshold]
    candidates = criteria_engine.evaluate_candidates(candidates, comparison, features)

    sources: list[dict] = []
    seen = set()
    for item in candidates:
        key = (item["source"], item["source_url"], item["locator"])
        if key in seen:
            continue
        seen.add(key)
        sources.append({
            "name": item["source"],
            "url": item["source_url"],
            "locator": item["locator"],
            "fault_type": item["fault_type"],
        })

    return {
        "query": query,
        "retrieval": store.retrieval_mode,
        "candidates": candidates,
        "insufficient_evidence": not candidates,
        "sources": sources,
        "degraded": store.degraded,
        "degraded_reason": store.degraded_reason,
    }