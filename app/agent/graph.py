"""Agent 编排图（LangGraph）：把一次诊断表达成固定节点 + 条件分支的显式图。

流程图（节点名即下面的函数名，边固定、可追溯，不做开放式 Agent 循环）：

    START
      → validate_node          校验请求；非法输入直接结束，不调用任何工具
      → mcp_analysis_node      4 次 MCP 工具调用，只能经 app/mcp/adapter.py
      → rag_retrieval_node     调 app/rag/retriever 的既有入口取候选（不计入工具调用）
      → evidence_check_node    判断证据是否充分（正常 vs 正常 / 无候选 → 证据不足）
      → 证据不足 → report_node  跳过模型，模板报告输出「无法确认」
        证据充分 → llm_node     要求模型返回结构化 JSON，任何问题一律降级模板
                   → report_node
      → END

约束全部落在节点内部，与改造前逐条一致：
  - 工具调用上限 MAX_TOOL_CALLS，超限返回结构化错误并不产生任何结论；
  - 工具信封 ok=false 或数据校验不通过即停机；
  - 证据不足不下结论（不基于高分候选强行判定）；
  - selected_fault_type 只能取候选集合里的 fault_type 或 unconfirmed；
  - 模型超时 / 非 JSON / 候选外类型一律 template_fallback，且已算出的数据与候选全部保留。

状态字段见 DiagnosisState；外部唯一入口是 orchestrator.diagnose() → run_diagnosis()。
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import math
import operator
import time
from pathlib import Path
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph

from app.agent.events import DiagnosisEmitter, new_session_id
from app.agent.llm_client import LLMClient, LLMError, extract_json_object
from app.agent.report import build_report, render_markdown
from app.config import settings
from app.mcp import adapter
from app.rag import retriever

# ---------------------------------------------------------------------------
# 事件发射辅助（P1）：emitter 为 None 时零开销，任何发射失败都不影响诊断
# ---------------------------------------------------------------------------
async def _emit(state: dict, event: str, **fields) -> None:
    emitter = state.get("emitter")
    if emitter is None:
        return
    try:
        await emitter.emit(event, **fields)
    except Exception:
        return


SYSTEM_PROMPT = (
    "你是旋转机械振动诊断助手。你的判断只能建立在用户给出的「数据事实」与「知识库候选」之上，"
    "严禁引入候选列表之外的故障类型，严禁编造数据、来源或引用。"
    "候选后附的「条款级核对结果」由确定性规则引擎给出，只能引用、不能改写；"
    "若首位候选存在「判据未命中」，需在 reasoning 中说明该冲突，并优先选择条款与数据一致的候选。"
    "如果候选不足以支撑判断，selected_fault_type 必须填 unconfirmed。"
    "只输出一个 JSON 对象，不要输出代码块标记或任何额外说明，字段如下："
    '{"selected_fault_type": "<只能从给定的候选 fault_type 或 unconfirmed 中选择>", '
    '"summary": "<中文结论：最可能故障类型、判断依据与不确定性>", '
    '"reasoning": ["<中文：为何选择该候选，以及仍需复核的部分>"], '
    '"review_suggestions": ["<中文：建议复核的数据或现场检查项目>"]}'
)


# ---------------------------------------------------------------------------
# 状态定义
# ---------------------------------------------------------------------------
class DiagnosisState(TypedDict, total=False):
    """图状态。前 15 个键是对外的诊断结果字段，其余为图内部字段。"""

    request: dict
    data_quality: dict | None
    normal_features: dict | None
    abnormal_features: dict | None
    comparison: dict | None
    rag: dict | None
    candidates: list[dict]
    llm_result: dict | None
    report: dict | None
    report_markdown: str | None
    status: str
    error: dict | None
    trace: dict

    # 供最终响应组装使用的附加结果字段
    device: dict
    features: dict | None
    sources: list[dict]
    review_suggestions: list[str]

    # 图内部字段
    limit: int
    llm_client: Any
    insufficient: bool
    visited: Annotated[list[str], operator.add]
    # P1 事件发射器与 P2 会话 id：不进入最终响应，仅图内部使用
    emitter: Any
    session_id: str


# ---------------------------------------------------------------------------
# 输入与提示词组装
# ---------------------------------------------------------------------------
def _basename(path) -> str | None:
    if not path:
        return None
    return Path(str(path)).name


def _device_block(request: dict) -> dict:
    return {
        "device_type": str(request.get("device_type") or "滚动轴承（旋转机械）"),
        "sampling_rate": request.get("sampling_rate"),
        "rotation_speed": request.get("rotation_speed"),
        "sensor_position": str(request.get("sensor_position") or "未指定"),
        "normal_file": _basename(request.get("normal_path")),
        "abnormal_file": _basename(request.get("abnormal_path")),
    }


def _validate_request(request: dict) -> str | None:
    """返回错误说明；None 表示输入合法。"""
    for field in ("normal_path", "abnormal_path"):
        value = request.get(field)
        if not isinstance(value, str) or not value.strip():
            return f"缺少必填字段 {field}（需为非空字符串路径），收到：{value!r}"
    for field in ("sampling_rate", "rotation_speed"):
        value = request.get(field)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) <= 0
        ):
            return f"字段 {field} 必须是有限正数，收到：{value!r}"
    return None


def _project_candidate(item: dict) -> dict:
    return {
        "fault_type": item.get("fault_type"),
        "title": item.get("title"),
        "score": item.get("score"),
        "alignment_score": item.get("alignment_score"),
        "criteria_check": item.get("criteria_check"),
        "source": item.get("source"),
        "source_url": item.get("source_url"),
        "locator": item.get("locator"),
        "evidence": item.get("evidence"),
        "frequency_family": item.get("frequency_family"),
    }


def _clause_lines(candidate: dict) -> str | None:
    """把候选的条款核对结果压成提示词里的一行：命中与未命中的条款分开列出。

    模型只看到「哪条判据成立 / 哪条与数据矛盾 / 实测多少」，不参与条款判定本身。
    """
    items = ((candidate or {}).get("criteria_check") or {}).get("items") or []
    if not items:
        return None
    grouped: dict[str, list[str]] = {"hit": [], "miss": [], "unknown": []}
    for item in items:
        grouped.setdefault(item.get("state"), []).append(
            f"{item.get('id')}（{item.get('claim')}；期望 {item.get('expect')}，实测 {_fmt_num(item.get('actual'))}）"
        )
    labels = [("hit", "判据命中"), ("miss", "判据未命中"), ("unknown", "判据未核对")]
    return "；".join(
        f"{label}：" + "、".join(grouped[state]) for state, label in labels if grouped.get(state)
    )


def _build_user_prompt(device: dict, data_quality: dict | None, comparison: dict, candidates: list[dict]) -> str:
    lines = [
        f"设备与工况：{device.get('device_type')}，{device.get('sensor_position')} 测点，"
        f"转速 {device.get('rotation_speed')} rpm，采样率 {device.get('sampling_rate')} Hz。",
        "数据质量："
        + ("校验通过、两样本可比。" if (data_quality or {}).get("valid") and (data_quality or {}).get("comparable") else "校验未通过或不完全可比。"),
        "数据事实（来自振动分析工具，禁止改写数值）：",
    ]
    evidence_lines = [f"- {e}" for e in (comparison.get("evidence") or [])]
    lines += evidence_lines or ["- （无）"]
    lines.append("知识库候选（只能从下列 fault_type 中选择）：")
    for candidate in candidates:
        line = (
            f"- {candidate.get('fault_type')}（{candidate.get('title')}，相似度 {candidate.get('score')}）："
            f"{candidate.get('evidence')}〔来源：{candidate.get('source')} · {candidate.get('locator')}〕"
        )
        clause = _clause_lines(candidate)
        if clause:
            line += "\n  条款级核对结果（由确定性规则引擎给出，禁止改写）：" + clause
        lines.append(line)
    lines.append("允许的 selected_fault_type 取值：" + "、".join([c.get("fault_type") or "" for c in candidates] + ["unconfirmed"]))
    lines.append(
        "输出要求：只输出一个 JSON 对象，字段为 selected_fault_type、summary（字符串）、"
        "reasoning（字符串数组）、review_suggestions（字符串数组），不要输出代码块标记或额外说明。"
    )
    return "\n".join(lines)


def _no_significant_change(comparison: dict) -> bool:
    """异常样本相对基线是否完全没有变化（无显著特征变化、主频未迁移）。

    用于拦住「拿正常样本当异常样本」这类输入：没有任何异常可解释时，
    不应基于检索到的高分候选强行下结论。
    """
    if comparison.get("changed_features"):
        return False
    for block in (comparison.get("feature_changes") or {}).values():
        ratio = (block or {}).get("ratio")
        if ratio is not None and abs(float(ratio) - 1.0) >= 0.3:
            return False
    shift = (comparison.get("frequency_change") or {}).get("shift_hz")
    return abs(float(shift or 0.0)) < 1e-9


def _ambiguous_dominant(comparison: dict) -> str | None:
    """包络谱主导特征频率族并列时，不给排他性结论。

    最高/次高能量差落在噪声量级（低于量级余量阈值）时，"哪个族主导"本身就不可判定，
    此时下任何单一故障类型都是过度解读，按证据不足处理。
    """
    dominant = comparison.get("dominant_characteristic") or {}
    if not dominant.get("ambiguous"):
        return None
    names = "、".join(str(item.get("label")) for item in dominant.get("tied_candidates") or [])
    return (
        f"包络谱特征频率族能量并列（{names or dominant.get('label')}，"
        f"最高/次高能量比 {_fmt_num(dominant.get('margin_ratio'))} 低于量级余量阈值），"
        "无法确定主导特征频率族，按证据不足处理，不做故障类型判断。"
    )


def _tied_family_resolution(candidates: list[dict], comparison: dict) -> dict | None:
    """并列时用知识库候选打破并列：仅当检索首位候选声明的特征频率族落在并列族内才采信。

    并列说明「哪个族主导」不可判定，但知识库条目自带 frequency_family 元数据，
    若检索首位候选（即与查询语句最匹配的条目）恰好属于并列族之一，则该候选与并列
    现象自洽，可给出该候选（置信度由 report 下调为低）；族不匹配时仍按证据不足处理。
    """
    dominant = comparison.get("dominant_characteristic") or {}
    if not dominant.get("ambiguous") or not candidates:
        return None
    tied = {
        str(item.get("label") or "").strip().upper()
        for item in dominant.get("tied_candidates") or []
        if item.get("label")
    }
    top = candidates[0]
    family = str(top.get("frequency_family") or "").strip().upper()
    if not tied or not family or family not in tied:
        return None
    return {
        "fault_type": top.get("fault_type"),
        "frequency_family": family,
        "tied_labels": sorted(tied),
        "margin_ratio": dominant.get("margin_ratio"),
    }


# ---------------------------------------------------------------------------
# 模型步骤（结构化 JSON 输出；失败一律降级，不抛出）
# ---------------------------------------------------------------------------
def _run_llm(
    llm_client,
    device: dict,
    data_quality: dict | None,
    comparison: dict,
    candidates: list[dict],
    trace: dict,
) -> dict | None:
    """返回经过候选约束校验的 llm_result；任何问题都返回 None 并写入 trace。

    trace 的 llm_mode 只取三值：template（未配置 Key/禁用）、
    template_fallback（调用失败、非 JSON、候选外类型）、llm（成功）。
    """
    client = llm_client if llm_client is not None else LLMClient()
    if not bool(getattr(client, "available", False)):
        trace["llm_mode"] = "template"
        trace["warnings"].append("没有可用的模型客户端（未配置 API Key 或已被禁用），已使用模板化报告。")
        return None

    try:
        trace["llm_calls"] += 1
        text = client.chat(
            SYSTEM_PROMPT, _build_user_prompt(device, data_quality, comparison, candidates)
        )
    except LLMError as exc:
        trace["llm_mode"] = "template_fallback"
        trace["llm_error"] = {"code": exc.code, "message": exc.message}
        trace["warnings"].append(f"模型调用失败（{exc.code}），已降级为模板报告：{exc.message}")
        return None
    except Exception as exc:  # 兜底：假客户端或 SDK 抛出非 LLMError 异常
        trace["llm_mode"] = "template_fallback"
        trace["llm_error"] = {"code": "llm_error", "message": f"{type(exc).__name__}: {exc}"}
        trace["warnings"].append(f"模型调用出现异常（{type(exc).__name__}），已降级为模板报告：{exc}")
        return None

    try:
        payload = extract_json_object(text)
    except Exception as exc:
        trace["llm_mode"] = "template_fallback"
        trace["warnings"].append(f"模型输出不是合法 JSON（{exc}），已降级为模板报告。")
        return None

    selected = str(payload.get("selected_fault_type") or "").strip()
    allowed = {c.get("fault_type") for c in candidates} | {"unconfirmed"}
    if selected not in allowed:
        trace["llm_mode"] = "template_fallback"
        trace["warnings"].append(
            f"模型输出了候选外的故障类型（{selected or '空值'}），已降级为模板报告。"
        )
        return None

    summary = str(payload.get("summary") or payload.get("conclusion") or "").strip()
    reasoning = [str(item).strip() for item in (payload.get("reasoning") or []) if str(item).strip()]
    review = [str(item).strip() for item in (payload.get("review_suggestions") or []) if str(item).strip()]

    trace["llm_mode"] = "llm"
    return {
        "selected_fault_type": selected,
        "summary": summary or None,
        "reasoning": reasoning,
        "review_suggestions": review,
        # 兼容 report.build_report 读取的键（模型只改写措辞，不改变事实部分）
        "conclusion": summary or None,
        "reason": "；".join(reasoning) or None,
    }


# ---------------------------------------------------------------------------
# 工具调用（只经适配层，保持原有上限与失败语义；P1 事件 + P3 有限重试）
# ---------------------------------------------------------------------------
# P3 有限重试：只重试基础设施类错误（连接/超时/协议/内部异常）；
# 数据内容错误（missing_value 等确定性失败）重试没有意义，直接停机。
_RETRYABLE_TOOL_ERRORS = {
    "mcp_connection_error",
    "mcp_timeout",
    "mcp_protocol_error",
    "internal_error",
}


async def _call_tool(
    trace: dict,
    limit: int,
    step: str,
    tool_name: str,
    func,
    *args,
    emitter: DiagnosisEmitter | None = None,
    summarize=None,
    **kwargs,
):
    """调用适配层工具。返回 (信封, error)；超限或工具失败时 error 非空。

    P1：emitter 非空时发射 tool_started / tool_completed（失败带 payload.error）。
    P3：可重试错误按 settings.mcp_max_retries 有限重试，重试不占用工具调用配额。
    """
    if trace["tool_calls"] >= limit:
        return None, {
            "code": "max_tool_calls_exceeded",
            "message": (
                f"工具调用次数已达上限 {limit}，在「{step}」步骤前停止，未产生任何诊断结论。"
            ),
        }
    trace["tool_calls"] += 1

    async def fire(event: str, **fields):
        if emitter is not None:
            await emitter.emit(event, **fields)

    await fire(
        "tool_started",
        node="mcp_analysis_node", tool=tool_name, status="running",
        label=f"调用 {tool_name}（{step}）",
    )

    envelope = await func(*args, **kwargs)
    retries = 0
    while not envelope.get("ok"):
        detail = envelope.get("error") or {}
        code = str(detail.get("code") or "tool_error")
        if code not in _RETRYABLE_TOOL_ERRORS or retries >= max(int(settings.mcp_max_retries), 0):
            break
        retries += 1
        trace["tool_retries"][step] = trace["tool_retries"].get(step, 0) + 1
        trace["warnings"].append(
            f"工具 {tool_name} 在「{step}」步骤失败（{code}），进行第 {retries} 次重试：{detail.get('message')}"
        )
        await fire(
            "warning",
            node="mcp_analysis_node", tool=tool_name, status="retrying",
            label=f"{tool_name} 调用失败，正在重试（第 {retries} 次）",
            payload={"error": detail, "retry": retries},
        )
        envelope = await func(*args, **kwargs)

    if not envelope.get("ok"):
        detail = envelope.get("error") or {}
        await fire(
            "tool_completed",
            node="mcp_analysis_node", tool=tool_name, status="failed",
            label=f"{tool_name}（{step}）调用失败",
            summary=str(detail.get("message") or "")[:200],
            payload={"error": detail, "retries": retries},
        )
        return envelope, {
            "code": str(detail.get("code") or "tool_error"),
            "message": (
                f"在「{step}」步骤调用工具 {envelope.get('tool')} 失败（{detail.get('code')}）："
                f"{detail.get('message')}"
            ),
        }

    summary = None
    if summarize is not None:
        try:
            summary = summarize(envelope.get("data"))
        except Exception:
            summary = None
    await fire(
        "tool_completed",
        node="mcp_analysis_node", tool=tool_name, status="completed",
        label=f"{tool_name}（{step}）完成",
        summary=summary,
        payload={"retries": retries} if retries else None,
    )
    return envelope, None


# ---------------------------------------------------------------------------
# 节点
# ---------------------------------------------------------------------------
async def validate_node(state: DiagnosisState) -> dict:
    """步骤 1：校验输入。非法输入直接结束，不调用任何工具。"""
    request = state.get("request") or {}
    device = _device_block(request)
    await _emit(
        state, "node_started",
        node="validate_node", status="running", label="开始校验输入数据",
    )
    invalid = _validate_request(request)
    if invalid:
        error = {"code": "invalid_argument", "message": f"输入校验未通过，流程未开始：{invalid}"}
        await _emit(
            state, "node_completed",
            node="validate_node", status="failed", label="输入校验",
            summary=error["message"], payload={"error": error},
        )
        return {
            "visited": ["validate_node"],
            "device": device,
            "status": "error",
            "error": error,
        }
    summary = (
        f"输入校验通过：转速 {device.get('rotation_speed')} rpm、采样率 {device.get('sampling_rate')} Hz、"
        f"测点 {device.get('sensor_position')}，基线 {device.get('normal_file')} / 异常 {device.get('abnormal_file')}"
    )
    await _emit(
        state, "node_completed",
        node="validate_node", status="completed", label="输入校验", summary=summary,
    )
    return {"visited": ["validate_node"], "device": device, "status": "ok", "error": None}


def _summarize_quality(data: dict | None) -> str | None:
    data = data or {}
    normal = data.get("normal") or {}
    abnormal = data.get("abnormal") or {}
    return (
        f"两样本均通过校验（正常 {normal.get('n_samples')} 点 / 异常 {abnormal.get('n_samples')} 点，"
        f"采样率 {normal.get('sampling_rate')} Hz）"
    )


def _summarize_features(data: dict | None) -> str | None:
    data = data or {}

    def fmt(value) -> str:
        try:
            return f"{float(value):.4g}"
        except (TypeError, ValueError):
            return str(value)

    return (
        f"RMS={fmt(data.get('rms'))}，主频 {fmt(data.get('dominant_frequency'))} Hz，"
        f"峭度 {fmt(data.get('kurtosis'))}"
    )


def _summarize_comparison(data: dict | None) -> str | None:
    data = data or {}
    changed = len(data.get("changed_features") or [])
    dominant = data.get("dominant_characteristic") or {}

    def fmt(value) -> str:
        try:
            return f"{float(value):.4g}"
        except (TypeError, ValueError):
            return str(value)

    if dominant:
        return (
            f"显著变化特征 {changed} 项，包络谱 {dominant.get('label')}（{fmt(dominant.get('frequency_hz'))} Hz）"
            f"能量为基线的 {fmt(dominant.get('ratio'))} 倍"
        )
    return f"显著变化特征 {changed} 项"


def _persist_audio(features: dict | None, session_id: str, slot: str) -> bool:
    """把特征里的 WAV（base64）落盘成旁挂文件，并从响应结构中移除原始字节。

    请求的临时目录结束后即删、会话记录也不含任何路径，所以音频必须在这里按 session_id
    落盘；前端用 /api/sessions/{id}/audio/{slot}.wav 懒加载，不把 PCM 塞进响应或数据库。
    """
    series = (features or {}).get("series")
    audio = series.get("audio") if isinstance(series, dict) else None
    if not isinstance(audio, dict):
        return True
    encoded = audio.pop("wav_base64", None)
    audio["available"] = False
    if not encoded:
        return True
    try:
        settings.audio_dir.mkdir(parents=True, exist_ok=True)
        (settings.audio_dir / f"{session_id}_{slot}.wav").write_bytes(base64.b64decode(encoded))
    except (OSError, ValueError, binascii.Error):
        return False
    audio["available"] = True
    _prune_audio(settings.audio_dir, settings.audio_max_files)
    return True


def _prune_audio(directory: Path, keep: int) -> None:
    """按修改时间保留最新的 keep 个旁挂音频，避免长期运行把磁盘堆满。"""
    if keep <= 0:
        return
    try:
        files = sorted(directory.glob("*.wav"), key=lambda item: item.stat().st_mtime, reverse=True)
        for stale in files[keep:]:
            stale.unlink()
    except OSError:
        return


async def mcp_analysis_node(state: DiagnosisState) -> dict:
    """步骤 2~4：4 次 MCP 工具调用（数据校验 → 正常特征 → 异常特征 → 对比）。"""
    trace = state["trace"]
    limit = int(state.get("limit") or settings.max_tool_calls)
    emitter = state.get("emitter")
    request = state.get("request") or {}
    normal_path = request["normal_path"]
    abnormal_path = request["abnormal_path"]
    sampling_rate = request["sampling_rate"]
    rotation_speed = request["rotation_speed"]
    sensor_position = request.get("sensor_position")

    await _emit(
        state, "node_started",
        node="mcp_analysis_node", status="running", label="开始调用振动分析工具（4 次 MCP 调用）",
    )

    def stop_with_error(error: dict, **extra) -> dict:
        return {"visited": ["mcp_analysis_node"], "status": "error", "error": error, **extra}

    async def emit_failed(summary: str) -> None:
        await _emit(
            state, "node_completed",
            node="mcp_analysis_node", status="failed", label="振动信号分析", summary=summary,
        )

    envelope, error = await _call_tool(
        trace, limit, "数据校验", "validate_vibration_data",
        adapter.validate_vibration_data, normal_path, abnormal_path, sampling_rate, sensor_position,
        emitter=emitter, summarize=_summarize_quality,
    )
    if error:
        await emit_failed(error["message"])
        return stop_with_error(error)
    data_quality = envelope["data"]
    if not data_quality.get("valid") or not data_quality.get("comparable"):
        errors = data_quality.get("errors") or []
        first = errors[0] if errors else {}
        reasons = "；".join(str(e.get("message")) for e in errors[:3]) or "数据校验未通过"
        error = {
            "code": str(first.get("code") or "invalid_data"),
            "message": (
                f"数据校验未通过（valid={data_quality.get('valid')}，comparable={data_quality.get('comparable')}），"
                f"流程在特征提取前停止：{reasons}"
            ),
        }
        await emit_failed(error["message"])
        return stop_with_error(error, data_quality=data_quality)
    out: dict = {"visited": ["mcp_analysis_node"], "data_quality": data_quality}

    envelope, error = await _call_tool(
        trace, limit, "正常样本特征提取", "extract_vibration_features",
        adapter.extract_vibration_features, normal_path, sampling_rate, rotation_speed=rotation_speed,
        include_series=True, emitter=emitter, summarize=_summarize_features,
    )
    if error:
        await emit_failed(error["message"])
        out.update({"status": "error", "error": error})
        return out
    normal_features = envelope["data"]
    if not _persist_audio(normal_features, state.get("session_id"), "normal"):
        trace["warnings"].append("正常样本波形音频落盘失败，本次不提供试听。")

    envelope, error = await _call_tool(
        trace, limit, "异常样本特征提取", "extract_vibration_features",
        adapter.extract_vibration_features, abnormal_path, sampling_rate, rotation_speed=rotation_speed,
        include_series=True, emitter=emitter, summarize=_summarize_features,
    )
    if error:
        await emit_failed(error["message"])
        out.update({"normal_features": normal_features, "features": {"normal": normal_features, "abnormal": None},
                    "status": "error", "error": error})
        return out
    abnormal_features = envelope["data"]
    if not _persist_audio(abnormal_features, state.get("session_id"), "abnormal"):
        trace["warnings"].append("异常样本波形音频落盘失败，本次不提供试听。")

    envelope, error = await _call_tool(
        trace, limit, "特征对比", "compare_normal_abnormal",
        adapter.compare_normal_abnormal, normal_features, abnormal_features, rotation_speed,
        emitter=emitter, summarize=_summarize_comparison,
    )
    if error:
        await emit_failed(error["message"])
        out.update({
            "normal_features": normal_features,
            "abnormal_features": abnormal_features,
            "features": {"normal": normal_features, "abnormal": abnormal_features},
            "status": "error",
            "error": error,
        })
        return out

    comparison = envelope["data"]
    await _emit(
        state, "node_completed",
        node="mcp_analysis_node", status="completed", label="振动信号分析",
        summary=_summarize_comparison(comparison),
    )
    out.update({
        "normal_features": normal_features,
        "abnormal_features": abnormal_features,
        "features": {"normal": normal_features, "abnormal": abnormal_features},
        "comparison": comparison,
        "status": "ok",
    })
    return out


async def rag_retrieval_node(state: DiagnosisState) -> dict:
    """步骤 5：RAG 检索（非 MCP 工具，不计入工具调用）。

    检索分负责「找条款」、阈值过滤候选集合；条款核对结果负责「验条款」——
    用 alignment_score（检索分 × 判据因子）决定候选先后，让判据与数据矛盾的条目自然下沉。
    """
    trace = state["trace"]
    await _emit(
        state, "node_started",
        node="rag_retrieval_node", status="running", label="开始检索故障知识（RAG）",
    )
    comparison = state.get("comparison") or {}
    abnormal_features = state.get("abnormal_features")
    query = retriever.build_query(state.get("device") or {}, comparison, abnormal_features)
    rag = retriever.retrieve_candidates(query, comparison=comparison, features=abnormal_features)
    trace["retrieval"] = rag.get("retrieval")

    # 检索层降级契约：degraded / degraded_reason 用 .get() 容错，缺失按未降级处理
    if rag.get("degraded"):
        reason = rag.get("degraded_reason")
        trace["warnings"].append(f"知识检索发生降级：{reason}" if reason else "知识检索发生降级（未提供降级原因）。")
        await _emit(
            state, "warning",
            node="rag_retrieval_node", status="degraded",
            label="知识检索发生降级",
            summary=reason or "知识检索降级（未提供原因）。",
        )

    candidates = [_project_candidate(item) for item in rag.get("candidates") or []]
    trace["criteria_summary"] = [
        {
            "fault_type": item.get("fault_type"),
            "state": (item.get("criteria_check") or {}).get("state"),
            "hits": (item.get("criteria_check") or {}).get("hits"),
            "misses": (item.get("criteria_check") or {}).get("misses"),
            "unknowns": (item.get("criteria_check") or {}).get("unknowns"),
            "checked": (item.get("criteria_check") or {}).get("checked"),
            "factor": (item.get("criteria_check") or {}).get("factor"),
            "score": item.get("score"),
            "alignment_score": item.get("alignment_score"),
        }
        for item in candidates
    ]
    top = candidates[0] if candidates else {}
    clause_state = (top.get("criteria_check") or {}).get("state")
    summary = (
        f"检索到 {len(candidates)} 条候选（{rag.get('retrieval')} 模式），"
        f"最匹配：{top.get('fault_type')}（检索分 {top.get('score')}，判据核对 {clause_state}）"
        if candidates else
        f"未检索到任何过阈值候选（{rag.get('retrieval')} 模式，证据不足）"
    )
    await _emit(
        state, "node_completed",
        node="rag_retrieval_node", status="completed", label="故障知识检索", summary=summary,
    )
    return {
        "visited": ["rag_retrieval_node"],
        "rag": rag,
        "candidates": candidates,
        "sources": list(rag.get("sources") or []),
    }


# ---------------------------------------------------------------------------
# P3 自适应分支的三个守卫（证据驱动，不改变候选白名单与结论约束）
# ---------------------------------------------------------------------------
def _fmt_num(value) -> str:
    try:
        return f"{float(value):.4g}"
    except (TypeError, ValueError):
        return str(value)


def _short_signal_reason(features: dict | None) -> str | None:
    """信号时长不足：停止强行诊断，返回补充数据建议。"""
    abnormal = (features or {}).get("abnormal") or {}
    duration = abnormal.get("duration_seconds")
    if duration is None:
        return None
    try:
        duration = float(duration)
    except (TypeError, ValueError):
        return None
    if duration < settings.min_signal_duration:
        return (
            f"异常信号时长仅 {duration:.3f} 秒（低于 {settings.min_signal_duration} 秒），"
            "频率分辨率不足以支撑包络谱特征频率核对，停止强行诊断；"
            "建议按当前采样率继续采集至 1 秒以上的连续片段后重新提交。"
        )
    return None


def _candidate_gap_review(candidates: list[dict], comparison: dict) -> dict | None:
    """前两名候选分数接近时，用频率证据复核并给出人工复核建议。

    返回复核摘要（写入 trace.frequency_review，并附 review 建议）；不接近则 None。
    """
    if len(candidates) < 2:
        return None
    top, second = candidates[0], candidates[1]
    try:
        gap = float(top.get("score") or 0.0) - float(second.get("score") or 0.0)
    except (TypeError, ValueError):
        return None
    if gap >= settings.candidate_gap_threshold:
        return None

    dominant = comparison.get("dominant_characteristic") or {}
    matches = comparison.get("fault_frequency_matches") or []
    aligned = [
        f"{m.get('label')}（{_fmt_num(m.get('frequency'))} Hz，偏差 {m.get('deviation_pct')}%）"
        for m in matches[:3]
        if m.get("label")
    ]
    return {
        "reason": (
            f"前两名候选分数接近（{top.get('fault_type')} {_fmt_num(top.get('score'))} 与 "
            f"{second.get('fault_type')} {_fmt_num(second.get('score'))}，差 {gap:.4f} < {settings.candidate_gap_threshold}），"
            "单凭检索分数不足以区分，已追加频率证据复核。"
        ),
        "dominant_characteristic": (
            f"包络谱最突出成分 {dominant.get('label')}（{_fmt_num(dominant.get('frequency_hz'))} Hz）"
            if dominant else "包络谱未见突出的特征频率成分"
        ),
        "aligned_peaks": aligned,
        "review_advice": (
            "建议人工核对 2×BSF 与 BPFI 族谐波能量与转频边带，"
            "并结合第二候选的特征频率判据复核后再下结论。"
        ),
    }


def _criteria_conflict(candidates: list[dict]) -> dict | None:
    """首位候选的判据条款与数据事实矛盾（misses ≥ hits 或全不成立）时的记账。

    规则层能判的直接判，判到「自相矛盾」就不替模型做选择：
    该候选虽仍留在候选集合内（阈值过滤只看检索分），但置信度下调、
    trace 记录冲突条款，并把人工确认建议交给报告层。
    """
    top = candidates[0] if candidates else {}
    check = (top or {}).get("criteria_check") or {}
    if check.get("state") != "conflict":
        return None
    misses = [item for item in check.get("items") or [] if item.get("state") == "miss"]
    if not misses:
        return None
    detail = "；".join(
        f"{item.get('id')}（期望 {item.get('expect')}，实测 {_fmt_num(item.get('actual'))}）"
        for item in misses
    )
    return {
        "fault_type": top.get("fault_type"),
        "missed_clauses": [item.get("id") for item in misses],
        "reason": (
            f"首位候选 {top.get('fault_type')} 的判据条款与本次数据事实冲突（{detail}），"
            "规则层不采信该候选，已交模型在候选范围内裁量，并下调置信度、追加人工确认建议。"
        ),
    }


def _evidence_conflict_note(comparison: dict) -> str | None:
    """时域与频域证据冲突（一边显著一边无变化）时提示人工复核。"""
    feature_changes = comparison.get("feature_changes") or {}
    changed = comparison.get("changed_features") or []

    def ratio_of(name: str):
        try:
            return float((feature_changes.get(name) or {}).get("ratio"))
        except (TypeError, ValueError):
            return None

    rms_ratio = ratio_of("rms")
    kurtosis_ratio = ratio_of("kurtosis")
    time_domain_changed = bool(changed) or (
        rms_ratio is not None and abs(rms_ratio - 1.0) >= 0.3
    ) or (kurtosis_ratio is not None and abs(kurtosis_ratio - 1.0) >= 0.3)

    shift = (comparison.get("frequency_change") or {}).get("shift_hz")
    try:
        shift = abs(float(shift))
    except (TypeError, ValueError):
        shift = None
    dominant = comparison.get("dominant_characteristic") or {}
    try:
        dominant_ratio = float(dominant.get("ratio"))
    except (TypeError, ValueError):
        dominant_ratio = None
    frequency_domain_changed = (shift is not None and shift >= 2.0) or (
        dominant_ratio is not None and dominant_ratio >= 2.0
    )

    if time_domain_changed and not frequency_domain_changed:
        return (
            "时域特征（RMS/峭度）发生明显变化，但主频未迁移、包络特征频率能量未见对应增长，"
            "时/频证据存在冲突：可能是冲击噪声或工况波动，建议人工复核频谱与包络谱后再判断。"
        )
    if frequency_domain_changed and not time_domain_changed:
        return (
            "频域证据（主频迁移/包络特征频率能量）发生变化，但时域特征（RMS/峭度）几乎无变化，"
            "时/频证据存在冲突：早期故障或调制现象可能出现这种情况，建议人工复核并考虑补充解调分析。"
        )
    return None


async def evidence_check_node(state: DiagnosisState) -> dict:
    """步骤 6：判断证据是否充分，并顺手清掉证据不足时的候选/来源残留。"""
    trace = state["trace"]
    rag = state.get("rag") or {}
    candidates = list(state.get("candidates") or [])
    comparison = state.get("comparison") or {}
    features = state.get("features")
    no_change = _no_significant_change(comparison)

    await _emit(
        state, "node_started",
        node="evidence_check_node", status="running", label="开始判断证据充分性",
    )

    # P3-3：数据不足守卫（时长不够 → 停止强行诊断，给补充数据建议）
    short_signal = None if no_change else _short_signal_reason(features)
    # 特征频率族并列时，若检索首位候选声明的族落在并列族内，用知识库候选打破并列（置信度下调为低）
    resolution = None if (no_change or short_signal) else _tied_family_resolution(candidates, comparison)
    ambiguous = None if (no_change or resolution) else _ambiguous_dominant(comparison)
    if resolution:
        trace["tied_resolution"] = resolution
        trace["warnings"].append(
            f"包络谱特征频率族并列（{'、'.join(resolution['tied_labels'])}，"
            f"最高/次高能量比 {_fmt_num(resolution['margin_ratio'])} 低于量级余量阈值），"
            f"但检索首位候选 {resolution['fault_type']} 声明的特征频率族 "
            f"{resolution['frequency_family']} 落在并列族内，据此采信该候选并下调置信度；"
            "并列本身仍提示需人工复核，不代表主导族已确定。"
        )
    if ambiguous:
        trace["warnings"].append(ambiguous)
    if short_signal:
        trace["warnings"].append(short_signal)

    if no_change:
        trace["warnings"].append(
            "异常样本相对基线未观察到任何显著变化（无显著变化的特征、主频未迁移），"
            "不存在可解释的异常，按证据不足处理，不做故障类型判断。"
        )

    insufficient = (
        no_change
        or bool(ambiguous)
        or bool(short_signal)
        or bool(rag.get("insufficient_evidence"))
        or not candidates
    )
    if insufficient:
        trace["llm_mode"] = "template"
        if not no_change and not ambiguous and not short_signal:
            trace["warnings"].append(
                "知识检索未命中任何候选（证据不足），已跳过模型诊断，改用模板报告输出「无法确认」。"
            )
        reason = (
            "异常样本相对基线无显著变化，不存在可解释的异常" if no_change else
            short_signal or ambiguous or "知识检索未命中任何候选"
        )
        # 证据不足即不下结论：候选、来源、rag 视图一并清空，避免 sources 残留候选来源
        clean_rag = (
            {**rag, "candidates": [], "sources": [], "insufficient_evidence": True} if no_change else rag
        )
        await _emit(
            state, "node_completed",
            node="evidence_check_node", status="completed", label="证据充分性判断",
            summary=f"证据不足：{reason}，跳过模型诊断，模板报告输出「无法确认」。",
        )
        await _emit(
            state, "branch_selected",
            node="evidence_check_node", status="skipped",
            label="证据不足 → 跳过模型诊断",
            summary=reason,
            payload={"branch": "report_node", "reason": reason},
        )
        return {
            "visited": ["evidence_check_node"],
            "insufficient": True,
            "status": "insufficient_evidence",
            "rag": clean_rag,
            "candidates": [],
            "sources": [],
        }

    # P3-1：候选分数接近 → 频率证据复核
    review = _candidate_gap_review(candidates, comparison)
    if review:
        trace["frequency_review"] = review
        trace["warnings"].append(review["reason"] + " " + review["review_advice"])
        await _emit(
            state, "warning",
            node="evidence_check_node", status="review",
            label="候选分数接近，已追加频率证据复核",
            summary=review["reason"],
            payload={"frequency_review": review},
        )

    # P3-2：时域 / 频域证据冲突 → 提示人工复核
    conflict = _evidence_conflict_note(comparison)
    if conflict:
        trace["evidence_conflict"] = conflict
        trace["warnings"].append(conflict)
        await _emit(
            state, "warning",
            node="evidence_check_node", status="review",
            label="时域与频域证据存在冲突，建议人工复核",
            summary=conflict,
        )

    # 条款冲突：首位候选的判据条款与数据事实矛盾 → 交模型裁量 + 置信下调 + 人工确认
    clause_conflict = _criteria_conflict(candidates)
    if clause_conflict:
        trace["criteria_conflict"] = clause_conflict
        trace["warnings"].append(clause_conflict["reason"])
        await _emit(
            state, "warning",
            node="evidence_check_node", status="review",
            label="首位候选的判据条款与数据冲突，已交模型裁量",
            summary=clause_conflict["reason"],
            payload={"criteria_conflict": clause_conflict},
        )

    await _emit(
        state, "node_completed",
        node="evidence_check_node", status="completed", label="证据充分性判断",
        summary=(
            f"证据充分：存在显著特征变化，且检索到 {len(candidates)} 条候选"
            + ("（特征频率族并列，已由检索首位候选打破并列、置信度下调）" if resolution else "")
            + ("（已触发频率复核/人工复核提示）" if (review or conflict or clause_conflict) else "")
        ),
    )
    await _emit(
        state, "branch_selected",
        node="evidence_check_node", status="selected",
        label="证据充分 → 调用模型生成诊断",
        summary=f"检索到 {len(candidates)} 条候选，最高分 {candidates[0].get('fault_type')}（{candidates[0].get('score')}）",
        payload={"branch": "llm_node", "reason": "存在显著特征变化且检索到过阈值候选"},
    )
    return {"visited": ["evidence_check_node"], "insufficient": False, "status": "ok"}


async def llm_node(state: DiagnosisState) -> dict:
    """证据充分时的模型步骤：要求结构化 JSON，失败降级但保留候选与证据。"""
    trace = state["trace"]
    await _emit(
        state, "node_started",
        node="llm_node", status="running", label="开始调用模型生成诊断（结构化 JSON）",
    )
    # The model SDK is synchronous; keep its network wait off the ASGI event loop.
    llm_result = await asyncio.to_thread(
        _run_llm,
        state.get("llm_client"),
        state.get("device") or {},
        state.get("data_quality"),
        state.get("comparison") or {},
        state.get("candidates") or [],
        trace,
    )
    mode = trace.get("llm_mode")
    if mode == "template_fallback":
        await _emit(
            state, "warning",
            node="llm_node", status="fallback",
            label="模型调用失败，降级为确定性模板报告",
            summary=trace.get("warnings")[-1] if trace.get("warnings") else None,
            payload={"llm_error": trace.get("llm_error")},
        )
    elif mode == "template":
        await _emit(
            state, "warning",
            node="llm_node", status="fallback",
            label="没有可用的模型客户端，使用确定性模板报告",
            summary="未配置模型 API Key 或模型被禁用，报告由确定性模板生成。",
        )
    await _emit(
        state, "node_completed",
        node="llm_node", status="completed",
        label="模型诊断" if mode == "llm" else "模型诊断（模板回退）",
        summary=(
            f"模型判定：{(llm_result or {}).get('selected_fault_type')}" if llm_result
            else f"模型未产出结果（llm_mode={mode}），由模板报告兜底"
        ),
    )
    return {"visited": ["llm_node"], "llm_result": llm_result}


async def report_node(state: DiagnosisState) -> dict:
    """步骤 7：确定性组装报告并渲染 Markdown。"""
    trace = state["trace"]
    device = state.get("device") or {}
    data_quality = state.get("data_quality")
    comparison = state.get("comparison") or {}
    rag = state.get("rag")
    candidates = state.get("candidates") or []
    llm_result = state.get("llm_result")

    await _emit(
        state, "node_started",
        node="report_node", status="running", label="开始组装可追溯诊断报告",
    )
    report = build_report(
        device=device,
        data_quality=data_quality,
        comparison=comparison,
        rag=rag,
        candidates=candidates,
        llm_result=llm_result,
    )
    markdown = render_markdown(
        report,
        {
            "device": device,
            "data_quality": data_quality,
            "comparison": comparison,
            "rag": rag,
            "candidates": candidates,
            "trace": trace,
        },
    )
    confidence = report.get("confidence") or {}
    await _emit(
        state, "node_completed",
        node="report_node", status="completed", label="报告生成",
        summary=(
            f"报告已生成：结论「{report.get('conclusion')}」，置信度{confidence.get('level')}"
        ),
    )
    return {
        "visited": ["report_node"],
        "report": report,
        "report_markdown": markdown,
        "review_suggestions": list(report.get("review_suggestions") or []),
    }


# ---------------------------------------------------------------------------
# 条件分支
# ---------------------------------------------------------------------------
def _route_after_validate(state: DiagnosisState) -> str:
    return END if state.get("status") == "error" else "mcp_analysis_node"


def _route_after_analysis(state: DiagnosisState) -> str:
    return END if state.get("status") == "error" else "rag_retrieval_node"


def _route_after_evidence_check(state: DiagnosisState) -> str:
    return "report_node" if state.get("insufficient") else "llm_node"


# ---------------------------------------------------------------------------
# 组图
# ---------------------------------------------------------------------------
def build_graph():
    """构建并编译诊断图。节点/边的顺序即真实执行顺序。"""
    builder = StateGraph(DiagnosisState)
    builder.add_node("validate_node", validate_node)
    builder.add_node("mcp_analysis_node", mcp_analysis_node)
    builder.add_node("rag_retrieval_node", rag_retrieval_node)
    builder.add_node("evidence_check_node", evidence_check_node)
    builder.add_node("llm_node", llm_node)
    builder.add_node("report_node", report_node)

    builder.add_edge(START, "validate_node")
    builder.add_conditional_edges(
        "validate_node", _route_after_validate, {"mcp_analysis_node": "mcp_analysis_node", END: END}
    )
    builder.add_conditional_edges(
        "mcp_analysis_node", _route_after_analysis, {"rag_retrieval_node": "rag_retrieval_node", END: END}
    )
    builder.add_edge("rag_retrieval_node", "evidence_check_node")
    builder.add_conditional_edges(
        "evidence_check_node", _route_after_evidence_check, {"llm_node": "llm_node", "report_node": "report_node"}
    )
    builder.add_edge("llm_node", "report_node")
    builder.add_edge("report_node", END)
    return builder.compile()


GRAPH = build_graph()


# ---------------------------------------------------------------------------
# 图执行 + 响应组装（orchestrator.diagnose() 的唯一实现）
# ---------------------------------------------------------------------------
def _initial_trace(limit: int) -> dict:
    return {
        "backend": adapter.describe_backend(),
        "tool_calls": 0,
        "max_tool_calls": limit,
        "tool_retries": {},
        "llm_mode": "template",
        "llm_calls": 0,
        "llm_error": None,
        "elapsed_ms": 0,
        "retrieval": None,
        "criteria_summary": [],
        "criteria_conflict": None,
        "warnings": [],
        "nodes": [],
    }


def _final_response(final: dict, request: dict, trace: dict, session_id: str) -> dict:
    """把图最终状态组装成原有响应结构（键名与改造前一致，另加 session_id）。"""
    return {
        "session_id": session_id,
        "status": final.get("status") or "ok",
        "error": final.get("error"),
        "device": final.get("device") or _device_block(request),
        "data_quality": final.get("data_quality"),
        "features": final.get("features"),
        "comparison": final.get("comparison"),
        "rag": final.get("rag"),
        "candidates": final.get("candidates") or [],
        "report": final.get("report"),
        "report_markdown": final.get("report_markdown"),
        "sources": final.get("sources") or [],
        "review_suggestions": final.get("review_suggestions") or [],
        "trace": trace,
    }


async def run_diagnosis(
    request: dict,
    *,
    max_tool_calls: int | None = None,
    llm_client=None,
    emitter: DiagnosisEmitter | None = None,
) -> dict:
    """在图上前向执行一次诊断，并把图状态组装成原有响应结构（键名与改造前一致）。

    P1：emitter 非空时向其发射真实节点/工具事件（SSE 与会话记录共用）；
    emitter 为 None 时同步旧路径零开销。响应新增 session_id 字段（旧接口兼容）。
    """
    started = time.perf_counter()
    request = request if isinstance(request, dict) else {}
    limit = settings.max_tool_calls if max_tool_calls is None else int(max_tool_calls)
    trace = _initial_trace(limit)
    session_id = emitter.session_id if emitter is not None else new_session_id()

    device_preview = _device_block(request)
    if emitter is not None:
        await emitter.emit(
            "diagnosis_started",
            status="running",
            label="开始诊断",
            summary=(
                f"{device_preview.get('device_type')}，{device_preview.get('sensor_position')} 测点，"
                f"转速 {device_preview.get('rotation_speed')} rpm，采样率 {device_preview.get('sampling_rate')} Hz"
            ),
        )

    try:
        final = await GRAPH.ainvoke(
            {
                "request": request,
                "trace": trace,
                "limit": limit,
                "llm_client": llm_client,
                "visited": [],
                "emitter": emitter,
                "session_id": session_id,
            }
        )
    except Exception as exc:  # 图外异常兜底：发失败事件并返回结构化错误，绝不裸抛
        error = {"code": "internal_error", "message": f"诊断流程出现未预期异常（{type(exc).__name__}），已安全停止。"}
        trace["warnings"].append(error["message"])
        trace["elapsed_ms"] = int((time.perf_counter() - started) * 1000)
        if emitter is not None:
            await emitter.emit(
                "diagnosis_failed", status="failed", label="诊断失败",
                summary=error["message"], payload={"error": error},
            )
        return _final_response({"status": "error", "error": error}, request, trace, session_id)

    trace["nodes"] = list(final.get("visited") or [])
    trace["elapsed_ms"] = int((time.perf_counter() - started) * 1000)
    response = _final_response(final, request, trace, session_id)

    if emitter is not None:
        status = response.get("status")
        if status == "error":
            await emitter.emit(
                "diagnosis_failed", status="failed", label="诊断失败",
                summary=str((response.get("error") or {}).get("message") or "")[:200],
                payload={"error": response.get("error"), "result": response},
            )
        else:
            report = response.get("report") or {}
            confidence = report.get("confidence") or {}
            await emitter.emit(
                "diagnosis_completed", status="completed", label="诊断完成",
                summary=(
                    f"结论「{report.get('conclusion')}」，置信度{confidence.get('level')}，"
                    f"工具调用 {trace.get('tool_calls')} 次，耗时 {trace.get('elapsed_ms')} ms"
                ),
                payload={"result": response},
            )
    return response
