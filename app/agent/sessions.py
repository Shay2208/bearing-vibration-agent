"""诊断会话存储、受控追问（P2）与会话内自由问答（P4）。

会话保存的是脱敏的结构化上下文（不含原始文件内容）：
  - request_summary：输入摘要（样例/文件名、转速、采样率、测点）
  - result：run_diagnosis 的完整响应（MCP 工具结果、RAG 候选与来源、节点轨迹、最终报告）
  - events：本次诊断的真实事件列表（节点/工具/分支/回退）

追问是受控的：
  - 只允许固定 7 个问题（ALLOWED_QUESTIONS），越权问题返回 question_not_allowed；
  - 回答完全由会话上下文确定性组装（不调模型、不引入外部事实），
    上下文没有依据时明确回答「当前诊断记录无法支持该结论」；
  - 每个回答都带 facts_used，声明用到了哪些上下文字段，可追溯。

自由问答（answer_free_question）是「证据约束 + 模型表达」：
  - 把会话事实拼成上下文摘要，只有该摘要会进入模型提示词；
  - 最近几轮对话（截断后）会一并进入提示词，用于理解指代与追问关系，但事实依据仍只有会话摘要；
  - 模型可用时生成回答，但会校验「候选外故障类型」「会话来源之外的链接」，任一命中即拒绝；
  - 模型不可用 / 超时 / 校验不通过，一律降级为确定性回答（受控问题直答或事实分主题回答），
    并如实返回 mode、model_error；会话没有对应事实时明确回答「无法支持」。

会话与对话消息落盘在 SQLite（重启不丢），因此前端刷新后可列出历史会话、
恢复其事件轨迹与结果，并继续在同一会话里追问；删除仅按容量上限淘汰最旧会话。
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import threading
import time

from app.agent.llm_client import LLMClient, LLMError
from app.agent.report import PROTOTYPE_LIMITS
from app.config import settings
from app.rag.store import load_knowledge

# ---------------------------------------------------------------------------
# 受控问题集（任务书 P2 首批 7 问）
# ---------------------------------------------------------------------------
ALLOWED_QUESTIONS: list[dict] = [
    {"key": "why_fault", "text": "为什么判断为这个故障类型？"},
    {"key": "key_evidence", "text": "哪些证据最关键？"},
    {"key": "why_confidence", "text": "为什么置信度只有低或中？"},
    {"key": "why_unconfirmed", "text": "为什么没有确认故障？"},
    {"key": "where_wrong", "text": "这个判断可能错在哪里？"},
    {"key": "what_data", "text": "还需要采集什么数据？"},
    {"key": "full_report", "text": "生成一份完整的分析报告。"},
]

_NO_SUPPORT = "当前诊断记录无法支持该结论。"

_MAX_SESSIONS = 50


def _normalize(text: str) -> str:
    """去空白与中英文标点，只保留实义字符，用于问题匹配。"""
    keep = []
    for ch in str(text or ""):
        if ch.isalnum() or "\u4e00" <= ch <= "\u9fff":
            keep.append(ch)
    return "".join(keep)


def match_question(question: str) -> dict | None:
    """归一化后匹配受控问题；用户多打的尾部字符容忍（前缀匹配）。"""
    target = _normalize(question)
    if not target:
        return None
    for item in ALLOWED_QUESTIONS:
        normalized = _normalize(item["text"])
        if target == normalized or target.startswith(normalized):
            return item
    return None


# ---------------------------------------------------------------------------
# 会话存储：单文件 SQLite（会话表 + 对话消息表），进程内加锁串行访问
# ---------------------------------------------------------------------------
_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id      TEXT PRIMARY KEY,
    created_at      REAL NOT NULL,
    request_summary TEXT NOT NULL,
    result          TEXT NOT NULL,
    events          TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    created_at REAL NOT NULL,
    role       TEXT NOT NULL,
    content    TEXT NOT NULL,
    meta       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, id);
"""


def _dumps(value) -> str:
    return json.dumps(value, ensure_ascii=False)


class SessionStore:
    """会话与对话消息的落盘存储。

    - save/get 保持既有会话语义：只保存脱敏的结构化上下文（输入摘要、完整响应、事件列表）；
    - append_message/history/messages 保存并读取对话记录，支撑多轮上下文与刷新后恢复；
    - 会话数超过 maxsize 时按创建时间淘汰最旧会话，同时删除其对话消息。
    """

    def __init__(self, path=None, maxsize: int = _MAX_SESSIONS):
        self._path = str(path or settings.sessions_db)
        self._maxsize = maxsize
        self._lock = threading.Lock()
        parent = os.path.dirname(self._path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        # 会话写入可能来自线程池（自由问答走 asyncio.to_thread），因此关闭线程归属检查并统一加锁
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def save(self, session_id: str, *, request_summary: dict, result: dict, events: list[dict]) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO sessions(session_id, created_at, request_summary, result, events) "
                "VALUES(?, ?, ?, ?, ?)",
                (
                    session_id,
                    time.time(),
                    _dumps(request_summary),
                    _dumps(result),
                    _dumps(list(events or [])),
                ),
            )
            self._prune()
            self._conn.commit()

    def get(self, session_id: str) -> dict | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT session_id, created_at, request_summary, result, events FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        if row is None:
            return None
        return {
            "session_id": row[0],
            "created_at": row[1],
            "request_summary": json.loads(row[2]),
            "result": json.loads(row[3]),
            "events": json.loads(row[4]),
        }

    def list_sessions(self, limit: int = _MAX_SESSIONS) -> list[dict]:
        """会话列表摘要（按创建时间倒序）：只含列表页要展示的字段。"""
        with self._lock:
            rows = self._conn.execute(
                "SELECT session_id, created_at, request_summary, result FROM sessions "
                "ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
            counts = dict(
                self._conn.execute("SELECT session_id, COUNT(*) FROM messages GROUP BY session_id").fetchall()
            )
        items: list[dict] = []
        for session_id, created_at, summary_raw, result_raw in rows:
            summary = json.loads(summary_raw)
            result = json.loads(result_raw)
            report = result.get("report") or {}
            items.append(
                {
                    "session_id": session_id,
                    "created_at": created_at,
                    "normal_input": summary.get("normal_input"),
                    "abnormal_input": summary.get("abnormal_input"),
                    "status": result.get("status"),
                    "conclusion": report.get("conclusion"),
                    "confidence_level": (report.get("confidence") or {}).get("level"),
                    "message_count": int(counts.get(session_id, 0)),
                }
            )
        return items

    def append_message(self, session_id: str, role: str, content: str, meta: dict | None = None) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO messages(session_id, created_at, role, content, meta) VALUES(?, ?, ?, ?, ?)",
                (session_id, time.time(), str(role), str(content or ""), _dumps(meta or {})),
            )
            self._conn.commit()

    def history(self, session_id: str, limit: int) -> list[dict]:
        """最近 limit 条对话消息，按时间正序返回（供提示词使用）。"""
        with self._lock:
            rows = self._conn.execute(
                "SELECT role, content, meta FROM messages WHERE session_id = ? ORDER BY id DESC LIMIT ?",
                (session_id, max(0, int(limit))),
            ).fetchall()
        return [
            {"role": row[0], "content": row[1], "meta": json.loads(row[2])}
            for row in reversed(rows)
        ]

    def messages(self, session_id: str) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT role, content, meta, created_at FROM messages WHERE session_id = ? ORDER BY id",
                (session_id,),
            ).fetchall()
        return [
            {"role": row[0], "content": row[1], "meta": json.loads(row[2]), "created_at": row[3]}
            for row in rows
        ]

    def _prune(self) -> None:
        """淘汰超出容量的最旧会话（连同对话消息）；调用方已持锁。"""
        stale = [
            row[0]
            for row in self._conn.execute(
                "SELECT session_id FROM sessions ORDER BY created_at DESC LIMIT -1 OFFSET ?",
                (self._maxsize,),
            ).fetchall()
        ]
        if not stale:
            return
        marks = ",".join("?" * len(stale))
        self._conn.execute(f"DELETE FROM messages WHERE session_id IN ({marks})", stale)
        self._conn.execute(f"DELETE FROM sessions WHERE session_id IN ({marks})", stale)

    def __len__(self) -> int:
        with self._lock:
            return int(self._conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0])


store = SessionStore()


# ---------------------------------------------------------------------------
# 回答组装：完全来自会话上下文，字段引用写入 facts_used
# ---------------------------------------------------------------------------
def _fmt(value) -> str:
    try:
        return f"{float(value):.4g}"
    except (TypeError, ValueError):
        return str(value)


def _answer_why_fault(ctx: dict) -> tuple[str, list[str]]:
    result = ctx["result"]
    report = result.get("report") or {}
    candidates = result.get("candidates") or []
    comparison = result.get("comparison") or {}
    status = result.get("status")
    facts: list[str] = []

    if status != "ok" or not report.get("conclusion"):
        return _NO_SUPPORT, []

    lines = [f"本次判断为「{report.get('conclusion')}」，依据来自三个环节："]
    if candidates:
        top = candidates[0]
        lines.append(
            f"1. 知识检索：RAG 命中 {len(candidates)} 条候选，最相似为 {top.get('fault_type')}"
            f"（score={top.get('score')}，来源：{top.get('source')} · {top.get('locator')}）。"
        )
        facts.append("candidates[0]")
    evidence = comparison.get("evidence") or []
    if evidence:
        lines.append("2. 数据事实：" + "；".join(str(e) for e in evidence[:3]) + "。")
        facts.append("comparison.evidence")
    dominant = comparison.get("dominant_characteristic") or {}
    if dominant:
        lines.append(f"3. 频率证据：{dominant.get('reason')}。")
        facts.append("comparison.dominant_characteristic")
    llm_mode = (result.get("trace") or {}).get("llm_mode")
    if llm_mode == "llm":
        lines.append("结论措辞由模型在候选集合内生成；若模型不可用，同一路径会降级为模板报告。")
        facts.append("trace.llm_mode")
    return "\n".join(lines), facts


def _answer_key_evidence(ctx: dict) -> tuple[str, list[str]]:
    result = ctx["result"]
    comparison = result.get("comparison") or {}
    report = result.get("report") or {}
    facts: list[str] = []
    lines: list[str] = []

    dominant = comparison.get("dominant_characteristic") or {}
    if dominant:
        lines.append(f"- 包络谱主导特征：{dominant.get('reason')}。")
        facts.append("comparison.dominant_characteristic")
    changed = comparison.get("changed_features") or []
    changes = comparison.get("feature_changes") or {}
    for name in changed[:3]:
        block = changes.get(name) or {}
        lines.append(f"- {name} 由 {_fmt(block.get('normal'))} 变为 {_fmt(block.get('abnormal'))}（约 {_fmt(block.get('ratio'))} 倍）。")
        facts.append("comparison.feature_changes")
    matches = comparison.get("fault_frequency_matches") or []
    for m in matches[:3]:
        lines.append(
            f"- 谱峰与 {m.get('label')} 特征频率 {_fmt(m.get('frequency'))} Hz 对齐"
            f"（偏差 {m.get('deviation_pct')}%）。"
        )
        facts.append("comparison.fault_frequency_matches")
    conclusion_evidence = report.get("conclusion_evidence") or []
    if not lines and conclusion_evidence:
        lines.extend(f"- {e}" for e in conclusion_evidence[:3])
        facts.append("report.conclusion_evidence")
    if not lines:
        return _NO_SUPPORT, []
    return "本次诊断的关键证据：\n" + "\n".join(lines), facts


def _answer_why_confidence(ctx: dict) -> tuple[str, list[str]]:
    report = (ctx["result"].get("report") or {})
    confidence = report.get("confidence") or {}
    if not confidence.get("reason"):
        return _NO_SUPPORT, []
    return (
        f"置信等级为「{confidence.get('level')}」。原因：{confidence.get('reason')}",
        ["report.confidence"],
    )


def _answer_why_unconfirmed(ctx: dict) -> tuple[str, list[str]]:
    result = ctx["result"]
    status = result.get("status")
    if status == "insufficient_evidence":
        warnings = (result.get("trace") or {}).get("warnings") or []
        relevant = [w for w in warnings if ("证据不足" in str(w)) or ("无显著变化" in str(w)) or ("停止" in str(w))]
        reason = relevant[0] if relevant else "证据不足，无法确认故障类型。"
        return (
            f"本次诊断未确认故障类型，状态为 insufficient_evidence。原因：{reason} 此时系统不会基于高分候选强行下结论，而是输出「无法确认」并给出复核建议。",
            ["trace.warnings", "status"],
        )
    if status == "error":
        error = result.get("error") or {}
        return (
            f"本次诊断未产生结论：流程在错误状态停止（{error.get('code')}：{error.get('message')}）。",
            ["error", "status"],
        )
    if status == "ok":
        report = result.get("report") or {}
        return (
            f"本次诊断已给出结论「{report.get('conclusion')}」。若您指的是置信度偏低的原因，请参考「为什么置信度只有低或中？」。",
            ["report.conclusion", "status"],
        )
    return _NO_SUPPORT, []


def _answer_where_wrong(ctx: dict) -> tuple[str, list[str]]:
    result = ctx["result"]
    report = result.get("report") or {}
    trace = result.get("trace") or {}
    facts: list[str] = ["report.review_suggestions"]
    lines = ["本次判断可能在以下方面出错（均来自诊断记录中的复核建议与已知限制）："]
    suggestions = report.get("review_suggestions") or []
    for s in suggestions[:4]:
        lines.append(f"- {s}")
    review = trace.get("frequency_review")
    if review:
        lines.append(f"- {review.get('reason')} {review.get('review_advice')}")
        facts.append("trace.frequency_review")
    conflict = trace.get("evidence_conflict")
    if conflict:
        lines.append(f"- {conflict}")
        facts.append("trace.evidence_conflict")
    lines.append("- 原型适用边界：" + PROTOTYPE_LIMITS[0])
    facts.append("prototype_limits")
    return "\n".join(lines), facts


def _answer_what_data(ctx: dict) -> tuple[str, list[str]]:
    result = ctx["result"]
    report = result.get("report") or {}
    data_quality = result.get("data_quality") or {}
    facts: list[str] = ["report.review_suggestions"]
    lines = ["基于本次诊断的数据情况，建议补充采集："]
    suggestions = [s for s in (report.get("review_suggestions") or []) if any(k in str(s) for k in ("采集", "测量", "记录", "复测", "拆检", "数据"))]
    for s in suggestions[:4]:
        lines.append(f"- {s}")
    warnings = [w for w in ((result.get("trace") or {}).get("warnings") or []) if ("采集" in str(w)) or ("时长" in str(w))]
    for w in warnings[:2]:
        lines.append(f"- {w}")
        facts.append("trace.warnings")
    if len(lines) == 1:
        lines.append("- 按同工况继续采集多段 1 秒以上连续片段，用于交叉验证本次结论。")
    return "\n".join(lines), facts


def _answer_full_report(ctx: dict) -> tuple[str, list[str]]:
    markdown = ctx["result"].get("report_markdown")
    if not markdown:
        return _NO_SUPPORT, []
    return markdown, ["report_markdown"]


_ANSWER_BUILDERS = {
    "why_fault": _answer_why_fault,
    "key_evidence": _answer_key_evidence,
    "why_confidence": _answer_why_confidence,
    "why_unconfirmed": _answer_why_unconfirmed,
    "where_wrong": _answer_where_wrong,
    "what_data": _answer_what_data,
    "full_report": _answer_full_report,
}

CONTEXT_SCOPE = "本次会话保存的上下文：输入摘要、MCP 工具结果、RAG 候选与来源、Agent 节点轨迹、最终报告"


def answer_question(session_id: str, question: str) -> dict:
    """受控追问入口。返回 {session_id, question, question_key, answer, facts_used, context_scope}。

    抛出 KeyError 表示会话不存在；ValueError 表示问题不在允许清单内。
    """
    session = store.get(session_id)
    if session is None:
        raise KeyError(session_id)

    matched = match_question(question)
    if matched is None:
        raise ValueError("question_not_allowed")

    answer, facts = _ANSWER_BUILDERS[matched["key"]](session)
    store.append_message(session_id, "user", str(question), {"kind": "ask", "question_key": matched["key"]})
    store.append_message(
        session_id,
        "assistant",
        answer,
        {
            "kind": "ask",
            "question_key": matched["key"],
            "facts_used": facts,
            "context_scope": CONTEXT_SCOPE,
            "no_support": answer == _NO_SUPPORT,
        },
    )
    return {
        "session_id": session_id,
        "question": str(question),
        "question_key": matched["key"],
        "answer": answer,
        "facts_used": facts,
        "context_scope": CONTEXT_SCOPE,
        "no_support": answer == _NO_SUPPORT,
    }


# ---------------------------------------------------------------------------
# 会话内自由问答（P4）：证据约束的模型回答 + 确定性降级
# ---------------------------------------------------------------------------
CHAT_CONTEXT_SCOPE = (
    "本次会话保存的上下文：输入摘要、诊断状态与错误、MCP 工具结果、RAG 候选与来源、"
    "Agent 节点轨迹与告警、最终报告"
)

CHAT_SYSTEM_PROMPT = (
    "你是一个工业轴承振动诊断会话的问答助手。你只能使用用户给出的「本次会话事实」回答问题，"
    "严禁引入会话事实之外的故障类型、数值、来源或链接，严禁编造数据与出处。"
    "如果会话事实不足以回答，直接回答「当前诊断记录无法支持该结论。」并说明缺少什么依据，不要猜测。"
    "回答使用简体中文，可以用短段落与「-」列表，不要输出 JSON，不要输出代码块标记。"
)

# 提示词里单条历史消息的最大字符数（超出截断，避免多轮对话把上下文撑爆）
_CHAT_HISTORY_CHARS = 400

# 自由问答的确定性降级：按关键词把问题路由到对应的事实分区
_CHAT_TOPICS: list[tuple[str, tuple[str, ...]]] = [
    ("input", ("转速", "采样率", "测点", "设备", "工况", "输入", "文件", "样本", "基线")),
    ("candidates", ("候选", "故障类型", "类型", "知识", "来源", "文献", "引用", "出处", "检索")),
    ("evidence", ("证据", "特征", "频谱", "包络", "rms", "峭度", "数据质量", "校验", "信号", "异常")),
    ("confidence", ("置信", "可信", "准确", "把握", "靠谱")),
    ("review", ("复核", "建议", "下一步", "怎么办", "怎么处理", "检查", "拆检", "采集", "复测")),
]

_PSEUDO_TYPES = {"normal"}


def _known_fault_types() -> list[str]:
    """知识库声明的全部故障类型，用于拦截模型编造的候选外类型。

    读取失败（知识目录缺失等）时返回空列表：此时不做该维度校验，
    但回答仍会在确定性降级路径上重新组装。
    """
    try:
        entries = load_knowledge(settings.knowledge_dir)
    except Exception:
        return []
    return sorted(
        {
            str(entry.get("fault_type") or "").strip()
            for entry in entries
            if str(entry.get("fault_type") or "").strip()
        }
        - _PSEUDO_TYPES
    )


def _allowed_fault_types(session: dict) -> set[str]:
    """本次会话允许出现的故障类型：候选、来源、报告结论所引用的类型。"""
    result = session.get("result") or {}
    report = result.get("report") or {}
    allowed: set[str] = set()
    for block in (
        result.get("candidates") or [],
        result.get("sources") or [],
        report.get("conclusion_sources") or [],
        report.get("knowledge_candidates") or [],
    ):
        for item in block:
            if isinstance(item, dict) and item.get("fault_type"):
                allowed.add(str(item["fault_type"]))
    return allowed


def _allowed_urls(session: dict) -> list[str]:
    result = session.get("result") or {}
    report = result.get("report") or {}
    urls: list[str] = []
    for item in (result.get("sources") or []) + (report.get("conclusion_sources") or []):
        if isinstance(item, dict):
            url = str(item.get("source_url") or item.get("url") or "").strip()
            if url:
                urls.append(url.rstrip("/"))
    return urls


def _validate_chat_answer(answer: str, session: dict) -> str | None:
    """校验模型回答；返回拒绝原因，None 表示通过。"""
    text = (answer or "").strip()
    if not text:
        return "模型返回了空回答。"

    allowed = _allowed_fault_types(session)
    for fault_type in _known_fault_types():
        if fault_type in text and fault_type not in allowed:
            return f"模型回答引用了本次会话候选之外的故障类型（{fault_type}）。"

    urls = re.findall(r"https?://[^\s，。；、）)\]}\"'<>]+", text)
    allowed_urls = _allowed_urls(session)
    for url in urls:
        if not any(url.startswith(item) or item.startswith(url) for item in allowed_urls):
            return f"模型回答给出了本次会话来源之外的链接（{url}）。"
    return None


def build_context_digest(session: dict) -> tuple[str, list[str]]:
    """把会话事实拼成模型可读的上下文摘要；返回 (摘要文本, 实际用到的上下文字段)。"""
    result = session.get("result") or {}
    summary = session.get("request_summary") or {}
    report = result.get("report") or {}
    trace = result.get("trace") or {}
    comparison = result.get("comparison") or {}
    data_quality = result.get("data_quality") or {}
    facts: list[str] = []

    lines = ["【本次会话事实】（只能使用以下事实，未列出的内容一律视为无依据）"]

    input_bits = [
        f"正常（基线）输入 {summary.get('normal_input') or '未记录'}",
        f"异常输入 {summary.get('abnormal_input') or '未记录'}",
    ]
    for label, value in (
        ("转速", summary.get("rotation_speed")),
        ("采样率", summary.get("sampling_rate")),
        ("测点", summary.get("sensor_position")),
        ("设备类型", summary.get("device_type")),
    ):
        if value not in (None, ""):
            input_bits.append(f"{label} {value}")
    lines.append("输入与工况：" + "；".join(input_bits) + "。")
    facts.append("request_summary")

    lines.append(
        f"诊断状态：{result.get('status')}；工具调用 {trace.get('tool_calls')} 次；"
        f"检索方式 {trace.get('retrieval')}；报告模式 {trace.get('llm_mode')}。"
    )
    facts.append("result.status")

    error = result.get("error") or {}
    if error:
        lines.append(f"错误：{error.get('code')}：{error.get('message')}。")
        facts.append("result.error")

    if data_quality:
        errors = [str(item.get("message")) for item in (data_quality.get("errors") or [])[:2]]
        lines.append(
            f"数据质量：valid={data_quality.get('valid')}，comparable={data_quality.get('comparable')}"
            + (f"，错误：{'；'.join(errors)}。" if errors else "。")
        )
        facts.append("result.data_quality")

    candidates = result.get("candidates") or []
    if candidates:
        lines.append("知识库候选（只能引用下列故障类型）：")
        for item in candidates:
            lines.append(
                f"- {item.get('fault_type')}（{item.get('title')}，相似度 {item.get('score')}）："
                f"{item.get('evidence')}〔来源：{item.get('source')} · {item.get('locator')}〕"
            )
        allowed = "、".join(str(item.get("fault_type")) for item in candidates) or "（无）"
        lines.append(f"允许出现的故障类型：{allowed}（其他类型一律不得出现）。")
        facts.append("result.candidates")

    sources = result.get("sources") or []
    if sources:
        lines.append("知识来源（只能引用下列来源与链接）：")
        for item in sources[:5]:
            lines.append(
                f"- {item.get('name') or item.get('fault_type')}｜{item.get('locator')}｜{item.get('url')}"
            )
        facts.append("result.sources")

    evidence = [str(item) for item in (comparison.get("evidence") or [])[:4]]
    if evidence:
        lines.append("数据事实（来自信号分析工具，禁止改写数值）：")
        lines.extend(f"- {item}" for item in evidence)
        facts.append("result.comparison")

    if report:
        lines.append(f"报告结论：{report.get('conclusion')}")
        confidence = report.get("confidence") or {}
        lines.append(f"置信等级：{confidence.get('level')}；说明：{confidence.get('reason')}")
        suggestions = [str(item) for item in (report.get("review_suggestions") or [])[:5]]
        if suggestions:
            lines.append("建议复核项：")
            lines.extend(f"- {item}" for item in suggestions)
        lines.append("原型适用边界：" + PROTOTYPE_LIMITS[0])
        facts.append("result.report")

    warnings = [str(item) for item in (trace.get("warnings") or [])[:3]]
    if warnings:
        lines.append("流程告警：")
        lines.extend(f"- {item}" for item in warnings)
        facts.append("result.trace")

    lines.append("回答约束：只能引用以上事实；事实不足以回答时，请回答「当前诊断记录无法支持该结论。」。")
    return "\n".join(lines), facts


def _chat_input_answer(session: dict) -> tuple[str, list[str]]:
    summary = session.get("request_summary") or {}
    device = (session.get("result") or {}).get("device") or {}
    rows = [
        ("正常（基线）输入", summary.get("normal_input") or device.get("normal_file") or "未记录"),
        ("异常输入", summary.get("abnormal_input") or device.get("abnormal_file") or "未记录"),
        ("转速", summary.get("rotation_speed") or device.get("rotation_speed")),
        ("采样率", summary.get("sampling_rate") or device.get("sampling_rate")),
        ("测点", summary.get("sensor_position") or device.get("sensor_position")),
        ("设备类型", summary.get("device_type") or device.get("device_type")),
    ]
    lines = ["本次诊断的输入与工况："]
    lines.extend(f"- {label}：{value}。" for label, value in rows if value not in (None, ""))
    return "\n".join(lines), ["request_summary", "result.device"]


def _chat_candidates_answer(session: dict) -> tuple[str, list[str]]:
    result = session.get("result") or {}
    candidates = result.get("candidates") or []
    sources = result.get("sources") or []
    if not candidates:
        return (
            "本次会话没有检索到任何过阈值的知识库候选，因此不对具体故障类型下判断；"
            "证据不足时系统只输出「无法确认」并给出复核建议。",
            ["result.candidates"],
        )
    lines = [
        f"本次会话检索到 {len(candidates)} 条候选（检索方式 {(result.get('trace') or {}).get('retrieval')}），"
        "候选只是知识条目与查询的匹配结果，不等同于确诊："
    ]
    for item in candidates:
        lines.append(
            f"- {item.get('fault_type')}（{item.get('title')}，相似度 {item.get('score')}）："
            f"{item.get('evidence')}〔来源：{item.get('source')} · {item.get('locator')}〕"
        )
    if sources:
        lines.append("知识来源：")
        for item in sources[:5]:
            lines.append(
                f"- {item.get('name') or item.get('fault_type')}｜{item.get('locator')}｜{item.get('url')}"
            )
    return "\n".join(lines), ["result.candidates", "result.sources"]


def _chat_evidence_answer(session: dict) -> tuple[str, list[str]]:
    result = session.get("result") or {}
    comparison = result.get("comparison") or {}
    data_quality = result.get("data_quality") or {}
    facts: list[str] = []
    lines: list[str] = []

    if data_quality:
        lines.append(
            f"数据质量：校验 valid={data_quality.get('valid')}，可比 comparable={data_quality.get('comparable')}。"
        )
        facts.append("result.data_quality")
    dominant = comparison.get("dominant_characteristic") or {}
    if dominant:
        lines.append(f"- {dominant.get('reason')}。")
        facts.append("result.comparison")
    for item in (comparison.get("evidence") or [])[:4]:
        lines.append(f"- {item}")
        facts.append("result.comparison")
    if not lines:
        return (
            "本次会话没有可用的对比证据（流程可能在校验或分析阶段就停止），无法讨论具体特征。",
            ["result.status"],
        )
    return "本次会话中与信号有关的事实：\n" + "\n".join(lines), facts


def _chat_confidence_answer(session: dict) -> tuple[str, list[str]]:
    report = ((session.get("result") or {}).get("report")) or {}
    confidence = report.get("confidence") or {}
    if not confidence.get("reason"):
        return _NO_SUPPORT, []
    return (
        f"本次结论的置信等级为「{confidence.get('level')}」。说明：{confidence.get('reason')}",
        ["result.report.confidence"],
    )


def _chat_review_answer(session: dict) -> tuple[str, list[str]]:
    report = ((session.get("result") or {}).get("report")) or {}
    trace = (session.get("result") or {}).get("trace") or {}
    lines = ["本次诊断给出的复核与后续建议："]
    suggestions = [str(item) for item in (report.get("review_suggestions") or [])[:5]]
    lines.extend(f"- {item}" for item in suggestions)
    for key in ("frequency_review", "evidence_conflict"):
        block = trace.get(key)
        if key == "frequency_review" and block:
            lines.append(f"- {block.get('reason')} {block.get('review_advice')}")
        elif key == "evidence_conflict" and block:
            lines.append(f"- {block}")
    if len(lines) == 1:
        return _NO_SUPPORT, []
    return "\n".join(lines), ["result.report.review_suggestions", "result.trace"]


_CHAT_TOPIC_BUILDERS = {
    "input": _chat_input_answer,
    "candidates": _chat_candidates_answer,
    "evidence": _chat_evidence_answer,
    "confidence": _chat_confidence_answer,
    "review": _chat_review_answer,
}


def _match_chat_topic(question: str) -> str | None:
    text = str(question or "").lower()
    for key, keywords in _CHAT_TOPICS:
        if any(keyword in text for keyword in keywords):
            return key
    return None


def _deterministic_chat_answer(session: dict, question: str) -> tuple[str, list[str]]:
    """确定性回答：受控问题直答 → 主题事实回答 → 事实摘要（并声明无法针对性回答）。"""
    matched = match_question(question)
    if matched is not None:
        return _ANSWER_BUILDERS[matched["key"]](session)

    topic = _match_chat_topic(question)
    if topic is not None:
        return _CHAT_TOPIC_BUILDERS[topic](session)

    digest, facts = build_context_digest(session)
    body = "\n".join(
        line for line in digest.splitlines()[1:] if line.strip() and not line.startswith("回答约束")
    )
    return (
        "本次会话记录无法直接支撑针对该问题的回答；以下是与本次诊断相关的可用事实：\n"
        + body
        + "\n如需针对性结论，请补充对应的测量数据或说明后重新诊断。",
        facts,
    )


def _format_history(messages: list[dict]) -> str:
    """把最近几轮对话压成提示词片段：空白折叠 + 单条截断，控制上下文长度。"""
    lines: list[str] = []
    for item in messages:
        role = "用户" if item.get("role") == "user" else "助手"
        text = " ".join(str(item.get("content") or "").split())
        if len(text) > _CHAT_HISTORY_CHARS:
            text = text[:_CHAT_HISTORY_CHARS] + "…"
        if text:
            lines.append(f"{role}：{text}")
    return "\n".join(lines)


def answer_free_question(session_id: str, question: str, *, client=None) -> dict:
    """会话内自由问答入口。返回 {session_id, question, answer, mode, facts_used, ...}。

    抛出 KeyError 表示会话不存在。mode=llm 表示回答由模型在会话事实内生成并通过校验；
    mode=deterministic 表示模型不可用 / 失败 / 回答被拒绝，改由确定性逻辑组装。
    每次问答都会写入会话消息表，既作为后续轮次的上下文，也供前端刷新后恢复。
    """
    session = store.get(session_id)
    if session is None:
        raise KeyError(session_id)

    question = str(question or "").strip()
    digest, digest_facts = build_context_digest(session)
    history = _format_history(store.history(session_id, settings.chat_history_turns * 2))
    mode = "deterministic"
    model_error: dict | None = None
    answer: str | None = None
    facts: list[str] = []

    active = client if client is not None else LLMClient()
    if bool(getattr(active, "available", False)):
        prompt = digest
        if history:
            prompt += (
                "\n\n【最近对话】（仅用于理解指代与追问关系，回答仍必须以下方会话事实为依据，"
                "不得把历史对话当成新的事实来源）\n" + history
            )
        prompt += f"\n\n用户问题：{question}"
        try:
            text = active.chat(CHAT_SYSTEM_PROMPT, prompt)
            rejected = _validate_chat_answer(text, session)
            if rejected is None:
                mode = "llm"
                answer = text.strip()
                facts = digest_facts
            else:
                model_error = {"code": "answer_rejected", "message": rejected}
        except LLMError as exc:
            model_error = exc.to_dict()
        except Exception as exc:  # 兜底：假客户端或 SDK 抛出非 LLMError 异常
            model_error = {"code": "llm_error", "message": f"{type(exc).__name__}: {exc}"}

    if answer is None:
        answer, facts = _deterministic_chat_answer(session, question)

    no_support = answer == _NO_SUPPORT
    store.append_message(session_id, "user", question, {"kind": "chat"})
    store.append_message(
        session_id,
        "assistant",
        answer,
        {"kind": "chat", "mode": mode, "facts_used": facts, "no_support": no_support, "model_error": model_error},
    )

    return {
        "session_id": session_id,
        "question": question,
        "answer": answer,
        "mode": mode,
        "facts_used": facts,
        "context_scope": CHAT_CONTEXT_SCOPE,
        "no_support": no_support,
        "model_error": model_error,
    }
