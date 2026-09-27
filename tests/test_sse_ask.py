"""P1/P2 新接口测试：SSE 事件流、断开安全、旧接口兼容与受控追问。

覆盖任务书 §6.3 要求的 7 类测试：
  1. SSE 正常路径事件顺序
  2. 工具失败事件（损坏样本）
  3. 证据不足分支（正常 vs 正常）
  4. 模型超时回退
  5. SSE 客户端断开
  6. 旧 /api/diagnose 兼容
  7. 追问越权/无依据拦截
"""

from __future__ import annotations

import asyncio
import json

import httpx

from app.agent import sessions
from app.main import app
from tests.helpers import CORRUPTED, INNER_RACE, NORMAL, ScriptedLLMClient, TimeoutLLMClient

BASE_URL = "http://testserver"

EVENT_KEYS = {"session_id", "event", "node", "tool", "status", "label", "summary", "elapsed_ms", "payload"}

INNER_FORM = {
    "sampling_rate": 12000,
    "rotation_speed": 1797,
    "normal_sample": NORMAL,
    "abnormal_sample": INNER_RACE,
}


def collect_stream(form: dict) -> list[dict]:
    """消费一次完整 SSE 流，返回解析后的事件列表。"""

    async def run() -> list[dict]:
        events: list[dict] = []
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url=BASE_URL) as client:
            async with client.stream("POST", "/api/diagnose/stream", data=form) as response:
                assert response.status_code == 200
                assert response.headers["content-type"].startswith("text/event-stream")
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        events.append(json.loads(line[6:]))
        return events

    return asyncio.run(run())


def call(method: str, url: str, **kwargs) -> httpx.Response:
    async def run() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url=BASE_URL) as client:
            return await client.request(method, url, **kwargs)

    return asyncio.run(run())


def assert_event_shape(events: list[dict]) -> None:
    for event in events:
        assert EVENT_KEYS <= set(event), f"事件缺少统一字段：{sorted(set(event))}"
        assert isinstance(event["elapsed_ms"], int) and event["elapsed_ms"] >= 0
        assert event["session_id"]


# ---------------------------------------------------------------------------
# 1. SSE 正常路径事件顺序
# ---------------------------------------------------------------------------
def test_sse_normal_path_event_sequence(local_backend):
    events = collect_stream(INNER_FORM)

    assert events, "SSE 流没有返回任何事件"
    assert_event_shape(events)
    seq = [e["event"] for e in events]

    # 首尾事件
    assert seq[0] == "diagnosis_started"
    assert seq[-1] == "diagnosis_completed"

    # 最低事件顺序（任务书 §4 P1）：各节点 started/completed 齐全
    started_nodes = [e["node"] for e in events if e["event"] == "node_started"]
    for node in ("validate_node", "mcp_analysis_node", "rag_retrieval_node", "evidence_check_node", "llm_node", "report_node"):
        assert node in started_nodes, f"缺少 {node} 的事件"
        node_done = [e for e in events if e["node"] == node and e["event"] == "node_completed"]
        assert node_done and node_done[0]["status"] == "completed", f"{node} 未正常完成"

    # 4 次工具调用的 started/completed 成对且顺序正确
    tool_starts = [e["tool"] for e in events if e["event"] == "tool_started"]
    assert tool_starts == [
        "validate_vibration_data",
        "extract_vibration_features",
        "extract_vibration_features",
        "compare_normal_abnormal",
    ]
    tools_completed = [e for e in events if e["event"] == "tool_completed"]
    assert all(e["status"] == "completed" for e in tools_completed)

    # 分支选择事件指向 llm_node
    branches = [e for e in events if e["event"] == "branch_selected"]
    assert len(branches) == 1
    assert branches[0]["payload"]["branch"] == "llm_node"

    # 完成事件携带完整结果
    final = events[-1]
    result = final["payload"]["result"]
    assert result["status"] == "ok"
    assert result["candidates"][0]["fault_type"] == "inner_race_fault"
    assert result["report"] and result["report_markdown"]
    assert result["trace"]["tool_calls"] == 4
    assert result["trace"]["nodes"] == [
        "validate_node", "mcp_analysis_node", "rag_retrieval_node",
        "evidence_check_node", "llm_node", "report_node",
    ]
    # 事件顺序与真实执行一致：validate 在 mcp 之前，rag 在 evidence 之前
    seq_nodes = [e["node"] for e in events if e["event"] == "node_started"]
    assert seq_nodes == [
        "validate_node", "mcp_analysis_node", "rag_retrieval_node",
        "evidence_check_node", "llm_node", "report_node",
    ]


# ---------------------------------------------------------------------------
# 2. 工具失败事件（损坏样本：校验工具失败即停）
# ---------------------------------------------------------------------------
def test_sse_tool_failure_emits_failed_events(local_backend):
    """损坏样本：数据校验不通过 → 节点失败事件 + diagnosis_failed，后续步骤全部停止。"""
    form = {**INNER_FORM, "abnormal_sample": CORRUPTED}
    events = collect_stream(form)
    assert_event_shape(events)

    seq = [e["event"] for e in events]
    assert seq[0] == "diagnosis_started"
    assert seq[-1] == "diagnosis_failed"

    # 数据校验步骤失败：mcp_analysis_node 以 failed 结束并携带错误摘要
    failed_nodes = [
        e for e in events
        if e["event"] == "node_completed" and e["status"] == "failed"
    ]
    assert len(failed_nodes) == 1
    assert failed_nodes[0]["node"] == "mcp_analysis_node"
    assert failed_nodes[0]["summary"]

    # 失败后流程停止：不再有后续工具与检索节点
    tool_starts = [e for e in events if e["event"] == "tool_started"]
    assert len(tool_starts) == 1
    assert tool_starts[0]["tool"] == "validate_vibration_data"
    started_nodes = [e["node"] for e in events if e["event"] == "node_started"]
    assert "rag_retrieval_node" not in started_nodes

    final = events[-1]
    result = final["payload"]["result"]
    assert result["status"] == "error"
    assert result["error"]["code"] == "missing_value"
    assert result["trace"]["tool_calls"] == 1


# ---------------------------------------------------------------------------
# 3. 证据不足分支（正常 vs 正常 → 跳过模型）
# ---------------------------------------------------------------------------
def test_sse_insufficient_evidence_branch(local_backend):
    form = {**INNER_FORM, "abnormal_sample": NORMAL}
    events = collect_stream(form)
    assert_event_shape(events)

    # 分支事件指向 report_node（跳过模型），并说明原因
    branches = [e for e in events if e["event"] == "branch_selected"]
    assert len(branches) == 1
    assert branches[0]["payload"]["branch"] == "report_node"
    assert branches[0]["status"] == "skipped"
    assert "无显著变化" in branches[0]["summary"]

    # 未执行模型节点
    started_nodes = [e["node"] for e in events if e["event"] == "node_started"]
    assert "llm_node" not in started_nodes
    assert "report_node" in started_nodes

    final = events[-1]
    result = final["payload"]["result"]
    assert final["event"] == "diagnosis_completed"
    assert result["status"] == "insufficient_evidence"
    assert result["candidates"] == [] and result["sources"] == []


# ---------------------------------------------------------------------------
# 4. 模型超时回退（注入假 LLMClient，SSE 仍走同一套核心逻辑）
# ---------------------------------------------------------------------------
def test_sse_llm_timeout_falls_back_to_template(local_backend, monkeypatch):
    monkeypatch.setattr("app.agent.graph.LLMClient", TimeoutLLMClient)
    events = collect_stream(INNER_FORM)

    # 模型降级警告事件
    warnings = [e for e in events if e["event"] == "warning" and e["node"] == "llm_node"]
    assert warnings and warnings[0]["status"] == "fallback"
    assert warnings[0]["payload"]["llm_error"]["code"] == "timeout"

    final = events[-1]
    result = final["payload"]["result"]
    assert final["event"] == "diagnosis_completed"
    assert result["trace"]["llm_mode"] == "template_fallback"
    # 降级保留已算出的证据与候选
    assert result["candidates"] and result["candidates"][0]["fault_type"] == "inner_race_fault"
    assert result["report"]


# ---------------------------------------------------------------------------
# 5. SSE 客户端断开：后台诊断继续跑完、会话照常保存、无未捕获异常
# ---------------------------------------------------------------------------
def test_sse_client_disconnect_is_safe(local_backend):
    async def run() -> dict:
        transport = httpx.ASGITransport(app=app)
        session_id = None
        async with httpx.AsyncClient(transport=transport, base_url=BASE_URL) as client:
            async with client.stream("POST", "/api/diagnose/stream", data=INNER_FORM) as response:
                assert response.status_code == 200
                count = 0
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        event = json.loads(line[6:])
                        if session_id is None:
                            session_id = event["session_id"]
                        count += 1
                        if count >= 3:  # 消费 3 个事件后主动断开
                            break
        assert count == 3
        # 客户端已断开：等后台诊断跑完（local 后端毫秒级）并保存会话
        for _ in range(100):
            if sessions.store.get(session_id) is not None:
                break
            await asyncio.sleep(0.05)
        return {"session_id": session_id, "saved": sessions.store.get(session_id) is not None}

    outcome = asyncio.run(run())
    assert outcome["saved"], "客户端断开后后台诊断未完成或会话未保存"
    session = sessions.store.get(outcome["session_id"])
    assert session["result"]["status"] == "ok"
    assert session["events"], "断开后会话仍应保留完整事件轨迹"


# ---------------------------------------------------------------------------
# 6. 旧 /api/diagnose 兼容（字段不变 + 新增 session_id 与会话保存）
# ---------------------------------------------------------------------------
def test_legacy_diagnose_stays_compatible_and_saves_session(local_backend):
    response = call("POST", "/api/diagnose", data=INNER_FORM)

    assert response.status_code == 200
    body = response.json()
    # 旧 13 键全部保留
    assert {
        "status", "error", "device", "data_quality", "features", "comparison",
        "rag", "candidates", "report", "report_markdown", "sources",
        "review_suggestions", "trace",
    } <= set(body)
    assert body["status"] == "ok"
    assert body["trace"]["tool_calls"] == 4
    assert body["candidates"][0]["fault_type"] == "inner_race_fault"

    # 新增：session_id 且会话可追问
    assert body["session_id"]
    session = sessions.store.get(body["session_id"])
    assert session is not None
    assert session["result"]["candidates"][0]["fault_type"] == "inner_race_fault"
    assert session["request_summary"]["abnormal_input"] == INNER_RACE


def test_sse_missing_input_returns_plain_400(local_backend):
    """流开始前输入就缺失：普通 4xx JSON，不产生半截事件流。"""
    response = call("POST", "/api/diagnose/stream", data={"sampling_rate": 12000, "rotation_speed": 1797})
    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "missing_input"


# ---------------------------------------------------------------------------
# 7. 受控追问：越权拦截、会话校验与无依据回答
# ---------------------------------------------------------------------------
def _diagnose_and_get_session(form: dict) -> str:
    response = call("POST", "/api/diagnose", data=form)
    assert response.status_code == 200
    return response.json()["session_id"]


def test_ask_answers_from_session_context(local_backend):
    session_id = _diagnose_and_get_session(INNER_FORM)

    response = call("POST", f"/api/sessions/{session_id}/ask", json={"question": "为什么判断为这个故障类型？"})
    assert response.status_code == 200
    body = response.json()
    assert body["question_key"] == "why_fault"
    assert body["answer"]
    assert "inner_race_fault" in body["answer"]
    assert body["facts_used"]
    assert "RAG" in body["context_scope"]


def test_ask_all_seven_questions_are_allowed(local_backend):
    session_id = _diagnose_and_get_session(INNER_FORM)
    for item in sessions.ALLOWED_QUESTIONS:
        response = call("POST", f"/api/sessions/{session_id}/ask", json={"question": item["text"]})
        assert response.status_code == 200, item["text"]
        assert response.json()["question_key"] == item["key"]


def test_ask_rejects_out_of_scope_question(local_backend):
    session_id = _diagnose_and_get_session(INNER_FORM)
    response = call("POST", f"/api/sessions/{session_id}/ask", json={"question": "帮我预测明天股票涨跌"})
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert detail["code"] == "question_not_allowed"
    assert len(detail["allowed_questions"]) == 7


def test_ask_unknown_session_returns_404(local_backend):
    response = call("POST", "/api/sessions/deadbeef00/ask", json={"question": "哪些证据最关键？"})
    assert response.status_code == 404
    assert response.json()["detail"]["code"] == "session_not_found"


def test_ask_without_factual_basis_returns_no_support(local_backend):
    """错误会话（无报告、无候选）上问「为什么判断为这个故障」：必须明确说无法支持。"""
    session_id = _diagnose_and_get_session({**INNER_FORM, "abnormal_sample": CORRUPTED})
    response = call("POST", f"/api/sessions/{session_id}/ask", json={"question": "为什么判断为这个故障类型？"})
    assert response.status_code == 200
    body = response.json()
    assert body["no_support"] is True
    assert body["answer"] == "当前诊断记录无法支持该结论。"
    assert body["facts_used"] == []


# ---------------------------------------------------------------------------
# 8. 会话内自由问答（P4）：事实约束、模型校验与确定性降级
# ---------------------------------------------------------------------------
def _chat(session_id: str, question: str) -> httpx.Response:
    return call("POST", f"/api/sessions/{session_id}/chat", json={"question": question})


def test_chat_answers_candidate_question_from_session_facts(local_backend):
    """无模型时走确定性回答：候选类问题必须来自本次会话的真实候选。"""
    session_id = _diagnose_and_get_session(INNER_FORM)

    response = _chat(session_id, "这次判断更可能是哪个故障类型？")
    assert response.status_code == 200
    body = response.json()
    assert body["mode"] == "deterministic"
    assert body["model_error"] is None
    assert "inner_race_fault" in body["answer"]
    assert "CWRU" in body["answer"]
    assert body["facts_used"]
    assert "候选" in body["context_scope"]


def test_chat_input_topic_uses_recorded_condition(local_backend):
    session_id = _diagnose_and_get_session(INNER_FORM)

    body = _chat(session_id, "这次用的转速和采样率是多少？").json()
    assert "1797" in body["answer"]
    assert "12000" in body["answer"]
    assert "request_summary" in body["facts_used"]


def test_chat_controlled_question_falls_back_to_deterministic_answer(local_backend):
    session_id = _diagnose_and_get_session(INNER_FORM)

    body = _chat(session_id, "为什么判断为这个故障类型？").json()
    assert body["mode"] == "deterministic"
    assert "inner_race_fault" in body["answer"]
    assert body["facts_used"]


def test_chat_accepts_model_answer_inside_session_scope(local_backend, monkeypatch):
    session_id = _diagnose_and_get_session(INNER_FORM)
    scripted = ScriptedLLMClient("本次候选为 inner_race_fault：包络谱 BPFI 族能量相对基线明显上升。")
    monkeypatch.setattr("app.agent.sessions.LLMClient", lambda: scripted)

    body = _chat(session_id, "这次判断更可能是哪个故障类型？").json()
    assert body["mode"] == "llm"
    assert body["answer"].startswith("本次候选为 inner_race_fault")
    assert body["model_error"] is None
    # 进入提示词的是会话事实摘要，而不是原始文件内容
    system, user = scripted.calls[0]
    assert "本次会话事实" in user
    assert "inner_race_fault" in user


def test_chat_rejects_fault_type_outside_candidates(local_backend, monkeypatch):
    session_id = _diagnose_and_get_session(INNER_FORM)
    scripted = ScriptedLLMClient("我判断为 ball_fault，因为 2×BSF 能量突出。")
    monkeypatch.setattr("app.agent.sessions.LLMClient", lambda: scripted)

    body = _chat(session_id, "这次判断更可能是哪个故障类型？").json()
    assert body["mode"] == "deterministic"
    assert body["model_error"]["code"] == "answer_rejected"
    assert "ball_fault" not in body["answer"]
    assert "inner_race_fault" in body["answer"]


def test_chat_rejects_url_outside_session_sources(local_backend, monkeypatch):
    session_id = _diagnose_and_get_session(INNER_FORM)
    scripted = ScriptedLLMClient("参考 https://example.com/fake-bearing-guide 的结论：内圈故障。")
    monkeypatch.setattr("app.agent.sessions.LLMClient", lambda: scripted)

    body = _chat(session_id, "这个结论的出处是什么？").json()
    assert body["mode"] == "deterministic"
    assert body["model_error"]["code"] == "answer_rejected"
    assert "example.com" not in body["answer"]


def test_chat_model_timeout_falls_back(local_backend, monkeypatch):
    session_id = _diagnose_and_get_session(INNER_FORM)
    monkeypatch.setattr("app.agent.sessions.LLMClient", TimeoutLLMClient)

    body = _chat(session_id, "这个结论的置信度如何？").json()
    assert body["mode"] == "deterministic"
    assert body["model_error"]["code"] == "timeout"
    assert body["answer"]


def test_chat_error_session_answers_from_error_facts(local_backend):
    """损坏样本会话：回答只能引用错误与校验事实，不得编造故障类型。"""
    session_id = _diagnose_and_get_session({**INNER_FORM, "abnormal_sample": CORRUPTED})

    body = _chat(session_id, "为什么没有确认故障？").json()
    assert body["answer"]
    assert ("missing_value" in body["answer"]) or ("校验" in body["answer"])
    assert "inner_race_fault" not in body["answer"]


def test_chat_empty_question_returns_400(local_backend):
    session_id = _diagnose_and_get_session(INNER_FORM)

    response = _chat(session_id, "   ")
    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "empty_question"


def test_chat_unknown_session_returns_404(local_backend):
    response = _chat("deadbeef00", "哪些证据最关键？")
    assert response.status_code == 404
    assert response.json()["detail"]["code"] == "session_not_found"
