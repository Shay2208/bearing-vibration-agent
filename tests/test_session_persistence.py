"""会话持久化测试：SQLite 落盘、会话列表 / 详情接口、多轮对话上下文与容量淘汰。

覆盖点：
  1. 会话与事件落盘，重建存储对象（模拟重启）后仍可读取；
  2. GET /api/sessions 列表字段与 GET /api/sessions/{id} 详情（含对话消息）；
  3. 未知会话详情返回 404 session_not_found；
  4. 受控追问与自由问答的消息落盘，前端可据此恢复；
  5. 第二轮提问的提示词携带最近一轮对话（且带「仅用于理解指代」约束）；
  6. 历史轮数受 CHAT_HISTORY_TURNS 限制，超出的更早轮次不进提示词；
  7. 会话数超上限时淘汰最旧会话，并连同其对话消息一起删除。
"""

from __future__ import annotations

import asyncio

import httpx

from app.agent import sessions
from app.config import settings
from app.main import app
from tests.helpers import INNER_RACE, NORMAL, ScriptedLLMClient

BASE_URL = "http://testserver"

FORM = {
    "sampling_rate": 12000,
    "rotation_speed": 1797,
    "normal_sample": NORMAL,
    "abnormal_sample": INNER_RACE,
}

REPLY = "本次候选为 inner_race_fault：包络谱 BPFI 族能量相对基线明显上升。"


def call(method: str, url: str, **kwargs) -> httpx.Response:
    async def run() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url=BASE_URL) as client:
            return await client.request(method, url, **kwargs)

    return asyncio.run(run())


def diagnose(form: dict | None = None) -> str:
    response = call("POST", "/api/diagnose", data=form or FORM)
    assert response.status_code == 200
    return response.json()["session_id"]


def chat(session_id: str, question: str) -> dict:
    response = call("POST", f"/api/sessions/{session_id}/chat", json={"question": question})
    assert response.status_code == 200
    return response.json()


# ---------------------------------------------------------------------------
# 1. 落盘与「重启不丢」
# ---------------------------------------------------------------------------
def test_session_and_events_survive_store_reopen(local_backend):
    session_id = diagnose()

    reopened = sessions.SessionStore(path=settings.sessions_db)
    session = reopened.get(session_id)
    assert session is not None, "重建存储对象后会话丢失，说明没有真正落盘"
    assert session["result"]["status"] == "ok"
    assert session["request_summary"]["abnormal_input"] == INNER_RACE
    assert any(item["event"] == "diagnosis_completed" for item in session["events"]), "事件轨迹应一并落盘"


def test_session_list_endpoint_returns_summary(local_backend):
    session_id = diagnose()

    body = call("GET", "/api/sessions").json()
    assert body["count"] >= 1
    entry = next(item for item in body["sessions"] if item["session_id"] == session_id)
    assert entry["status"] == "ok"
    assert entry["abnormal_input"] == INNER_RACE
    assert entry["conclusion"]
    assert entry["message_count"] == 0
    assert entry["created_at"] > 0
    # 列表接口不返回完整结果与事件，避免把大对象塞进列表
    assert "result" not in entry and "events" not in entry


def test_session_detail_endpoint_returns_full_record(local_backend):
    session_id = diagnose()

    body = call("GET", f"/api/sessions/{session_id}").json()
    assert body["session_id"] == session_id
    assert body["result"]["candidates"][0]["fault_type"] == "inner_race_fault"
    assert body["events"], "详情应带回事件轨迹供前端重放"
    assert body["messages"] == []


def test_session_detail_unknown_returns_404(local_backend):
    response = call("GET", "/api/sessions/deadbeef00")
    assert response.status_code == 404
    assert response.json()["detail"]["code"] == "session_not_found"


# ---------------------------------------------------------------------------
# 2. 对话消息落盘与刷新恢复
# ---------------------------------------------------------------------------
def test_chat_and_ask_messages_are_persisted(local_backend):
    session_id = diagnose()

    chat(session_id, "这次判断更可能是哪个故障类型？")
    call("POST", f"/api/sessions/{session_id}/ask", json={"question": "哪些证据最关键？"})

    messages = sessions.store.messages(session_id)
    assert [item["role"] for item in messages] == ["user", "assistant", "user", "assistant"]
    assert messages[0]["meta"]["kind"] == "chat"
    assert messages[1]["meta"]["mode"] == "deterministic"
    assert messages[1]["meta"]["facts_used"], "回答的依据字段应一并落盘"
    assert messages[2]["meta"]["kind"] == "ask"
    assert messages[3]["meta"]["question_key"] == "key_evidence"

    # 刷新后走详情接口，同样能拿回完整对话
    detail = call("GET", f"/api/sessions/{session_id}").json()
    assert len(detail["messages"]) == 4
    assert detail["messages"][0]["content"] == "这次判断更可能是哪个故障类型？"
    assert detail["messages"][1]["content"]


def test_session_list_counts_messages(local_backend):
    session_id = diagnose()
    chat(session_id, "这次判断更可能是哪个故障类型？")

    body = call("GET", "/api/sessions").json()
    entry = next(item for item in body["sessions"] if item["session_id"] == session_id)
    assert entry["message_count"] == 2


# ---------------------------------------------------------------------------
# 3. 多轮上下文（最近对话进提示词，事实依据不变）
# ---------------------------------------------------------------------------
def test_second_turn_prompt_carries_previous_turn(local_backend, monkeypatch):
    session_id = diagnose()
    scripted = ScriptedLLMClient(REPLY)
    monkeypatch.setattr("app.agent.sessions.LLMClient", lambda: scripted)

    chat(session_id, "标记甲：这次判断更可能是哪个故障类型？")
    chat(session_id, "标记丙：那它的置信度如何？")

    assert len(scripted.calls) == 2
    first_prompt = scripted.calls[0][1]
    second_prompt = scripted.calls[1][1]
    assert "最近对话" not in first_prompt, "首轮没有历史，不应出现历史片段"
    assert "最近对话" in second_prompt and "标记甲" in second_prompt
    assert "不得把历史对话当成新的事实来源" in second_prompt
    # 事实摘要仍在提示词中，多轮不改变依据范围
    assert "本次会话事实" in second_prompt and "inner_race_fault" in second_prompt


def test_history_window_follows_turn_limit(local_backend, monkeypatch):
    session_id = diagnose()
    scripted = ScriptedLLMClient(REPLY)
    monkeypatch.setattr("app.agent.sessions.LLMClient", lambda: scripted)

    previous = settings.chat_history_turns
    object.__setattr__(settings, "chat_history_turns", 1)
    try:
        chat(session_id, "标记甲：这次判断更可能是哪个故障类型？")
        chat(session_id, "标记乙：那它的置信度如何？")
        chat(session_id, "标记丙：还需要采集什么数据？")
    finally:
        object.__setattr__(settings, "chat_history_turns", previous)

    third_prompt = scripted.calls[2][1]
    assert "标记乙" in third_prompt, "最近一轮应进入提示词"
    assert "标记甲" not in third_prompt, "超出轮数上限的更早对话不应进入提示词"


# ---------------------------------------------------------------------------
# 4. 容量淘汰
# ---------------------------------------------------------------------------
def test_store_prunes_oldest_session_with_its_messages(tmp_path):
    store = sessions.SessionStore(path=tmp_path / "sessions.db", maxsize=2)
    for index in range(3):
        store.save(
            f"session-{index}",
            request_summary={"abnormal_input": f"sample-{index}"},
            result={"status": "ok"},
            events=[],
        )
        store.append_message(f"session-{index}", "user", f"问题 {index}")

    assert len(store) == 2
    assert store.get("session-0") is None
    assert store.messages("session-0") == []
    assert store.get("session-2") is not None
    assert [item["session_id"] for item in store.list_sessions()] == ["session-2", "session-1"]