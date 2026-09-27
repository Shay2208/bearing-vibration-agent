"""接口层测试（app/main.py）：/health、/api/samples、/api/diagnose 正常与异常路径。

用 httpx.ASGITransport 直接驱动 ASGI 应用，不需要起真实服务器。
"""

from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path

import httpx

from app.config import settings
from app.agent import orchestrator
from app.main import app
from app.rag import retriever
from app.rag.store import KnowledgeStore
from tests.helpers import BALL, CORRUPTED, INNER_RACE, NORMAL, OUTER_RACE, fake_embedder, build_request

BASE_URL = "http://testserver"
SAMPLE_IDS = {NORMAL, INNER_RACE, OUTER_RACE, BALL, CORRUPTED}


def call(method: str, url: str, **kwargs) -> httpx.Response:
    async def run() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url=BASE_URL) as client:
            return await client.request(method, url, **kwargs)

    return asyncio.run(run())


def test_health_returns_200(local_backend):
    response = call("GET", "/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["mcp_backend"] == "local"
    assert body["sample_count"] == 5
    assert body["llm_available"] is False
    assert body["retrieval_mode"] == "bm25"


def test_health_stays_responsive_while_model_waits(local_backend):
    """A waiting model request must leave the shared event loop available."""
    entered = threading.Event()
    release = threading.Event()

    class WaitingModel:
        available = True

        def chat(self, system, user):
            entered.set()
            if not release.wait(5):
                raise TimeoutError("Test could not release model; event loop may be blocked")
            return json.dumps({
                "selected_fault_type": "inner_race_fault",
                "summary": "待复核的内圈故障候选",
                "reasoning": [],
                "review_suggestions": [],
            })

    async def exercise():
        task = asyncio.create_task(orchestrator.diagnose(
            build_request(NORMAL, INNER_RACE), llm_client=WaitingModel()
        ))
        try:
            assert await asyncio.to_thread(entered.wait, 5)
            assert not task.done(), "Model wait blocked the event loop until timeout"
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url=BASE_URL
            ) as client:
                response = await asyncio.wait_for(client.get("/health"), timeout=2)
            assert response.status_code == 200
            assert not task.done()
        finally:
            release.set()
            result = await task
        assert result["trace"]["llm_mode"] == "llm"

    asyncio.run(exercise())


def test_app_alias_redirects_to_demo_page(local_backend):
    response = call("GET", "/app", follow_redirects=False)

    assert response.status_code == 307
    assert response.headers["location"] == "/"


def test_list_samples_returns_five_items_without_server_paths(local_backend):
    response = call("GET", "/api/samples")

    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 5
    assert len(body["samples"]) == 5
    assert {item["sample_id"] for item in body["samples"]} == SAMPLE_IDS
    for item in body["samples"]:
        assert item["file"] and item["file"].endswith(".csv")
        assert "path" not in item
        assert (settings.samples_dir / item["file"]).is_file()


def test_diagnose_ok_path_with_builtin_samples(local_backend):
    response = call(
        "POST",
        "/api/diagnose",
        data={
            "sampling_rate": 12000,
            "rotation_speed": 1797,
            "normal_sample": NORMAL,
            "abnormal_sample": INNER_RACE,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["candidates"][0]["fault_type"] == "inner_race_fault"
    assert body["sources"]
    assert body["report"] and body["report_markdown"]
    assert body["trace"]["backend"] == "local"
    assert body["trace"]["tool_calls"] == 4
    assert body["device"]["sensor_position"] == "drive_end"


def test_diagnose_missing_input_returns_400(local_backend):
    response = call("POST", "/api/diagnose", data={"sampling_rate": 12000, "rotation_speed": 1797})

    assert response.status_code == 400
    detail = response.json()["detail"]
    assert detail["code"] == "missing_input"
    assert "normal" in detail["message"] and "abnormal" in detail["message"]


def test_diagnose_unknown_sample_id_returns_400(local_backend):
    response = call(
        "POST",
        "/api/diagnose",
        data={
            "sampling_rate": 12000,
            "rotation_speed": 1797,
            "normal_sample": NORMAL,
            "abnormal_sample": "no_such_sample",
        },
    )

    assert response.status_code == 400
    detail = response.json()["detail"]
    assert detail["code"] == "sample_not_found"
    assert "no_such_sample" in detail["message"]


def test_diagnose_invalid_sampling_rate_returns_422(local_backend):
    response = call(
        "POST",
        "/api/diagnose",
        data={
            "sampling_rate": 0,
            "rotation_speed": 1797,
            "normal_sample": NORMAL,
            "abnormal_sample": INNER_RACE,
        },
    )

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail["code"] == "invalid_argument"
    assert "sampling_rate" in detail["message"]


def test_diagnose_bad_data_returns_200_with_error_status(local_backend):
    response = call(
        "POST",
        "/api/diagnose",
        data={
            "sampling_rate": 12000,
            "rotation_speed": 1797,
            "normal_sample": NORMAL,
            "abnormal_sample": CORRUPTED,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "error"
    assert body["error"]["code"] == "missing_value"
    assert body["trace"]["tool_calls"] == 1
    assert body["report"] is None
    assert body["report_markdown"] is None
    assert body["candidates"] == []


def test_diagnose_rejects_oversized_csv_upload(local_backend):
    original_limit = settings.max_upload_bytes
    object.__setattr__(settings, "max_upload_bytes", 8)
    try:
        response = call(
            "POST",
            "/api/diagnose",
            data={"sampling_rate": 12000, "rotation_speed": 1797, "abnormal_sample": INNER_RACE},
            files={
                "normal_file": ("normal.csv", b"timestamp,amplitude\n0,0.1\n", "text/csv"),
            },
        )
    finally:
        object.__setattr__(settings, "max_upload_bytes", original_limit)

    assert response.status_code == 413
    detail = response.json()["detail"]
    assert detail["code"] == "file_too_large"
    assert "大小限制" in detail["message"]


# ---------------------------------------------------------------------------
# 向量库 / 编排改造后新增的接口契约
# ---------------------------------------------------------------------------
def test_health_reports_retrieval_mode_and_llm_availability(local_backend):
    body = call("GET", "/health").json()

    assert body["retrieval_mode"] in {"hybrid", "bm25"}
    # 当前未配置 MODEL_API_BASE / MODEL_API_KEY，应稳定走纯 BM25、且模型不可用
    assert body["retrieval_mode"] == "bm25"
    assert isinstance(body["llm_available"], bool)
    assert body["llm_available"] is False


def test_diagnose_trace_backend_matches_health(local_backend):
    health = call("GET", "/health").json()

    body = call(
        "POST",
        "/api/diagnose",
        data={
            "sampling_rate": 12000,
            "rotation_speed": 1797,
            "normal_sample": NORMAL,
            "abnormal_sample": INNER_RACE,
        },
    ).json()

    assert health["mcp_backend"] == "local"
    assert body["trace"]["backend"] == health["mcp_backend"]
    assert body["trace"]["retrieval"] == health["retrieval_mode"]


def test_normal_vs_normal_returns_no_sources(local_backend):
    """边界回归：正常样本对正常样本时不得留下任何候选或来源残留。"""
    response = call(
        "POST",
        "/api/diagnose",
        data={
            "sampling_rate": 12000,
            "rotation_speed": 1797,
            "normal_sample": NORMAL,
            "abnormal_sample": NORMAL,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "insufficient_evidence"
    assert body["sources"] == []
    assert body["candidates"] == []
    assert body["report"]["conclusion_sources"] == []
    assert body["report"]["knowledge_candidates"] == []


def test_diagnose_still_works_when_chroma_unavailable(local_backend, monkeypatch, tmp_path):
    """Chroma 索引目录不存在时（注入 embedder 触发混合检索尝试）接口仍应可用。"""
    store = KnowledgeStore(
        knowledge_dir=settings.knowledge_dir,
        chroma_dir=tmp_path / "missing_chroma",
        embedder=fake_embedder,
    )
    assert store.retrieval_mode == "bm25" and store.degraded is True
    monkeypatch.setattr(retriever, "_store", store)

    response = call(
        "POST",
        "/api/diagnose",
        data={
            "sampling_rate": 12000,
            "rotation_speed": 1797,
            "normal_sample": NORMAL,
            "abnormal_sample": INNER_RACE,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["trace"]["retrieval"] == "bm25"
    assert body["candidates"][0]["fault_type"] == "inner_race_fault"
    assert body["report"] and body["report_markdown"]
