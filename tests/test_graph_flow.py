"""LangGraph 编排图测试：节点顺序、条件分支、结构化输出约束与降级语义。

覆盖 app/agent/graph.py 的六个节点与三条条件边：
  validate_node → mcp_analysis_node → rag_retrieval_node → evidence_check_node
      → 证据不足 → report_node
      → 证据充分 → llm_node → report_node

结论一律在 local 后端或纯函数层验证；模型行为用注入的假客户端驱动，不依赖 API Key。
"""

from __future__ import annotations

import asyncio
import json
import types

import pytest

from app.agent import graph, orchestrator
from app.rag import retriever
from tests.helpers import (
    CORRUPTED,
    INNER_RACE,
    NORMAL,
    TimeoutLLMClient,
    backend_mode,
    build_indexed_store,
    build_request,
    diagnose_local,
    exploding_embedder,
)

FULL_FLOW = [
    "validate_node",
    "mcp_analysis_node",
    "rag_retrieval_node",
    "evidence_check_node",
    "llm_node",
    "report_node",
]


class ScriptedLLMClient:
    """假模型客户端：available=True，返回预先给定的 JSON 文本，并记录调用次数。"""

    available = True

    def __init__(self, payload: dict | str):
        self._payload = payload
        self.calls = 0
        self.last_user_prompt = ""

    def chat(self, system: str, user: str) -> str:  # noqa: ARG002 - 只需匹配真实客户端签名
        self.calls += 1
        self.last_user_prompt = user
        if isinstance(self._payload, str):
            return self._payload
        return json.dumps(self._payload, ensure_ascii=False)


# ---------------------------------------------------------------------------
# 1. 正常样本 → 完整图流程（含 llm_node），候选 top1 命中内圈故障
# ---------------------------------------------------------------------------
def test_full_graph_flow_for_inner_race_sample(local_backend):
    result = diagnose_local(NORMAL, INNER_RACE)

    assert result["status"] == "ok"
    assert result["error"] is None
    assert result["trace"]["nodes"] == FULL_FLOW
    assert result["trace"]["tool_calls"] == 4
    assert result["trace"]["retrieval"] == "bm25"
    assert result["candidates"][0]["fault_type"] == "inner_race_fault"
    assert result["report"] and result["report_markdown"]


# ---------------------------------------------------------------------------
# 2. 输入非法 → 在校验节点终止（不调用任何工具、不产出报告）
# ---------------------------------------------------------------------------
def _drop(field: str):
    def mutate(request: dict) -> None:
        request.pop(field, None)

    return mutate


def _set(field: str, value):
    def mutate(request: dict) -> None:
        request[field] = value

    return mutate


INVALID_INPUTS = [
    pytest.param(_drop("sampling_rate"), id="missing-sampling_rate"),
    pytest.param(_drop("rotation_speed"), id="missing-rotation_speed"),
    pytest.param(_set("sampling_rate", 0), id="zero-sampling_rate"),
    pytest.param(_set("rotation_speed", -10), id="negative-rotation_speed"),
    pytest.param(_set("sampling_rate", "12000"), id="non-numeric-sampling_rate"),
    pytest.param(_set("rotation_speed", True), id="bool-rotation_speed"),
    pytest.param(_drop("normal_path"), id="missing-normal_path"),
    pytest.param(_set("abnormal_path", "   "), id="blank-abnormal_path"),
]


@pytest.mark.parametrize("mutate", INVALID_INPUTS)
def test_invalid_request_stops_at_validate_node(local_backend, mutate):
    request = build_request(NORMAL, INNER_RACE)
    mutate(request)

    with backend_mode("local"):
        result = asyncio.run(orchestrator.diagnose(request))

    assert result["status"] == "error"
    assert result["error"]["code"] == "invalid_argument"
    assert "输入校验未通过" in result["error"]["message"]
    assert result["trace"]["nodes"] == ["validate_node"]
    assert result["trace"]["tool_calls"] == 0
    assert result["trace"]["llm_calls"] == 0
    assert result["report"] is None and result["report_markdown"] is None
    assert result["candidates"] == [] and result["sources"] == []


# ---------------------------------------------------------------------------
# 3. 正常 vs 正常 → 证据不足，跳过模型节点
# ---------------------------------------------------------------------------
def test_normal_vs_normal_skips_llm_node(local_backend):
    result = diagnose_local(NORMAL, NORMAL)

    assert result["status"] == "insufficient_evidence"
    assert result["trace"]["nodes"] == [
        "validate_node",
        "mcp_analysis_node",
        "rag_retrieval_node",
        "evidence_check_node",
        "report_node",
    ]
    assert "llm_node" not in result["trace"]["nodes"]
    assert result["trace"]["llm_calls"] == 0
    assert result["candidates"] == [] and result["sources"] == []


# ---------------------------------------------------------------------------
# 4. RAG 无候选 → 不调用模型
# ---------------------------------------------------------------------------
def test_empty_candidates_never_call_model(local_backend, monkeypatch):
    # 阈值调到 1.0 以上，模拟知识库全部落在阈值之外
    monkeypatch.setattr(
        retriever, "settings", types.SimpleNamespace(rag_top_k=5, rag_score_threshold=0.99)
    )

    result = diagnose_local(NORMAL, INNER_RACE)

    assert result["status"] == "insufficient_evidence"
    assert result["candidates"] == []
    assert result["trace"]["llm_calls"] == 0
    assert result["trace"]["llm_mode"] == "template"
    assert "llm_node" not in result["trace"]["nodes"]
    assert any("证据不足" in w for w in result["trace"]["warnings"])


# ---------------------------------------------------------------------------
# 5. 工具失败 → 结构化错误，流程在分析节点终止
# ---------------------------------------------------------------------------
def test_corrupted_sample_returns_structured_error(local_backend):
    result = diagnose_local(NORMAL, CORRUPTED)

    assert result["status"] == "error"
    assert result["error"]["code"] == "missing_value"
    assert result["trace"]["tool_calls"] == 1
    assert result["trace"]["nodes"] == ["validate_node", "mcp_analysis_node"]
    assert result["report"] is None and result["report_markdown"] is None
    assert result["rag"] is None and result["candidates"] == []


# ---------------------------------------------------------------------------
# 6. 模型超时 → 模板报告，但候选与证据全部保留
# ---------------------------------------------------------------------------
def test_llm_timeout_keeps_candidates_and_evidence(local_backend):
    result = diagnose_local(NORMAL, INNER_RACE, llm_client=TimeoutLLMClient())

    assert result["status"] == "ok"
    trace = result["trace"]
    assert trace["nodes"] == FULL_FLOW
    assert trace["llm_mode"] == "template_fallback"
    assert trace["llm_calls"] == 1
    assert trace["llm_error"]["code"] == "timeout"
    assert any("模型调用失败" in w for w in trace["warnings"])

    # 已算出的证据与候选不受降级影响
    assert result["candidates"] and result["candidates"][0]["fault_type"] == "inner_race_fault"
    assert result["comparison"] and result["rag"]
    assert result["report"] and result["report_markdown"]
    assert result["report"]["knowledge_candidates"]
    assert result["report"]["conclusion"].startswith("最可能故障类型：")


# ---------------------------------------------------------------------------
# 7. 模型返回候选外故障类型 → 降级模板，且结论绝不采用该类型
# ---------------------------------------------------------------------------
def test_llm_out_of_candidate_fault_type_is_rejected(local_backend):
    client = ScriptedLLMClient({
        "selected_fault_type": "gear_fault",
        "summary": "判定为齿轮故障。",
        "reasoning": ["齿轮啮合频率异常"],
        "review_suggestions": ["检查齿轮箱"],
    })

    result = diagnose_local(NORMAL, INNER_RACE, llm_client=client)

    assert client.calls == 1
    assert result["status"] == "ok"
    assert result["trace"]["llm_mode"] == "template_fallback"
    assert any("候选外的故障类型" in w for w in result["trace"]["warnings"])

    report = result["report"]
    assert "gear_fault" not in report["conclusion"]
    assert "齿轮故障" not in report["conclusion"]
    assert "gear_fault" not in (result["report_markdown"] or "")
    assert "齿轮啮合频率异常" not in report["inferences"]
    # 降级后仍用模板结论（候选 top1）
    assert report["conclusion"].startswith("最可能故障类型：")
    assert result["candidates"][0]["fault_type"] == "inner_race_fault"


# ---------------------------------------------------------------------------
# 8. 模型返回合法结构化输出 → llm 模式，且选中类型落在候选内
# ---------------------------------------------------------------------------
def test_llm_valid_structured_output_is_accepted(local_backend):
    summary = "依据 BPFI 族谱峰与边带，判断为内圈故障。"
    client = ScriptedLLMClient({
        "selected_fault_type": "inner_race_fault",
        "summary": summary,
        "reasoning": ["BPFI 及其谐波处出现成组峰值", "峰侧存在转频间隔边带"],
        "review_suggestions": ["按实际转速复核 BPFI 偏差"],
    })

    result = diagnose_local(NORMAL, INNER_RACE, llm_client=client)

    assert client.calls == 1
    assert result["trace"]["llm_mode"] == "llm"
    assert result["trace"]["llm_calls"] == 1
    # 提示词里明确给出了允许的取值集合
    assert "inner_race_fault" in client.last_user_prompt
    assert "unconfirmed" in client.last_user_prompt

    report = result["report"]
    assert report["conclusion"] == summary
    assert any("BPFI 及其谐波处出现成组峰值" in item for item in report["inferences"])
    assert any("按实际转速复核 BPFI 偏差" in item for item in report["review_suggestions"])
    # 选中类型取自已检索候选
    candidates = {c["fault_type"] for c in result["candidates"]}
    assert "inner_race_fault" in candidates
    assert report["conclusion_sources"] and report["knowledge_candidates"]


# ---------------------------------------------------------------------------
# 9. 证书级集成：LangGraph 层与真实检索层的降级契约对齐
# ---------------------------------------------------------------------------
def test_graph_layer_honours_real_retriever_degradation(monkeypatch, tmp_path, local_backend):
    """用真实 app/rag/retriever.py + 抛错的 embedder（Chroma 指向 tmp_path）验证契约。

    两个改造分别由不同人完成：这里让向量分支在查询时失败，断言编排层确实读到了
    retriever 返回的 degraded/degraded_reason，并把降级原因写进 trace。
    """
    chroma_dir = tmp_path / "chroma"
    # 索引用可用的假 embedder 建好，随后打开索引的 store 使用「查询即失败」的 embedder
    store, summary = build_indexed_store(chroma_dir, store_embedder=exploding_embedder)
    assert summary["count"] > 0
    assert store.retrieval_mode == "hybrid"  # 索引可打开，向量分支尚未被调用

    monkeypatch.setattr(retriever, "_store", store)

    result = diagnose_local(NORMAL, INNER_RACE)

    trace = result["trace"]
    assert trace["retrieval"] == "bm25"
    assert store.retrieval_mode == "bm25" and store.degraded is True
    degradation = [w for w in trace["warnings"] if "知识检索发生降级" in w]
    assert degradation, f"trace 缺少降级告警：{trace['warnings']}"
    assert "已降级 BM25" in degradation[0]
    assert store.degraded_reason in degradation[0]

    # 降级不阻断诊断：BM25 分支照常给出候选
    assert result["status"] == "ok"
    assert result["candidates"][0]["fault_type"] == "inner_race_fault"
    # 编排层投影后的候选只留结论字段，原始检索结果在 rag 视图里
    assert result["rag"]["retrieval"] == "bm25"
    assert all(item["retrieval"] == "bm25" for item in result["rag"]["candidates"])
    assert result["report"]["knowledge_candidates"]


# ---------------------------------------------------------------------------
# 10. 并列族打破规则：只有首位候选声明的特征频率族落在并列族内才采信
# ---------------------------------------------------------------------------
def _tied_comparison() -> dict:
    return {
        "dominant_characteristic": {
            "label": "BPFI",
            "ambiguous": True,
            "margin_ratio": 1.0706,
            "tied_candidates": [{"label": "BPFI"}, {"label": "BSF"}],
        }
    }


def test_tied_family_resolution_accepts_candidate_in_tied_families():
    candidates = [{"fault_type": "ball_fault", "frequency_family": "BSF"}]
    resolution = graph._tied_family_resolution(candidates, _tied_comparison())
    assert resolution["fault_type"] == "ball_fault"
    assert resolution["frequency_family"] == "BSF"
    assert resolution["tied_labels"] == ["BPFI", "BSF"]


@pytest.mark.parametrize(
    "candidates",
    [
        [{"fault_type": "outer_race_fault", "frequency_family": "BPFO"}],  # 族不在并列族内
        [{"fault_type": "ball_fault", "frequency_family": None}],  # 条目未声明族，无法校验
        [],  # 无候选
    ],
)
def test_tied_family_resolution_rejects_unmatched_candidates(candidates):
    assert graph._tied_family_resolution(candidates, _tied_comparison()) is None


def test_tied_family_resolution_ignores_non_ambiguous_comparison():
    candidates = [{"fault_type": "ball_fault", "frequency_family": "BSF"}]
    comparison = {"dominant_characteristic": {"label": "BPFI", "ambiguous": False}}
    assert graph._tied_family_resolution(candidates, comparison) is None


# ---------------------------------------------------------------------------
# 图结构本身：节点名与条件边与文档一致（纯结构断言，不跑流程）
# ---------------------------------------------------------------------------
def test_graph_exposes_expected_nodes():
    nodes = set(graph.GRAPH.get_graph().nodes)
    assert {
        "validate_node",
        "mcp_analysis_node",
        "rag_retrieval_node",
        "evidence_check_node",
        "llm_node",
        "report_node",
    } <= nodes