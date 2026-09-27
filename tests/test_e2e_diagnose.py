"""SubTask 9.3：端到端测试 —— 调用真实 orchestrator.diagnose()（local 后端）。

7 个场景：正常对正常、正常对内圈、正常对外圈、正常对滚动体（并列族经知识库候选打破）、
数据损坏、无法匹配知识库、模型 API 超时。

真实标签（CWRU condition）只用于离线评测断言，不进入在线诊断链路。
"""

from __future__ import annotations

import types

import pytest

from app.agent import report as report_module
from app.rag import retriever
from tests.helpers import (
    BALL,
    CORRUPTED,
    INNER_RACE,
    NORMAL,
    OUTER_RACE,
    REPORT_KEYS,
    TimeoutLLMClient,
    diagnose_local,
)

# ---------------------------------------------------------------------------
# 1. 正常对正常 → 证据不足（不得基于高分候选强行下结论）
# ---------------------------------------------------------------------------
def test_normal_vs_normal_is_insufficient_evidence(local_backend):
    result = diagnose_local(NORMAL, NORMAL)

    assert result["status"] == "insufficient_evidence"
    assert result["error"] is None
    assert result["candidates"] == []
    assert result["data_quality"]["valid"] is True
    assert result["report"]["knowledge_candidates"] == []
    assert result["report"]["conclusion"].startswith("无法确认")
    assert result["trace"]["tool_calls"] == 4
    assert result["trace"]["llm_calls"] == 0
    assert result["trace"]["llm_mode"] == "template"
    assert any("未观察到任何显著变化" in warning for warning in result["trace"]["warnings"])
    # 报告仍完整可输出
    assert set(REPORT_KEYS) <= set(result["report"])
    assert result["report_markdown"] and "无法确认" in result["report_markdown"]


# ---------------------------------------------------------------------------
# 2. 正常 vs 内圈 → 候选 top1 = inner_race_fault（与 CWRU 标签一致）
# ---------------------------------------------------------------------------
def test_normal_vs_inner_race_top_candidate(local_backend):
    result = diagnose_local(NORMAL, INNER_RACE)

    assert result["status"] == "ok"
    assert result["error"] is None
    top = result["candidates"][0]
    assert top["fault_type"] == "inner_race_fault"
    assert top["score"] >= 0.35
    assert result["trace"]["tool_calls"] == 4
    assert result["trace"]["retrieval"] == "bm25"
    assert result["trace"]["llm_mode"] == "template"

    # 来源引用：sources 非空，且报告里出现来源名
    assert result["sources"]
    assert all({"name", "url", "locator", "fault_type"} <= set(item) for item in result["sources"])
    assert result["report"]["conclusion_sources"]
    assert "CWRU Bearing Data Center" in result["report_markdown"]
    assert "滚动轴承内圈故障" in result["report_markdown"]


# ---------------------------------------------------------------------------
# 3. 正常 vs 外圈 → 候选 top1 = outer_race_fault（与 CWRU 标签一致）
# ---------------------------------------------------------------------------
def test_normal_vs_outer_race_top_candidate(local_backend):
    result = diagnose_local(NORMAL, OUTER_RACE)

    assert result["status"] == "ok"
    top = result["candidates"][0]
    assert top["fault_type"] == "outer_race_fault"
    assert top["score"] >= 0.35
    assert result["trace"]["tool_calls"] == 4
    assert result["sources"] and result["report"]["conclusion_sources"]
    assert result["comparison"]["dominant_characteristic"]["label"] == "BPFO"
    assert "滚动轴承外圈故障" in result["report_markdown"]


# ---------------------------------------------------------------------------
# 4. 正常 vs 滚动体 → 主导特征频率族并列，由检索首位候选打破并列（置信下调为 low）
# ---------------------------------------------------------------------------
def test_normal_vs_ball_fault_tie_broken_by_top_candidate(local_backend):
    """真实标签 ball_fault：BPFI 与 BSF 包络能量仅差约 7%，不足以判定主次。

    量级余量判据命中（并列）时本不给排他性结论；但检索首位候选 ball_fault 声明的
    特征频率族 BSF 落在并列族内（知识库条目自带 frequency_family 元数据），
    该候选与并列现象自洽，据此采信该候选并把置信度下调为 low。
    """
    result = diagnose_local(NORMAL, BALL)

    assert result["status"] == "ok"
    assert result["error"] is None

    comparison = result["comparison"]
    dominant = comparison["dominant_characteristic"]
    assert dominant["label"] == "BPFI"
    assert dominant["ambiguous"] is True
    assert dominant["margin_ratio"] == pytest.approx(1.0706, abs=1e-3)
    assert [(item["label"], item["frequency_hz"]) for item in dominant["tied_candidates"]] == [
        ("BPFI", pytest.approx(162.186, abs=1e-3)),
        ("BSF", pytest.approx(70.5838, abs=1e-3)),
    ]

    energy = comparison["characteristic_energy_changes"]
    bpfi_energy = energy["bpfi"]["abnormal"]
    bsf_energy = energy["bsf"]["abnormal"]
    assert bpfi_energy == pytest.approx(4.20470e-4, rel=1e-6)
    assert bsf_energy == pytest.approx(3.92749e-4, rel=1e-6)
    # 差异只有约 7%，属于噪声级差距：不宣称 BPFI 最突出，只按并列族匹配采信首选候选
    assert bpfi_energy / bsf_energy == pytest.approx(1.0706, abs=1e-3)

    # 检索侧：知识库判据表达 + 检索排序修正后，首位候选已是 ball_fault 并透出特征频率族
    top = result["candidates"][0]
    assert top["fault_type"] == "ball_fault"
    assert top["frequency_family"] == "BSF"
    assert "滚动轴承滚动体" in result["report"]["conclusion"]

    # 并列采信：置信度下调为 low，trace 记录并列族与采信依据
    assert result["report"]["confidence"]["level"] == "low"
    resolution = result["trace"]["tied_resolution"]
    assert resolution["fault_type"] == "ball_fault"
    assert resolution["frequency_family"] == "BSF"
    assert resolution["tied_labels"] == ["BPFI", "BSF"]
    assert any("并列" in warning and "采信" in warning for warning in result["trace"]["warnings"])
    assert any("能量并列" in item for item in result["report"]["review_suggestions"])
    assert "采信" in result["report_markdown"]
    assert result["trace"]["llm_calls"] == 0
    assert result["trace"]["llm_mode"] == "template"


# ---------------------------------------------------------------------------
# 5. 数据损坏 → error / missing_value / 只调用 1 次工具 / 不产出报告
# ---------------------------------------------------------------------------
def test_corrupted_sample_stops_with_missing_value_error(local_backend):
    result = diagnose_local(NORMAL, CORRUPTED)

    assert result["status"] == "error"
    assert result["error"]["code"] == "missing_value"
    assert "第 19 行 amplitude 为空值" in result["error"]["message"]
    assert result["trace"]["tool_calls"] == 1
    assert result["data_quality"]["valid"] is False
    assert result["data_quality"]["comparable"] is False
    # 流程在特征提取前停止
    assert result["features"] is None
    assert result["comparison"] is None
    assert result["rag"] is None
    assert result["report"] is None
    assert result["report_markdown"] is None
    assert result["candidates"] == [] and result["sources"] == []
    assert result["trace"]["llm_calls"] == 0


# ---------------------------------------------------------------------------
# 6. 无法匹配知识库 → 证据不足且候选为空
# ---------------------------------------------------------------------------
def test_knowledge_base_miss_is_insufficient_evidence(local_backend, monkeypatch):
    # 调高检索阈值，模拟「知识库中没有可用条目」
    monkeypatch.setattr(retriever, "settings", types.SimpleNamespace(rag_top_k=5, rag_score_threshold=0.99))

    result = diagnose_local(NORMAL, INNER_RACE)

    assert result["status"] == "insufficient_evidence"
    assert result["candidates"] == []
    assert result["rag"]["insufficient_evidence"] is True
    assert result["rag"]["candidates"] == []
    assert result["sources"] == []
    assert result["report"]["knowledge_candidates"] == []
    assert result["report"]["conclusion_sources"] == []
    assert any("证据不足" in warning for warning in result["trace"]["warnings"])
    assert result["trace"]["llm_calls"] == 0
    assert result["trace"]["llm_mode"] == "template"
    assert result["trace"]["retrieval"] == "bm25"


# ---------------------------------------------------------------------------
# 7. 模型 API 超时 → 降级为模板报告，报告字段仍完整
# ---------------------------------------------------------------------------
def test_llm_timeout_falls_back_to_template_report(local_backend):
    result = diagnose_local(NORMAL, INNER_RACE, llm_client=TimeoutLLMClient())

    assert result["status"] == "ok"
    trace = result["trace"]
    assert trace["llm_mode"] == "template_fallback"
    assert trace["llm_calls"] == 1
    assert trace["llm_error"] == {"code": "timeout", "message": "模拟模型调用超时"}
    assert any("模型调用失败（timeout）" in warning for warning in trace["warnings"])

    report = result["report"]
    assert set(REPORT_KEYS) <= set(report)
    assert all(report[key] is not None for key in REPORT_KEYS)
    assert report["conclusion"].startswith("最可能故障类型：")
    assert report["knowledge_candidates"] and report["conclusion_sources"]

    markdown = result["report_markdown"]
    assert markdown and len(markdown) > 1000
    for section in report_module.MARKDOWN_SECTIONS:
        assert section in markdown
    # 降级只影响措辞，不影响候选与来源
    assert result["candidates"][0]["fault_type"] == "inner_race_fault"