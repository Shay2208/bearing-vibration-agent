"""SubTask 9.2：MCP 错误信封、RAG 空结果、报告字段完整性。

其中 MCP 错误结构一致性测试会真实启动 mcp:stdio 子进程（标记 stdio），
其余测试都在 local 后端或纯函数层完成。
"""

from __future__ import annotations

import asyncio
import re

import pytest

from app.agent import report as report_module
from app.mcp import adapter, vibration
from app.rag import retriever
from tests.helpers import REPORT_KEYS, backend_mode, sample_path

DEVICE = {
    "device_type": "bearing",
    "sensor_position": "drive_end",
    "rotation_speed": 1797.0,
    "sampling_rate": 12000.0,
    "normal_file": "normal_1797.csv",
    "abnormal_file": "inner_race_1797.csv",
}


# ---------------------------------------------------------------------------
# 7. MCP 错误返回：ok=false、error.code/message 可读，local 与 mcp:stdio 结构一致
# ---------------------------------------------------------------------------
def _normalize_numbers(text: str) -> str:
    """把消息里的数字字面量归一，用于跨后端比较。

    MCP 工具签名声明 sampling_rate: float，int 0 经协议边界会被强制成 0.0，
    因此两个后端的 message 只在被回显的非法值写法上不同（0 与 0.0），
    error 的 code / field / 结构完全一致。
    """
    return re.sub(r"\d+(?:\.\d+)?", "#", text)


def test_bad_sampling_rate_error_envelope_is_identical_across_backends():
    csv_path = sample_path("normal_1797")

    with backend_mode("local"):
        local_envelope = asyncio.run(adapter.extract_vibration_features(csv_path, 0))
    with backend_mode("stdio"):
        stdio_envelope = asyncio.run(adapter.extract_vibration_features(csv_path, 0))

    for envelope, backend in ((local_envelope, "local"), (stdio_envelope, "mcp:stdio")):
        assert envelope["ok"] is False
        assert envelope["backend"] == backend
        assert envelope["tool"] == "extract_vibration_features"
        assert set(envelope["error"]) == {"code", "message", "field"}
        assert envelope["error"]["code"] == vibration.ERROR_INVALID_ARGUMENT
        assert envelope["error"]["field"] == "sampling_rate"
        assert "采样率必须是有限正数" in envelope["error"]["message"]
        assert envelope["error"]["message"].strip() != ""

    # 两个后端错误的键集合、code、field 完全一致；message 仅在被回显数值的写法上不同
    assert set(local_envelope["error"]) == set(stdio_envelope["error"])
    assert local_envelope["error"]["code"] == stdio_envelope["error"]["code"]
    assert local_envelope["error"]["field"] == stdio_envelope["error"]["field"]
    assert _normalize_numbers(local_envelope["error"]["message"]) == _normalize_numbers(
        stdio_envelope["error"]["message"]
    )
    # 唯一的差别就是被回显非法值的写法：local 原样回显 int 0，stdio 侧被强转成 float 0.0
    prefix = "采样率必须是有限正数，收到："
    assert local_envelope["error"]["message"] == prefix + "0"
    assert stdio_envelope["error"]["message"] == prefix + "0.0"


def test_missing_file_error_envelope_is_identical_across_backends(tmp_path):
    missing = str((tmp_path / "not_exist.csv").resolve())

    with backend_mode("local"):
        local_envelope = asyncio.run(adapter.extract_vibration_features(missing, 12000))
    with backend_mode("stdio"):
        stdio_envelope = asyncio.run(adapter.extract_vibration_features(missing, 12000))

    for envelope in (local_envelope, stdio_envelope):
        assert envelope["ok"] is False
        assert envelope["error"]["code"] == vibration.ERROR_FILE_NOT_FOUND
        assert envelope["error"]["field"] == "path"
        assert "信号文件不存在" in envelope["error"]["message"]

    assert local_envelope["error"] == stdio_envelope["error"]


def test_compare_missing_field_error_is_structured():
    """缺字段的特征字典应返回 missing_value，而不是抛异常穿透。"""
    complete = vibration.extract_vibration_features(sample_path("normal_1797"), 12000)
    incomplete = {key: value for key, value in complete.items() if key != "rms"}

    with backend_mode("local"):
        envelope = asyncio.run(adapter.compare_normal_abnormal(complete, incomplete, 1797))

    assert envelope["ok"] is False
    assert envelope["backend"] == "local"
    assert envelope["error"]["code"] == vibration.ERROR_MISSING_VALUE
    assert envelope["error"]["field"] == "rms"
    assert "rms" in envelope["error"]["message"]


@pytest.mark.stdio
def test_stdio_backend_success_matches_local_backend():
    """至少保留一个真实走 mcp:stdio 后端的成功路径测试。"""
    csv_path = sample_path("normal_1797")

    with backend_mode("local"):
        local_envelope = asyncio.run(adapter.extract_vibration_features(csv_path, 12000, rotation_speed=1797))
    with backend_mode("stdio"):
        stdio_envelope = asyncio.run(adapter.extract_vibration_features(csv_path, 12000, rotation_speed=1797))

    assert local_envelope["ok"] is True and stdio_envelope["ok"] is True
    assert local_envelope["backend"] == "local"
    assert stdio_envelope["backend"] == "mcp:stdio"

    local_data = local_envelope["data"]
    stdio_data = stdio_envelope["data"]
    for key in ("rms", "peak", "kurtosis", "crest_factor", "dominant_frequency", "dominant_amplitude"):
        assert stdio_data[key] == pytest.approx(local_data[key], rel=1e-9)
    assert stdio_data["band_energy"] == local_data["band_energy"]
    assert stdio_data["spectrum_peaks"] == local_data["spectrum_peaks"]
    assert stdio_data["envelope"]["characteristic_energy"] == local_data["envelope"]["characteristic_energy"]


# ---------------------------------------------------------------------------
# 8. RAG 空结果处理
# ---------------------------------------------------------------------------
def test_rag_unrelated_query_returns_no_candidate():
    rag = retriever.retrieve_candidates("完全无关的查询：今天天气晴朗，适合出行与拍照，还有火锅推荐")

    assert rag["insufficient_evidence"] is True
    assert rag["candidates"] == []
    assert rag["sources"] == []
    assert rag["retrieval"] == "bm25"


def test_rag_high_threshold_returns_no_candidate():
    rag = retriever.retrieve_candidates("轴承 内圈 故障 BPFI 谱峰 峭度 冲击", threshold=0.99)

    assert rag["insufficient_evidence"] is True
    assert rag["candidates"] == []
    assert rag["sources"] == []


def test_rag_relevant_query_still_hits_candidate():
    """反例：相关信息确实能命中，说明上面的空结果不是「永远为空」。"""
    rag = retriever.retrieve_candidates("轴承 内圈 故障 BPFI 谱峰 峭度 冲击")

    assert rag["candidates"], "内圈相关查询应至少命中一条候选"
    assert rag["candidates"][0]["fault_type"] == "inner_race_fault"
    assert rag["candidates"][0]["score"] >= 0.35
    assert rag["insufficient_evidence"] is False
    assert rag["sources"][0]["name"] and rag["sources"][0]["url"] and rag["sources"][0]["locator"]


def _build_real_inputs():
    normal_features = vibration.extract_vibration_features(sample_path("normal_1797"), 12000, rotation_speed=1797)
    abnormal_features = vibration.extract_vibration_features(sample_path("inner_race_1797"), 12000, rotation_speed=1797)
    comparison = vibration.compare_normal_abnormal(normal_features, abnormal_features, 1797)
    data_quality = vibration.validate_vibration_data(
        sample_path("normal_1797"), sample_path("inner_race_1797"), 12000, "drive_end"
    )
    query = retriever.build_query(DEVICE, comparison, abnormal_features)
    return data_quality, comparison, retriever.retrieve_candidates(query)


def test_rag_empty_result_marks_insufficient_evidence_without_guessing():
    """检索为空时只标记证据不足，绝不返回猜测的故障类型。"""
    data_quality, comparison, rag = _build_real_inputs()
    assert rag["insufficient_evidence"] is False  # 前提：默认阈值下内圈样本能命中

    strict_rag = retriever.retrieve_candidates(rag["query"], threshold=0.99)
    assert strict_rag["insufficient_evidence"] is True

    report = report_module.build_report(
        device=DEVICE, data_quality=data_quality, comparison=comparison, rag=strict_rag, candidates=[]
    )

    assert report["knowledge_candidates"] == []
    assert report["conclusion_sources"] == []
    assert report["confidence"]["level"] == "low"
    assert report["conclusion"].startswith("无法确认")
    assert "inner_race_fault" not in report["conclusion"]
    assert report["knowledge_basis"] == ["检索阈值内未命中任何知识条目，本报告不做故障类型判断。"]


# ---------------------------------------------------------------------------
# 9. 报告字段完整性
# ---------------------------------------------------------------------------
def test_report_fields_and_markdown_are_complete():
    data_quality, comparison, rag = _build_real_inputs()
    candidates = rag["candidates"]
    assert candidates, "内圈样本应命中候选，否则本测试前提不成立"

    report = report_module.build_report(
        device=DEVICE, data_quality=data_quality, comparison=comparison, rag=rag, candidates=candidates
    )
    markdown = report_module.render_markdown(
        report,
        {
            "device": DEVICE,
            "data_quality": data_quality,
            "comparison": comparison,
            "rag": rag,
            "candidates": candidates,
            "trace": {"backend": "local", "tool_calls": 4, "max_tool_calls": 8},
        },
    )

    # 字段齐全
    assert set(REPORT_KEYS) <= set(report)
    assert all(report[key] is not None for key in REPORT_KEYS)

    # 各小节内容非空
    assert report["analysis_target"].strip()
    assert report["data_quality_summary"].strip()
    assert len(report["feature_comparison"]) == 6
    for row in report["feature_comparison"]:
        assert set(row) == {"feature", "normal", "abnormal", "delta", "ratio", "note"}
    assert report["band_energy_comparison"]
    assert report["spectrum_notes"].strip()
    assert report["knowledge_candidates"] and report["knowledge_candidates"][0]["source"]
    assert report["conclusion"].startswith("最可能故障类型：")
    assert report["conclusion_evidence"]
    assert report["conclusion_sources"]
    assert {"source", "url", "locator"} <= set(report["conclusion_sources"][0])
    assert report["confidence"]["level"] in {"low", "medium", "high"}
    assert report["confidence"]["reason"].strip()
    assert report["data_facts"] and report["knowledge_basis"] and report["inferences"]
    assert report["review_suggestions"]
    assert report["prototype_limits"]

    # markdown：11 个小节标题齐全且整体非空
    assert markdown.strip()
    assert len(markdown) > 1000
    for section in report_module.MARKDOWN_SECTIONS:
        assert section in markdown
    assert "滚动轴承内圈故障" in markdown
    assert "CWRU Bearing Data Center" in markdown