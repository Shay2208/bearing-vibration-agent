"""条款级判据核对：知识条款词表、核对三态、判据因子与候选重排、报告与 trace 的条款证据。

覆盖三层：
  - app/rag/criteria.py ：确定性核对引擎（三态 / 因子 / 重排），纯函数、不依赖数据文件；
  - app/rag/store.py    ：条款解析与条款级证据（search 结果新增 criteria / clause_hits）；
  - app/agent/graph.py + report.py：条款证据进入 trace、提示词与报告的链路。

场景断言用的是仓库内置 CWRU 样例（内圈 / 外圈 / 滚动体），
因而不依赖任何模型与网络：判据核对完全由规则引擎给出。
"""

from __future__ import annotations

import pytest

from app.agent import graph, report as report_module
from app.config import settings
from app.mcp import vibration
from app.rag import criteria as engine
from app.rag import retriever
from app.rag.store import KnowledgeStore, load_knowledge
from tests.helpers import BALL, DEFAULT_SAMPLING_RATE, INNER_RACE, NORMAL, OUTER_RACE, diagnose_local, sample_path

DEVICE = {
    "device_type": "bearing",
    "sensor_position": "drive_end",
    "rotation_speed": 1797.0,
    "sampling_rate": DEFAULT_SAMPLING_RATE,
}
ROTATION_SPEED = 1797.0


# ---------------------------------------------------------------------------
# 公共构造：真实样例 → 对比事实 → 候选（含条款核对结果）
# ---------------------------------------------------------------------------
def _analyze(abnormal_stem: str) -> tuple[dict, dict]:
    normal = vibration.extract_vibration_features(sample_path(NORMAL), DEFAULT_SAMPLING_RATE, rotation_speed=ROTATION_SPEED)
    abnormal = vibration.extract_vibration_features(
        sample_path(abnormal_stem), DEFAULT_SAMPLING_RATE, rotation_speed=ROTATION_SPEED
    )
    comparison = vibration.compare_normal_abnormal(normal, abnormal, ROTATION_SPEED)
    query = retriever.build_query(DEVICE, comparison, abnormal)
    rag = retriever.retrieve_candidates(query, comparison=comparison, features=abnormal)
    return comparison, rag["candidates"]


def _ctx(**overrides) -> dict:
    """构造一份可核对事实：默认值让多数条款成立，个别用 overrides 改。"""
    comparison = {
        "feature_changes": {
            "rms": {"ratio": 2.0},
            "kurtosis": {"ratio": 1.1},
            "crest_factor": {"ratio": 1.05},
        },
        "changed_features": ["rms"],
        "frequency_change": {"shift_hz": 0.0, "abnormal_dominant": 30.0},
        "fault_frequencies": {"shaft_frequency": 30.0},
        "fault_frequency_matches": [
            {"label": "BPFI", "harmonic_order": 1},
            {"label": "BPFI", "harmonic_order": 2},
        ],
        "characteristic_energy_changes": {
            "bpfi": {"abnormal": 0.02, "ratio": 3.0},
            "bsf": {"abnormal": 0.01, "ratio": 2.0},
        },
        "dominant_characteristic": {"label": "BPFI", "ambiguous": False, "margin_ratio": 2.0},
    }
    comparison.update(overrides)
    return {"comparison": comparison, "features": {"dominant_frequency": 30.0}}


# ---------------------------------------------------------------------------
# 1. 核对引擎：三态、词表守卫、因子与状态边界
# ---------------------------------------------------------------------------
def test_check_criterion_returns_three_states():
    ctx = _ctx()
    hit = engine.check_criterion(
        {"id": "c1", "metric": "rms_ratio", "op": ">=", "value": "1.3", "claim": "RMS 上升"}, ctx
    )
    miss = engine.check_criterion(
        {"id": "c2", "metric": "changed_feature_count", "op": "<=", "value": "0", "claim": "无变化"}, ctx
    )
    unknown = engine.check_criterion(
        {"id": "c3", "metric": "tied_family", "family": "BSF", "op": "==", "value": "true", "claim": "并列含 BSF"}, ctx
    )

    assert (hit["state"], hit["actual"]) == ("hit", 2.0)
    assert (miss["state"], miss["actual"]) == ("miss", 1.0)
    # 未并列时 tied_family 不可核对：记 unknown，不算「不成立」
    assert (unknown["state"], unknown["actual"]) == ("unknown", None)
    assert hit["expect"] == "rms_ratio >= 1.3"
    assert engine.check_criterion({"id": "c4", "metric": "rms_ratio", "op": ">=", "value": "1.3", "family": "BPFI"}, ctx)[
        "expect"
    ] == "rms_ratio(BPFI) >= 1.3"


def test_check_criterion_rejects_metric_or_op_outside_vocabulary():
    bad_metric = engine.check_criterion({"id": "x1", "metric": "magic_score", "op": ">=", "value": "1"}, _ctx())
    bad_op = engine.check_criterion({"id": "x2", "metric": "rms_ratio", "op": "≈", "value": "1.3"}, _ctx())

    assert bad_metric["state"] == "unknown"
    assert bad_op["state"] == "unknown"


def test_boolean_declaration_words_are_accepted():
    for word in ("true", "是", "1"):
        row = engine.check_criterion(
            {"id": "b1", "metric": "dominant_ambiguous", "op": "==", "value": word},
            _ctx(dominant_characteristic={"label": "BPFI", "ambiguous": True, "margin_ratio": 1.05}),
        )
        assert row["state"] == "hit", word


def test_factor_and_state_boundaries():
    hits = [{"id": "h1", "metric": "rms_ratio", "op": ">=", "value": "1.3"}]
    misses = [{"id": "m1", "metric": "rms_ratio", "op": ">=", "value": "99"}]
    unknowns = [{"id": "u1", "metric": "envelope_rank", "family": "FTF", "op": "<=", "value": "2"}]
    ctx = _ctx()

    empty = engine.evaluate([], ctx)
    assert (empty["state"], empty["factor"]) == ("none", 1.0)

    all_hit = engine.evaluate(hits * 2, ctx)
    assert (all_hit["state"], all_hit["factor"], all_hit["checked"]) == ("consistent", 1.0, 2)

    all_miss = engine.evaluate(misses * 2, ctx)
    assert (all_miss["state"], all_miss["factor"]) == ("conflict", 0.4)

    tied = engine.evaluate(hits + misses, ctx)
    assert (tied["state"], tied["factor"]) == ("conflict", 0.7)

    partial = engine.evaluate(hits * 2 + misses, ctx)
    assert (partial["state"], partial["factor"]) == ("partial", 0.8)

    unverified = engine.evaluate(unknowns, ctx)
    # 没有可核对指标时不罚分，只标记未核对
    assert (unverified["state"], unverified["factor"], unverified["checked"]) == ("unverified", 1.0, 0)


def test_evaluate_candidates_keeps_order_without_comparison():
    candidates = [{"fault_type": "a", "score": 0.4}, {"fault_type": "b", "score": 0.5}]
    assert engine.evaluate_candidates(candidates, None) == candidates
    assert engine.evaluate_candidates(candidates, {}) == candidates


def test_evaluate_candidates_sinks_entry_contradicted_by_data():
    """检索分高但判据与数据矛盾的条目应被 alignment_score 压到后面。"""
    hit_spec = [{"id": "h", "metric": "rms_ratio", "op": ">=", "value": "1.3"}]
    miss_spec = [{"id": "m", "metric": "rms_ratio", "op": ">=", "value": "99"}]
    candidates = [
        {"fault_type": "high_score_but_conflict", "score": 0.6283, "criteria": miss_spec},
        {"fault_type": "lower_score_but_consistent", "score": 0.4453, "criteria": hit_spec},
    ]

    ranked = engine.evaluate_candidates(candidates, _ctx()["comparison"], _ctx()["features"])

    assert [item["fault_type"] for item in ranked] == ["lower_score_but_consistent", "high_score_but_conflict"]
    assert ranked[0]["alignment_score"] == pytest.approx(0.4453, abs=1e-4)
    assert ranked[1]["alignment_score"] == pytest.approx(0.6283 * 0.4, abs=1e-4)
    assert ranked[0]["criteria_check"]["state"] == "consistent"
    assert ranked[1]["criteria_check"]["state"] == "conflict"


# ---------------------------------------------------------------------------
# 2. 知识库条款词表守卫与条款级证据
# ---------------------------------------------------------------------------
def test_knowledge_criteria_are_machine_checkable():
    entries = load_knowledge(settings.knowledge_dir)
    assert entries, "知识库不应为空"

    seen_ids: set[str] = set()
    for entry in entries:
        assert entry["criteria"], f"{entry['fault_type']} 缺少可核对的判据条款"
        for criterion in entry["criteria"]:
            assert criterion["metric"] in engine.METRICS, criterion
            assert criterion["op"] in engine.OPS, criterion
            assert criterion["claim"].strip()
            # claim 由 front-matter 的 `key: value | key: value` 解析而来，
            # 出现 ASCII 冒号或竖线会被误拆成字段，因此只允许中文标点
            assert ":" not in criterion["claim"] and "|" not in criterion["claim"], criterion
            assert criterion["id"] not in seen_ids, f"条款 id 重复：{criterion['id']}"
            seen_ids.add(criterion["id"])


def test_store_search_exposes_clause_level_evidence():
    top = KnowledgeStore().search("轴承 内圈 故障 BPFI 谱峰 峭度 冲击")[0]

    assert top["fault_type"] == "inner_race_fault"
    assert top["criteria"] and top["clause_hits"]
    assert f"[{top['clause_hits'][0]['id']}]" in top["evidence"]
    assert all({"id", "claim", "metric", "op", "value"} <= set(item) for item in top["criteria"])


# ---------------------------------------------------------------------------
# 3. 三个样例场景：top1 与条款状态
# ---------------------------------------------------------------------------
def test_inner_race_scenario_matches_all_clauses():
    _, candidates = _analyze(INNER_RACE)
    top = candidates[0]

    assert top["fault_type"] == "inner_race_fault"
    assert top["criteria_check"]["state"] == "consistent"
    assert top["criteria_check"]["hits"] == top["criteria_check"]["total"] == 6
    # 全部命中 → 判据因子 1.0，alignment_score 与检索分一致，排序不被改写
    assert top["alignment_score"] == pytest.approx(top["score"], abs=1e-4)


def test_outer_race_scenario_sinks_conflicting_candidate():
    _, candidates = _analyze(OUTER_RACE)

    assert candidates[0]["fault_type"] == "outer_race_fault"
    assert candidates[0]["criteria_check"]["state"] == "consistent"
    ball = next(item for item in candidates if item["fault_type"] == "ball_fault")
    # 外圈样本上「峭度温和 + 能量分散」这类滚动体判据与数据矛盾 → 该候选沉到末位
    assert ball["criteria_check"]["state"] == "conflict"
    assert candidates[-1]["fault_type"] == "ball_fault"
    assert ball["alignment_score"] < ball["score"]


def test_ball_scenario_clauses_confirm_dispersed_energy():
    _, candidates = _analyze(BALL)
    top = candidates[0]
    states = {item["id"]: item["state"] for item in top["criteria_check"]["items"]}

    assert top["fault_type"] == "ball_fault"
    assert top["criteria_check"]["state"] == "consistent"
    assert states["ball-dispersed"] == "hit"
    assert states["ball-tie-bsf"] == "hit"
    assert top["alignment_score"] == pytest.approx(top["score"], abs=1e-4)


# ---------------------------------------------------------------------------
# 4. 编排与报告：条款证据进入 trace / 提示词 / 报告 / markdown
# ---------------------------------------------------------------------------
def test_diagnose_exposes_clause_evidence_in_trace_and_report(local_backend):
    result = diagnose_local(NORMAL, INNER_RACE)
    top = result["candidates"][0]
    check = top["criteria_check"]

    assert top["fault_type"] == "inner_race_fault"
    assert (check["state"], check["hits"], check["total"]) == ("consistent", 6, 6)

    summary = result["trace"]["criteria_summary"]
    assert summary[0]["fault_type"] == "inner_race_fault"
    assert (summary[0]["state"], summary[0]["misses"], summary[0]["factor"]) == ("consistent", 0, 1.0)
    assert result["trace"]["criteria_conflict"] is None

    candidate = result["report"]["knowledge_candidates"][0]
    assert candidate["criteria_state"] == "consistent"
    assert candidate["criteria_factor"] == 1.0
    assert {row["clause_id"] for row in candidate["criteria"]} >= {"inner-rms-up", "inner-bpfi-peaks"}
    assert all(row["state"] in {"hit", "miss", "unknown"} for row in candidate["criteria"])

    assert "判据条款核对" in result["report_markdown"]
    assert "inner-bpfi-peaks" in result["report_markdown"]
    assert "判据条款核对" in result["report"]["confidence"]["reason"]


def test_conflicting_candidate_is_not_accepted_and_needs_human_review():
    comparison, candidates = _analyze(OUTER_RACE)
    ball = next(item for item in candidates if item["fault_type"] == "ball_fault")
    assert ball["criteria_check"]["state"] == "conflict"

    conflict = graph._criteria_conflict([ball])
    assert conflict["fault_type"] == "ball_fault"
    assert set(conflict["missed_clauses"]) == {"ball-kurt-mild", "ball-crest-mild", "ball-dispersed"}
    # 判据一致的候选不触发冲突记账
    assert graph._criteria_conflict([candidates[0]]) is None

    prompt = graph._build_user_prompt(DEVICE, None, comparison, [ball])
    assert "判据未命中" in prompt and "ball-dispersed" in prompt

    report = report_module.build_report(
        device=DEVICE, data_quality=None, comparison=comparison, rag={"sources": []}, candidates=[ball]
    )
    assert report["confidence"]["level"] == "low"
    assert "矛盾" in report["confidence"]["reason"]
    assert any("人工确认" in item for item in report["review_suggestions"])
    assert report["knowledge_candidates"][0]["criteria_state"] == "conflict"