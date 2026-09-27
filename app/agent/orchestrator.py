"""Agent 编排：把 MCP 工具链、RAG 检索与 LLM 表达串成一条可追溯的诊断流程。

自本次改造起，流程本身由 LangGraph 图表达（`app/agent/graph.py`），本模块保留
**唯一外部入口** `diagnose()`：`app/main.py`、`app/web/index.html` 与测试都只依赖它，
签名与返回键集合与改造前完全一致。

图的固定顺序（节点 + 条件分支，非开放式 Agent 循环）：

    validate_node → mcp_analysis_node → rag_retrieval_node → evidence_check_node
        → 证据不足 → report_node（输出「无法确认」）
        → 证据充分 → llm_node（结构化 JSON，失败降级）→ report_node

约束（详见 graph.py 各节点注释）：MCP 工具只经适配层调用、工具调用上限、
工具失败即停、数据校验不通过即停、证据不足不下结论且不留候选/来源残留、
故障类型只能来自 RAG 候选、模型失败降级为模板报告、模型超时保留已算出的证据。
"""

from __future__ import annotations

# 兼容性再导出：docs 与 scripts/smoke_llm.py 引用了这些名字，统一指向图中的实现
from app.agent.graph import (  # noqa: F401
    GRAPH,
    SYSTEM_PROMPT,
    _build_user_prompt,
    _device_block,
    _no_significant_change,
    _project_candidate,
    _run_llm,
    _validate_request,
    build_graph,
    run_diagnosis,
)
from app.agent.llm_client import extract_json_object as _parse_llm_json  # noqa: F401

__all__ = ["diagnose", "run_diagnosis", "GRAPH", "build_graph", "SYSTEM_PROMPT"]


async def diagnose(
    request: dict,
    *,
    max_tool_calls: int | None = None,
    llm_client=None,
    emitter=None,
) -> dict:
    """唯一外部入口。内部在线性图 `graph.GRAPH` 上执行一次诊断。

    emitter（app.agent.events.DiagnosisEmitter）非空时收集真实节点/工具事件，
    供 SSE 接口与会话记录共用同一套核心逻辑。
    """
    return await run_diagnosis(request, max_tool_calls=max_tool_calls, llm_client=llm_client, emitter=emitter)