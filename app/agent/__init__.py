"""Agent 编排与报告生成。

- `graph.py`：LangGraph 图，诊断流程的唯一实现（节点 + 条件分支）。
- `orchestrator.py`：唯一外部入口 `diagnose()`，委托给图执行。
- `llm_client.py`：兼容 OpenAI 协议的模型客户端与结构化 JSON 解析。
- `report.py`：确定性报告组装与 Markdown 渲染。
"""