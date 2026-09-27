"""HTTP 接口层的请求 / 响应模型。

刻意保持宽松：诊断结果的内部结构由编排层定义并会演进，
这里只做字段镜像与透传，不在 Pydantic 层重复一遍校验。
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class HealthResponse(BaseModel):
    status: str
    mcp_backend: str
    llm_available: bool
    retrieval_mode: str
    sample_count: int


class SampleInfo(BaseModel):
    """只公开内置样例元数据，不向浏览器暴露服务端文件系统路径。"""

    model_config = ConfigDict(extra="ignore")

    sample_id: str
    file: str | None = None
    device_type: str | None = None
    sampling_rate: float | None = None
    rotation_speed: float | None = None
    sensor_position: str | None = None
    condition: str | None = None
    condition_label: str | None = None
    n_samples: int | None = None
    note: str | None = None


class SampleListResponse(BaseModel):
    count: int
    samples: list[SampleInfo]


class DiagnoseResponse(BaseModel):
    """orchestrator.diagnose() 返回 dict 的字段镜像（status 为唯一强约束字段）。"""

    model_config = ConfigDict(extra="allow")

    session_id: str | None = None
    status: str
    error: dict | None = None
    device: dict = Field(default_factory=dict)
    data_quality: dict | None = None
    features: dict | None = None
    comparison: dict | None = None
    rag: dict | None = None
    candidates: list[dict] = Field(default_factory=list)
    report: dict | None = None
    report_markdown: str | None = None
    sources: list[dict] = Field(default_factory=list)
    review_suggestions: list[str] = Field(default_factory=list)
    trace: dict = Field(default_factory=dict)


class AskRequest(BaseModel):
    """P2 受控追问请求：问题必须是会话允许清单内的原文（或其归一化形式）。"""

    question: str = Field(min_length=1, max_length=200)


class AskResponse(BaseModel):
    """受控追问回答：全部字段来自本次会话保存的上下文，可追溯。"""

    session_id: str
    question: str
    question_key: str
    answer: str
    facts_used: list[str] = Field(default_factory=list)
    context_scope: str
    no_support: bool = False


class ChatRequest(BaseModel):
    """会话内自由问答请求：问题自由输入，回答仍被限制在本次会话事实内。"""

    question: str = Field(default="", max_length=300)


class ChatResponse(BaseModel):
    """自由问答回答：mode=llm 由模型生成并校验通过；mode=deterministic 为确定性降级。"""

    session_id: str
    question: str
    answer: str
    mode: str
    facts_used: list[str] = Field(default_factory=list)
    context_scope: str
    no_support: bool = False
    model_error: dict | None = None


class SessionSummary(BaseModel):
    """会话列表条目：只含列表页展示所需字段（不含完整报告与事件）。"""

    session_id: str
    created_at: float
    normal_input: str | None = None
    abnormal_input: str | None = None
    status: str | None = None
    conclusion: str | None = None
    confidence_level: str | None = None
    message_count: int = 0


class SessionListResponse(BaseModel):
    count: int
    sessions: list[SessionSummary]


class ChatMessage(BaseModel):
    """落盘的对话消息：kind=ask 为受控追问，kind=chat 为自由问答。"""

    role: str
    content: str
    meta: dict = Field(default_factory=dict)
    created_at: float | None = None


class SessionDetailResponse(BaseModel):
    """单个会话的完整记录：前端刷新后据此恢复事件轨迹、结果面板与对话。"""

    session_id: str
    created_at: float
    request_summary: dict
    result: dict
    events: list[dict] = Field(default_factory=list)
    messages: list[ChatMessage] = Field(default_factory=list)
