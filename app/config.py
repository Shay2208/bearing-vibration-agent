"""项目配置：所有可变参数统一从环境变量读取，不写入代码。"""

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_dotenv(path: Path) -> None:
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


_load_dotenv(PROJECT_ROOT / ".env")


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ[name])
    except (KeyError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ[name])
    except (KeyError, ValueError):
        return default


@dataclass(frozen=True)
class Settings:
    # 模型接口：兼容 OpenAI 协议
    model_api_base: str = field(default_factory=lambda: os.environ.get("MODEL_API_BASE", ""))
    model_api_key: str = field(default_factory=lambda: os.environ.get("MODEL_API_KEY", ""))
    model_name: str = field(default_factory=lambda: os.environ.get("MODEL_NAME", "gpt-4o-mini"))
    embedding_model: str = field(default_factory=lambda: os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small"))
    # 向量接口：text=标准 OpenAI 兼容 /embeddings；multimodal=方舟 /embeddings/multimodal（每条文本单独请求）
    embedding_api: str = field(default_factory=lambda: os.environ.get("EMBEDDING_API", "text").lower())
    use_llm: bool = field(default_factory=lambda: os.environ.get("USE_LLM", "auto").lower())

    # MCP：stdio 走真实 MCP 服务，local 走进程内函数
    # 公开演示默认走进程内后端，避免每次请求都启动 MCP 子进程；stdio 仍可显式开启验证协议链路。
    mcp_mode: str = field(default_factory=lambda: os.environ.get("MCP_MODE", "local").lower())
    mcp_server_cmd: str = field(default_factory=lambda: os.environ.get("MCP_SERVER_CMD", ""))

    # Agent 约束
    max_tool_calls: int = field(default_factory=lambda: _env_int("MAX_TOOL_CALLS", 8))
    llm_timeout: float = field(default_factory=lambda: _env_float("LLM_TIMEOUT", 20.0))
    llm_max_retries: int = field(default_factory=lambda: _env_int("LLM_MAX_RETRIES", 1))
    max_upload_bytes: int = field(default_factory=lambda: _env_int("MAX_UPLOAD_BYTES", 20 * 1024 * 1024))
    # P3 自适应分支：MCP 工具失败时仅对这些可重试错误做有限重试（数据内容错误不重试）
    mcp_max_retries: int = field(default_factory=lambda: _env_int("MCP_MAX_RETRIES", 1))
    # 前两名候选分数差小于该阈值时触发频率证据复核
    candidate_gap_threshold: float = field(default_factory=lambda: _env_float("CANDIDATE_GAP_THRESHOLD", 0.05))
    # 信号时长低于该秒数时按数据不足处理，停止强行诊断
    min_signal_duration: float = field(default_factory=lambda: _env_float("MIN_SIGNAL_DURATION", 0.25))

    # RAG
    rag_top_k: int = field(default_factory=lambda: _env_int("RAG_TOP_K", 5))
    rag_score_threshold: float = field(default_factory=lambda: _env_float("RAG_SCORE_THRESHOLD", 0.35))
    # 会话与对话消息落盘位置；自由问答进入提示词的最近对话轮数（1 轮 = 用户问 + 助手答）
    sessions_db: Path = field(
        default_factory=lambda: Path(os.environ.get("SESSIONS_DB_PATH") or (PROJECT_ROOT / "data" / "sessions.db"))
    )
    chat_history_turns: int = field(default_factory=lambda: _env_int("CHAT_HISTORY_TURNS", 3))
    # 诊断过程中旁挂落盘的波形音频目录与保留上限（懒加载端点按 session_id + 槽位取文件）
    audio_dir: Path = field(
        default_factory=lambda: Path(os.environ.get("AUDIO_DIR") or (PROJECT_ROOT / "data" / "audio"))
    )
    audio_max_files: int = field(default_factory=lambda: _env_int("AUDIO_MAX_FILES", 200))

    # RAG 向量库：Chroma 持久化目录 + 混合检索权重与切分粒度
    chroma_collection: str = field(
        default_factory=lambda: os.environ.get("CHROMA_COLLECTION", "bearing_knowledge")
    )
    rag_vector_weight: float = field(default_factory=lambda: _env_float("RAG_VECTOR_WEIGHT", 0.6))
    rag_bm25_weight: float = field(default_factory=lambda: _env_float("RAG_BM25_WEIGHT", 0.4))
    rag_chunk_size: int = field(default_factory=lambda: _env_int("RAG_CHUNK_SIZE", 500))
    rag_chunk_overlap: int = field(default_factory=lambda: _env_int("RAG_CHUNK_OVERLAP", 80))
    # 向量召回需要真实 embedding 质量验证后再开启；注入测试 embedder 不受此开关影响。
    enable_hybrid_retrieval: bool = field(
        default_factory=lambda: os.environ.get("ENABLE_HYBRID_RETRIEVAL", "false").lower() in {"1", "true", "yes", "on"}
    )

    data_dir: Path = PROJECT_ROOT / "data"
    samples_dir: Path = PROJECT_ROOT / "data" / "samples"
    knowledge_dir: Path = PROJECT_ROOT / "app" / "rag" / "knowledge"
    chroma_dir: Path = field(
        default_factory=lambda: Path(os.environ.get("CHROMA_DIR") or (PROJECT_ROOT / "data" / "chroma"))
    )
    web_dir: Path = PROJECT_ROOT / "app" / "web"

    @property
    def llm_available(self) -> bool:
        if self.use_llm == "never":
            return False
        return bool(self.model_api_base and self.model_api_key)

    @property
    def embedding_available(self) -> bool:
        return bool(self.model_api_base and self.model_api_key and self.embedding_model)


settings = Settings()
