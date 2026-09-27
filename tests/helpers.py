"""测试公共工具：路径、后端切换、请求构造、诊断调用与报告字段清单。

注意：这里不修改 app/ 下任何实现，只做测试侧的准备与编排调用。
"""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
from pathlib import Path

from app.agent import orchestrator
from app.agent.llm_client import LLMError
from app.config import settings

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "data" / "samples"

DEFAULT_SAMPLING_RATE = 12000.0
DEFAULT_ROTATION_SPEED = 1797.0

NORMAL = "normal_1797"
INNER_RACE = "inner_race_1797"
OUTER_RACE = "outer_race_1797"
BALL = "ball_1797"
CORRUPTED = "corrupted_1797"

# 报告必须齐全的字段（与 app/agent/report.py 的 build_report 一一对应）
REPORT_KEYS = (
    "analysis_target",
    "data_quality_summary",
    "feature_comparison",
    "band_energy_comparison",
    "spectrum_notes",
    "knowledge_candidates",
    "conclusion",
    "conclusion_evidence",
    "conclusion_sources",
    "confidence",
    "data_facts",
    "knowledge_basis",
    "inferences",
    "review_suggestions",
    "prototype_limits",
)


def sample_path(stem: str) -> str:
    """内置样例 CSV 的绝对路径（stem 不含 .csv）。"""
    return str((SAMPLES / f"{stem}.csv").resolve())


@contextmanager
def backend_mode(mode: str):
    """临时切换 MCP 后端（"local" / "stdio"）。

    settings 是 frozen dataclass，普通赋值会抛 FrozenInstanceError，这里用
    object.__setattr__ 绕过 dataclass 生成的 __setattr__，退出时恢复原值。
    """
    previous = settings.mcp_mode
    object.__setattr__(settings, "mcp_mode", mode)
    try:
        yield mode
    finally:
        object.__setattr__(settings, "mcp_mode", previous)


def write_signal_csv(path, values, *, sampling_rate: float = DEFAULT_SAMPLING_RATE, timestamps: bool = True) -> str:
    """把一维数组写成 timestamp,amplitude（或单列 amplitude）CSV，返回绝对路径。"""
    path = Path(path)
    lines = ["timestamp,amplitude"] if timestamps else ["amplitude"]
    for index, value in enumerate(values):
        if timestamps:
            lines.append(f"{index / float(sampling_rate):.9f},{float(value):.9f}")
        else:
            lines.append(f"{float(value):.9f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path.resolve())


def build_request(
    normal_stem: str,
    abnormal_stem: str,
    *,
    sampling_rate: float = DEFAULT_SAMPLING_RATE,
    rotation_speed: float = DEFAULT_ROTATION_SPEED,
    sensor_position: str = "drive_end",
    device_type: str = "bearing",
) -> dict:
    return {
        "normal_path": sample_path(normal_stem),
        "abnormal_path": sample_path(abnormal_stem),
        "sampling_rate": sampling_rate,
        "rotation_speed": rotation_speed,
        "sensor_position": sensor_position,
        "device_type": device_type,
    }


def diagnose_local(normal_stem: str, abnormal_stem: str, *, llm_client=None, **kwargs) -> dict:
    """在 local 后端下跑一次真实编排流程（端到端测试与评测记录统一走这里）。"""
    request = build_request(normal_stem, abnormal_stem, **kwargs)
    with backend_mode("local"):
        return asyncio.run(orchestrator.diagnose(request, llm_client=llm_client))


class TimeoutLLMClient:
    """假模型客户端：available=True 且 chat() 直接抛 LLMError(code="timeout")。"""

    available = True

    def chat(self, system: str, user: str) -> str:  # noqa: ARG002 - 只需匹配真实客户端签名
        raise LLMError("timeout", "模拟模型调用超时")


class ScriptedLLMClient:
    """假模型客户端：available=True，chat() 固定返回预设文本，并记录收到的提示词。"""

    available = True

    def __init__(self, reply: str):
        self.reply = reply
        self.calls: list[tuple[str, str]] = []

    def chat(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        return self.reply


# ---------------------------------------------------------------------------
# 离线假 embedder 与向量索引构造（不依赖 API Key，不发任何网络请求）
# ---------------------------------------------------------------------------
_EMBED_DIM = 256


def fake_embedder(texts):
    """确定性离线假 embedder：复用知识库分词器做 bag-of-words 哈希投影。

    同一进程内 hash 稳定，因此「写索引」与「查询」用同一套向量；
    不同文本因词分布不同而得到不同向量，混合检索的向量分支才有区分度。
    """
    from app.rag.store import tokenize

    vectors: list[list[float]] = []
    for text in texts:
        vec = [0.0] * _EMBED_DIM
        for token in tokenize(text):
            vec[hash(token) % _EMBED_DIM] += 1.0
        vectors.append(vec)
    return vectors


def exploding_embedder(texts):
    """模拟向量服务不可用：任何 embedding 调用都抛异常，用于验证降级为 BM25。"""
    raise RuntimeError("模拟 embedding 服务不可用")


def build_indexed_store(chroma_dir, *, embedder=None, store_embedder=None, knowledge_dir=None):
    """在 chroma_dir 下建好向量索引并返回 (KnowledgeStore, rebuild 摘要)。

    索引目录固定为调用方传入的临时目录，绝不写 data/chroma/。
    `store_embedder` 可单独指定「打开索引的那个 store」用的 embedder，
    便于构造「索引写好后查询时 embedding 失败」的降级场景。
    """
    from app.rag import ingest
    from app.rag.store import KnowledgeStore

    knowledge_dir = knowledge_dir or settings.knowledge_dir
    active = embedder or fake_embedder
    summary = ingest.rebuild(active, knowledge_dir=knowledge_dir, chroma_dir=chroma_dir)
    store = KnowledgeStore(
        knowledge_dir=knowledge_dir,
        chroma_dir=chroma_dir,
        embedder=active if store_embedder is None else store_embedder,
    )
    return store, summary