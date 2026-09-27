"""向量库改造测试：Chroma 持久化索引、混合排序、降级 BM25、幂等重建与切分规则。

全部离线完成：embedding 通过**注入的假 embedder** 提供，不依赖 API Key；
索引目录一律建在 `tmp_path` 下，绝不写入 data/chroma/。

契约来源：
  - app/rag/ingest.py   ：rebuild / build_chunks / open_collection / chunk_metadata
  - app/rag/store.py    ：KnowledgeStore（retrieval_mode ∈ hybrid|bm25、degraded/degraded_reason、
                          候选含 retrieval/chunk_id/document/vector_score/bm25_score）
  - app/rag/retriever.py：retrieve_candidates 透出 degraded / degraded_reason
"""

from __future__ import annotations

import re
import socket

import pytest

from app.config import settings
from app.rag import ingest, retriever
from app.rag.store import KnowledgeStore, load_knowledge
from tests.helpers import build_indexed_store, exploding_embedder, fake_embedder

KNOWLEDGE = settings.knowledge_dir
# 一条与知识库高度相关的中文查询（内圈故障特征术语）
QUERY = "轴承 内圈 故障 BPFI 谱峰 峭度 冲击"

# 每个候选/元数据都必须带的字段（retriever 契约）
CANDIDATE_KEYS = {"retrieval", "chunk_id", "document", "vector_score", "bm25_score"}
REQUIRED_METADATA = {"fault_type", "title", "source", "source_url", "locator", "document", "chunk_id"}
CHUNK_ID_RE = re.compile(r"^[a-z_]+#chunk-\d{3}$")


@pytest.fixture
def indexed(tmp_path):
    """在临时目录建好向量索引，返回 (store, summary, chroma_dir)。"""
    chroma_dir = tmp_path / "chroma"
    store, summary = build_indexed_store(chroma_dir)
    return store, summary, chroma_dir


def _overlap(previous: str, following: str) -> int:
    """前一块的结尾与后一块的开头的最长重叠字符数。"""
    best = 0
    for size in range(1, min(len(previous), len(following)) + 1):
        if previous[-size:] == following[:size]:
            best = size
    return best


# ---------------------------------------------------------------------------
# 1. 知识条目可以写入 Chroma（临时目录，不污染 data/chroma/）
# ---------------------------------------------------------------------------
def test_rebuild_writes_entries_into_chroma(indexed):
    store, summary, chroma_dir = indexed

    assert summary["count"] == summary["chunks"] > 0
    assert summary["count"] == len(summary["chunk_ids"])
    assert summary["documents"] == len(load_knowledge(KNOWLEDGE)) > 0
    assert str(chroma_dir) in summary["chroma_dir"]

    collection = ingest.open_collection(chroma_dir, settings.chroma_collection)
    assert collection.count() == summary["count"]

    # 索引可用 → 进入 hybrid 模式（构造阶段不调用 embedding）
    assert store.retrieval_mode == "hybrid"
    assert store.degraded is False and store.degraded_reason is None


# ---------------------------------------------------------------------------
# 2. Chroma 查询结果包含完整元数据
# ---------------------------------------------------------------------------
def test_chroma_query_returns_complete_metadata(indexed):
    _, _, chroma_dir = indexed
    collection = ingest.open_collection(chroma_dir, settings.chroma_collection)

    result = collection.query(
        query_embeddings=[fake_embedder([QUERY])[0]],
        n_results=3,
        include=["metadatas", "documents", "distances"],
    )
    metadatas = result["metadatas"][0]
    assert metadatas, "向量索引应能查到条目"

    for meta in metadatas:
        assert REQUIRED_METADATA <= set(meta), f"元数据缺字段：{REQUIRED_METADATA - set(meta)}"
        for key in ("fault_type", "title", "document", "chunk_id"):
            assert str(meta[key]).strip(), f"元数据 {key} 不应为空：{meta}"
        # 块 ID 由 fault_type + 序号构成，且与元数据自洽
        assert meta["chunk_id"].startswith(f"{meta['fault_type']}#chunk-")
        assert meta["document"].endswith(".md")


# ---------------------------------------------------------------------------
# 3. 向量检索与 BM25 混合排序正常：final = 0.6*vector + 0.4*bm25
# ---------------------------------------------------------------------------
def test_hybrid_ranking_uses_weighted_fusion(indexed):
    store, _, _ = indexed
    assert store.retrieval_mode == "hybrid"

    results = store.search(QUERY, top_k=5)
    assert results, "混合检索应返回候选"
    assert all(item["retrieval"] == "hybrid" for item in results)
    assert all(CANDIDATE_KEYS <= set(item) for item in results)

    for item in results:
        assert 0.0 <= item["vector_score"] <= 1.0
        assert 0.0 <= item["bm25_score"] <= 1.0
        assert item["score"] == pytest.approx(
            settings.rag_vector_weight * item["vector_score"]
            + settings.rag_bm25_weight * item["bm25_score"],
            abs=2e-4,  # 三个分数各自四舍五入到 4 位小数带来的累积误差
        )
        assert CHUNK_ID_RE.match(item["chunk_id"])
        assert item["document"].endswith(".md")

    # 结果是按 final_score 降序排列的
    scores = [item["score"] for item in results]
    assert scores == sorted(scores, reverse=True)
    # 同一故障类型只保留最高分文本块，不重复占位
    assert len({item["fault_type"] for item in results}) == len(results)


# ---------------------------------------------------------------------------
# 4. Embedding 失败时降级 BM25
# ---------------------------------------------------------------------------
def test_embedding_failure_degrades_to_bm25(indexed, monkeypatch):
    _, _, chroma_dir = indexed
    store = KnowledgeStore(
        knowledge_dir=KNOWLEDGE, chroma_dir=chroma_dir, embedder=exploding_embedder
    )
    # 构造阶段只打开已持久化的集合，不调用 embedding，因此仍是 hybrid
    assert store.retrieval_mode == "hybrid"

    results = store.search(QUERY, top_k=3)

    assert store.retrieval_mode == "bm25"
    assert store.degraded is True
    assert store.degraded_reason and "已降级 BM25" in store.degraded_reason
    assert results and all(item["retrieval"] == "bm25" for item in results)
    # 降级后的 BM25 结果不再带向量专属字段
    assert all("vector_score" not in item for item in results)

    # 检索入口同样透出降级契约
    monkeypatch.setattr(retriever, "_store", store)
    rag = retriever.retrieve_candidates(QUERY, top_k=3)
    assert rag["retrieval"] == "bm25"
    assert rag["degraded"] is True
    assert rag["degraded_reason"]


# ---------------------------------------------------------------------------
# 5. 没有 API Key 时不发起任何网络请求
# ---------------------------------------------------------------------------
def test_without_api_key_no_network_request(monkeypatch, tmp_path):
    def _blocked_connect(self, *args, **kwargs):  # noqa: ARG001
        raise AssertionError("未配置 embedding 时不应发起任何网络连接")

    monkeypatch.setattr(socket.socket, "connect", _blocked_connect)

    assert settings.embedding_available is False
    assert ingest.build_embedder() is None

    store = KnowledgeStore(knowledge_dir=KNOWLEDGE, chroma_dir=tmp_path / "missing_chroma")
    assert store.retrieval_mode == "bm25"
    assert store.degraded is False  # 无 embedding 配置是设计内行为，不算降级

    monkeypatch.setattr(retriever, "_store", store)
    rag = retriever.retrieve_candidates(QUERY)
    assert rag["retrieval"] == "bm25"
    assert rag["candidates"], "BM25 分支应能正常召回"
    assert all(item["retrieval"] == "bm25" for item in rag["candidates"])


# ---------------------------------------------------------------------------
# 6. 索引重建不产生重复条目（幂等）
# ---------------------------------------------------------------------------
def test_rebuild_twice_is_idempotent(tmp_path):
    chroma_dir = tmp_path / "chroma"

    first = ingest.rebuild(fake_embedder, knowledge_dir=KNOWLEDGE, chroma_dir=chroma_dir)
    collection = ingest.open_collection(chroma_dir, settings.chroma_collection)
    count_after_first = collection.count()

    second = ingest.rebuild(fake_embedder, knowledge_dir=KNOWLEDGE, chroma_dir=chroma_dir)
    collection = ingest.open_collection(chroma_dir, settings.chroma_collection)

    assert first["chunk_ids"] == second["chunk_ids"]
    assert set(first["chunk_ids"]) == set(second["chunk_ids"])
    assert first["count"] == second["count"] == count_after_first
    assert collection.count() == first["count"]
    assert second["pruned"] == 0

    info = ingest.status(chroma_dir=chroma_dir)
    assert info["exists"] is True and info["error"] is None
    assert info["count"] == first["count"]
    assert set(info["chunk_ids"]) == set(first["chunk_ids"])


# ---------------------------------------------------------------------------
# 7. 切分规则：≤500 字符、相邻重叠约 80 字符、块 ID 稳定
# ---------------------------------------------------------------------------
def test_chunk_split_rule_and_stable_ids():
    entries = load_knowledge(KNOWLEDGE)
    chunks = ingest.build_chunks(entries, KNOWLEDGE)
    again = ingest.build_chunks(load_knowledge(KNOWLEDGE), KNOWLEDGE)

    assert chunks and [c["chunk_id"] for c in chunks] == [c["chunk_id"] for c in again]

    for chunk in chunks:
        assert len(chunk["body"]) <= settings.rag_chunk_size
        assert CHUNK_ID_RE.match(chunk["chunk_id"])
        assert chunk["chunk_id"].startswith(f"{chunk['fault_type']}#chunk-")
        # 块头把结构化信息附在每个块上，检索时可回填元数据
        assert f"chunk_id: {chunk['chunk_id']}" in chunk["text"]
        assert f"fault_type: {chunk['fault_type']}" in chunk["text"]

    # 同一 fault_type 的块序号从 000 起连续
    by_fault: dict[str, list[dict]] = {}
    for chunk in chunks:
        by_fault.setdefault(chunk["fault_type"], []).append(chunk)
    for fault_type, group in by_fault.items():
        assert [c["chunk_id"] for c in group] == [
            f"{fault_type}#chunk-{i:03d}" for i in range(len(group))
        ]

    # 相邻块重叠约等于 RAG_CHUNK_OVERLAP（按断句点切分，实测为 79 ≈ 80）
    overlaps = [
        _overlap(a["body"], b["body"])
        for group in by_fault.values()
        for a, b in zip(group, group[1:])
    ]
    assert overlaps, "知识条目正文应长于单块上限，才能观察到块间重叠"
    for value in overlaps:
        assert value == pytest.approx(settings.rag_chunk_overlap, abs=2)