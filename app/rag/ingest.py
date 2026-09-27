"""知识条目切分、Chroma 向量索引构建与现状查询。

在项目根目录执行：

    python -m app.rag.ingest --rebuild   # 重建向量索引（幂等：稳定 ID + upsert）
    python -m app.rag.ingest --status    # 打印索引现状（集合条目数与块 ID）

切分规则：每个知识条目的正文按「≤ RAG_CHUNK_SIZE 个字符、相邻块重叠 RAG_CHUNK_OVERLAP 个字符」
切开，并把 front matter 的结构化信息（fault_type / title / applicable / typical_features /
frequency_signature / sources / source_url / locator / document）附加到**每一个**文本块上。
块 ID 形如 `inner_race_fault#chunk-000`，由「fault_type + 块序号」决定，重复构建得到完全相同的 ID 集合。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from app.config import settings
from app.rag.store import load_knowledge

_COLLECTION_METADATA = {"hnsw:space": "cosine"}
_BOUNDARY_CHARS = "\n。；;"
_BOUNDARY_LOOKBACK = 100
_EMBED_BATCH_SIZE = 64


class EmbeddingNotConfigured(RuntimeError):
    """没有可用的 embedding 服务（未配置 MODEL_API_BASE / MODEL_API_KEY / EMBEDDING_MODEL）。"""


# ---------------------------------------------------------------------------
# 切分
# ---------------------------------------------------------------------------
def _find_boundary(text: str, start: int, end: int) -> int:
    """在窗口尾部附近找一个更自然的断点（换行 / 句号 / 分号），找不到就原样返回 end。"""
    limit = max(start + 1, end - _BOUNDARY_LOOKBACK)
    for pos in range(end - 1, limit - 1, -1):
        if text[pos] in _BOUNDARY_CHARS:
            return pos + 1
    return end


def split_text(text: str, size: int, overlap: int) -> list[str]:
    """把正文按字符数滑窗切块，块长 ≤ size，块间重叠 overlap。"""
    text = (text or "").strip()
    if not text:
        return []
    if size <= 0:
        return [text]
    overlap = max(0, min(int(overlap), size - 1))
    pieces: list[str] = []
    start, length = 0, len(text)
    while start < length:
        end = min(start + size, length)
        if end < length:
            boundary = _find_boundary(text, start, end)
            if boundary > start:
                end = boundary
        piece = text[start:end].strip()
        if piece:
            pieces.append(piece)
        if end >= length:
            break
        start = max(end - overlap, start + 1)
    return pieces


def _source_fields(entry: dict) -> tuple[str, str, str]:
    sources = entry.get("sources") or []
    first = sources[0] if sources else {}
    return (
        str(first.get("name") or ""),
        str(first.get("url") or ""),
        str(first.get("locator") or ""),
    )


def _chunk_header(entry: dict, source: str, source_url: str, locator: str, chunk_id: str) -> str:
    """把 front matter 的结构化信息拼成块头，附加到每一个文本块。"""
    typical = "；".join(entry.get("typical_features") or [])
    source_names = "；".join(str(s.get("name") or "") for s in (entry.get("sources") or []))
    return "\n".join([
        f"fault_type: {entry.get('fault_type', '')}",
        f"title: {entry.get('title', '')}",
        f"applicable: {entry.get('applicable', '')}",
        f"typical_features: {typical}",
        f"frequency_signature: {entry.get('frequency_signature', '')}",
        f"source: {source}",
        f"sources: {source_names}",
        f"source_url: {source_url}",
        f"locator: {locator}",
        f"document: {entry.get('document', '')}",
        f"chunk_id: {chunk_id}",
    ])


def build_chunks(entries: list[dict], knowledge_dir: Path | None = None) -> list[dict]:
    """把知识条目切成文本块；块 ID 稳定可读，形如 `inner_race_fault#chunk-000`。"""
    knowledge_dir = Path(knowledge_dir) if knowledge_dir else settings.knowledge_dir
    size = int(settings.rag_chunk_size)
    overlap = int(settings.rag_chunk_overlap)

    chunks: list[dict] = []
    for entry_index, entry in enumerate(entries):
        fault_type = entry.get("fault_type", "")
        document = entry.get("document") or f"{fault_type}.md"
        source, source_url, locator = _source_fields(entry)
        pieces = split_text(entry.get("body") or "", size, overlap) or [""]
        for chunk_index, piece in enumerate(pieces):
            chunk_id = f"{fault_type}#chunk-{chunk_index:03d}"
            chunks.append({
                "chunk_id": chunk_id,
                "chunk_index": chunk_index,
                "entry_index": entry_index,
                "fault_type": fault_type,
                "title": entry.get("title", ""),
                "source": source,
                "source_url": source_url,
                "locator": locator,
                "document": document,
                "body": piece,
                "text": f"{_chunk_header(entry, source, source_url, locator, chunk_id)}\n\n{piece}".strip(),
            })
    return chunks


def chunk_metadata(chunk: dict) -> dict:
    """写入 Chroma 的元数据；值只能是 str/int/float/bool，缺失一律写空串。"""
    return {
        "fault_type": chunk["fault_type"],
        "title": chunk["title"],
        "source": chunk["source"],
        "source_url": chunk["source_url"],
        "locator": chunk["locator"],
        "document": chunk["document"],
        "chunk_id": chunk["chunk_id"],
        "chunk_index": int(chunk["chunk_index"]),
    }


# ---------------------------------------------------------------------------
# embedding 与 Chroma
# ---------------------------------------------------------------------------
_EMBED_INSTRUCTIONS = (
    "Target_modality: text.\n"
    "Instruction:Represent the bearing-fault knowledge text for semantic retrieval.\n"
    "Query:"
)


def build_embedder():
    """生产用 embedding 回调；未配置时返回 None，绝不发起网络请求。

    两种接口由 EMBEDDING_API 选择：默认 text 走标准 OpenAI 兼容 `/embeddings`；
    方舟多模态端点（vision 系列模型）走 multimodal。
    """
    if not settings.embedding_available:
        return None
    if settings.embedding_api == "multimodal":
        return _build_multimodal_embedder()
    from openai import OpenAI

    client = OpenAI(base_url=settings.model_api_base, api_key=settings.model_api_key)

    def embed(texts: list[str]) -> list[list[float]]:
        response = client.embeddings.create(model=settings.embedding_model, input=list(texts))
        return [list(map(float, item.embedding)) for item in response.data]

    return embed


def _build_multimodal_embedder():
    """方舟 `/embeddings/multimodal`：整个 input 列表会被融合成一个向量，因此每条文本单独发一次请求。"""
    import httpx

    url = settings.model_api_base.rstrip("/") + "/embeddings/multimodal"
    headers = {"Authorization": f"Bearer {settings.model_api_key}"}

    def embed(texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            response = httpx.post(
                url,
                headers=headers,
                json={
                    "model": settings.embedding_model,
                    "instructions": _EMBED_INSTRUCTIONS,
                    "encoding_format": "float",
                    "input": [{"type": "text", "text": text}],
                },
                timeout=settings.llm_timeout,
            )
            if response.status_code != 200:
                raise RuntimeError(f"多模态向量接口返回 {response.status_code}：{response.text[:300]}")
            vectors.append([float(value) for value in response.json()["data"]["embedding"]])
        return vectors

    return embed


def _client(chroma_dir: Path):
    """打开 Chroma 持久化客户端；显式关闭匿名遥测，避免任何外发请求。"""
    import chromadb

    return chromadb.PersistentClient(
        path=str(chroma_dir), settings=chromadb.Settings(anonymized_telemetry=False)
    )


def open_collection(chroma_dir: Path, collection_name: str):
    """打开已持久化的 Chroma 集合；目录或集合不存在时抛出异常（由调用方降级）。"""
    path = Path(chroma_dir)
    if not path.is_dir():
        raise FileNotFoundError(f"向量索引目录不存在：{path}")
    return _client(path).get_collection(collection_name)


def _embed_all(embedder, texts: list[str], batch_size: int = _EMBED_BATCH_SIZE) -> list[list[float]]:
    vectors: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        vectors.extend(embedder(batch))
    if len(vectors) != len(texts):
        raise RuntimeError(f"embedding 返回条数不匹配：期望 {len(texts)}，实际 {len(vectors)}")
    return vectors


def rebuild(
    embedder=None,
    *,
    knowledge_dir: Path | None = None,
    chroma_dir: Path | None = None,
    collection_name: str | None = None,
) -> dict:
    """构建/更新 Chroma 索引。稳定 ID + upsert，重复执行不产生重复条目（幂等）。"""
    knowledge_dir = Path(knowledge_dir) if knowledge_dir else settings.knowledge_dir
    chroma_dir = Path(chroma_dir) if chroma_dir else settings.chroma_dir
    collection_name = collection_name or settings.chroma_collection

    entries = load_knowledge(knowledge_dir)
    chunks = build_chunks(entries, knowledge_dir)
    if not chunks:
        raise RuntimeError(f"知识库为空：{knowledge_dir} 下没有可索引的条目")

    active_embedder = embedder or build_embedder()
    if active_embedder is None:
        raise EmbeddingNotConfigured(
            "未配置 embedding 服务：MODEL_API_BASE / MODEL_API_KEY 为空（或 EMBEDDING_MODEL 为空），"
            "无法生成向量索引。"
        )

    vectors = _embed_all(active_embedder, [chunk["text"] for chunk in chunks])

    chroma_dir.mkdir(parents=True, exist_ok=True)
    collection = _client(chroma_dir).get_or_create_collection(
        collection_name, metadata=_COLLECTION_METADATA
    )

    ids = [chunk["chunk_id"] for chunk in chunks]
    collection.upsert(
        ids=ids,
        embeddings=[list(map(float, vector)) for vector in vectors],
        documents=[chunk["text"] for chunk in chunks],
        metadatas=[chunk_metadata(chunk) for chunk in chunks],
    )
    # 清理历史遗留块（例如知识条目被改写后块数变少），保证 ID 集合与当前切分完全一致
    stale = [cid for cid in collection.get(include=["metadatas"]).get("ids", []) if cid not in set(ids)]
    if stale:
        collection.delete(ids=stale)

    return {
        "collection": collection_name,
        "chroma_dir": str(chroma_dir),
        "documents": len(entries),
        "chunks": len(chunks),
        "upserted": len(ids),
        "pruned": len(stale),
        "count": collection.count(),
        "chunk_ids": ids,
    }


def status(*, chroma_dir: Path | None = None, collection_name: str | None = None) -> dict:
    """查看索引现状；目录不存在时不会创建任何文件。"""
    chroma_dir = Path(chroma_dir) if chroma_dir else settings.chroma_dir
    collection_name = collection_name or settings.chroma_collection
    info: dict = {
        "chroma_dir": str(chroma_dir),
        "collection": collection_name,
        "exists": chroma_dir.is_dir(),
        "count": 0,
        "chunk_ids": [],
        "error": None,
    }
    if not info["exists"]:
        return info
    try:
        collection = open_collection(chroma_dir, collection_name)
    except Exception as exc:  # 集合不存在 / 目录里没有可读索引
        info["error"] = f"{type(exc).__name__}: {exc}"
        return info
    info["count"] = collection.count()
    info["chunk_ids"] = sorted(collection.get(include=["metadatas"]).get("ids", []))
    return info


# ---------------------------------------------------------------------------
# 命令行入口
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m app.rag.ingest",
        description="知识库向量索引：构建（--rebuild）与现状查询（--status）",
    )
    parser.add_argument("--rebuild", action="store_true", help="重建向量索引（幂等：稳定 ID + upsert）")
    parser.add_argument("--status", action="store_true", help="打印索引现状（集合条目数与块 ID）")
    parser.add_argument("--knowledge-dir", default=None, help="知识条目目录，默认 app/rag/knowledge")
    parser.add_argument("--chroma-dir", default=None, help="Chroma 持久化目录，默认 data/chroma")
    parser.add_argument("--collection", default=None, help="Chroma 集合名，默认 bearing_knowledge")
    args = parser.parse_args(argv)

    if not args.rebuild and not args.status:
        parser.print_help()
        return 0

    if args.rebuild:
        try:
            summary = rebuild(
                knowledge_dir=args.knowledge_dir,
                chroma_dir=args.chroma_dir,
                collection_name=args.collection,
            )
        except EmbeddingNotConfigured as exc:
            print(f"[ingest] 构建失败（embedding_not_configured）：{exc}", file=sys.stderr)
            print(
                "[ingest] 请先配置 MODEL_API_BASE / MODEL_API_KEY / EMBEDDING_MODEL"
                "（可复制 .env.example 为 .env 后填写）。未配置 embedding 时系统只走 BM25，"
                "不会发起任何网络请求。",
                file=sys.stderr,
            )
            return 2
        except Exception as exc:
            print(f"[ingest] 构建失败（embedding_call_failed，{type(exc).__name__}）：{exc}", file=sys.stderr)
            print(
                f"[ingest] 请检查 MODEL_API_BASE 与 EMBEDDING_MODEL（当前为 {settings.embedding_model}）"
                "是否正确，以及该向量模型是否已在方舟控制台开通、是否支持文本向量化接口。"
                "本次未写入任何索引；检索仍会退回 BM25。",
                file=sys.stderr,
            )
            return 1
        print(f"[ingest] 索引已写入：集合={summary['collection']} 目录={summary['chroma_dir']}")
        print(
            f"[ingest] 知识条目 {summary['documents']} 条 → 文本块 {summary['chunks']} 块"
            f"（本次 upsert {summary['upserted']}，清理历史遗留 {summary['pruned']}）"
        )
        print(f"[ingest] 集合当前条目数：{summary['count']}")

    if args.status:
        info = status(chroma_dir=args.chroma_dir, collection_name=args.collection)
        if not info["exists"]:
            print(f"[ingest] 索引目录不存在：{info['chroma_dir']}（尚未构建，检索会退回 BM25）")
        elif info["error"]:
            print(f"[ingest] 读取集合失败：{info['error']}")
        else:
            print(f"[ingest] 索引现状：集合={info['collection']} 条目数={info['count']}")
            print(f"[ingest] 块 ID（{len(info['chunk_ids'])} 个）：{', '.join(info['chunk_ids'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())