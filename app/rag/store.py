"""知识库加载与检索索引（向量 + BM25 混合优先，不可用时自动降级 BM25）。

三种检索模式：
1. 有 embedding 配置且 Chroma 持久化索引可用 → `hybrid`：Chroma 向量检索 + BM25 混合排序；
2. 没有 embedding 配置 → `bm25`：纯 BM25，不发起任何网络请求；
3. 向量服务 / embedding 调用异常 → 降级 `bm25`，并记录降级原因（`degraded` / `degraded_reason`）。

混合分数公式：`final_score = rag_vector_weight * vector_score + rag_bm25_weight * bm25_score`
（默认 0.6 / 0.4，可用环境变量覆盖；两种分数都归一化到 [0, 1]，再由 rag_score_threshold 过滤）。
"""

from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from app.config import settings

_BM25_K1 = 1.5
_BM25_B = 0.75
# 词法相似度整体小于向量余弦，统一放大到与 embedding 余弦可比的量级，使 rag_score_threshold 对两条分支都成立
_BM25_SCALE = 2.5


# ---------------------------------------------------------------------------
# 极简 front-matter 解析（仅支持 key: value、缩进 "- " 列表、sources 的 "a: x | b: y"）
# ---------------------------------------------------------------------------
def _parse_list_item(item: str):
    """把 `key: value | key: value` 形式的列表项解析成字段字典，其余原样返回。

    同时覆盖 sources（name/url/locator）与 criteria（id/claim/metric/op/value）两类条目。
    """
    if "|" not in item:
        return item
    fields = {}
    for seg in item.split("|"):
        key, _, value = seg.partition(":")
        if value.strip():
            fields[key.strip()] = value.strip()
    return fields


def _parse_front_matter(text: str) -> tuple[dict, str]:
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, text.strip()
    meta: dict = {}
    current = None
    end = len(lines)
    for i in range(1, len(lines)):
        line = lines[i]
        if line.strip() == "---":
            end = i
            break
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("- ") and current:
            bucket = meta.get(current)
            if isinstance(bucket, list):
                bucket.append(_parse_list_item(stripped[2:].strip()))
            continue
        if ":" in stripped:
            key, _, value = stripped.partition(":")
            key, value = key.strip(), value.strip()
            meta[key] = value if value else []
            current = key
    return meta, "\n".join(lines[end + 1:]).strip()


# ---------------------------------------------------------------------------
# 简易中文/英文 tokenizer：英文整词小写、小数整体保留，中文只切相邻双字
# （单字几乎在所有条目里都出现，只会稀释检索信号，故不保留）
# ---------------------------------------------------------------------------
_CHUNK_RE = re.compile(r"[A-Za-z]+|\d+(?:\.\d+)?|[\u4e00-\u9fff]+")


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for chunk in _CHUNK_RE.findall(text or ""):
        if chunk[0].isascii():
            tokens.append(chunk.lower())
        else:
            tokens.extend(chunk[i:i + 2] for i in range(len(chunk) - 1))
    return tokens


def load_knowledge(knowledge_dir: Path) -> list[dict]:
    """解析 knowledge_dir 下的 *.md，返回条目字典列表。

    criteria 为具名判据条款：每条含 id / claim（中文判据）/ metric / op / value，
    另有可选的 family（特征频率族）与 order（转频倍次）；value 保持字符串，由核对引擎按 op 转换。
    """
    knowledge_dir = Path(knowledge_dir)
    entries: list[dict] = []
    for path in sorted(knowledge_dir.glob("*.md")):
        meta, body = _parse_front_matter(path.read_text(encoding="utf-8"))
        if not meta.get("fault_type"):
            continue
        typical = meta.get("typical_features") or []
        sources = meta.get("sources") or []
        criteria = []
        for item in meta.get("criteria") or []:
            if not isinstance(item, dict) or not item.get("id") or not item.get("metric"):
                continue
            criteria.append({
                "id": item["id"],
                "claim": item.get("claim", ""),
                "metric": item["metric"],
                "op": (item.get("op") or "==").strip().strip("\"'"),
                "value": item.get("value", ""),
                "family": (item.get("family") or "").strip().upper() or None,
                "order": item.get("order"),
            })
        entry = {
            "fault_type": meta.get("fault_type", ""),
            "title": meta.get("title", ""),
            "applicable": meta.get("applicable", ""),
            "typical_features": [t for t in typical if isinstance(t, str)],
            "frequency_signature": meta.get("frequency_signature", ""),
            "frequency_family": (meta.get("frequency_family") or "").strip().upper() or None,
            "criteria": criteria,
            "sources": [s for s in sources if isinstance(s, dict)],
            "document": path.name,
            "body": body,
        }
        entries.append(entry)
    return entries


def _lexical_text(entry: dict) -> str:
    """词法索引文本：只取有区分度的结构化字段。"""
    return " ".join(
        [entry["fault_type"], entry["title"], entry["applicable"]]
        + entry["typical_features"]
        + [entry["frequency_signature"]]
    )


# ---------------------------------------------------------------------------
# BM25 词法打分器：idf × 饱和项，再做词权重点积的余弦归一后放大到 [0, 1]
# ---------------------------------------------------------------------------
class _Bm25Index:
    """对一组文档建立 BM25 词权重索引，`scores(query)` 返回归一化到 [0, 1] 的分数。"""

    def __init__(self, docs: list[str]):
        self.doc_count = len(docs)
        # epsilon=0：在全部文档中都出现的词 idf 记 0，避免通用词靠数量堆积压过区分性术语
        self._bm25 = BM25Okapi([tokenize(d) for d in docs], epsilon=0.0)
        self._avgdl = sum(self._bm25.doc_len) / max(self.doc_count, 1)
        self._doc_vectors = self._weight_vectors(tokenize(d) for d in docs)

    def _weight_vector(self, tokens: list[str], length: float) -> dict[str, float]:
        """BM25 词权重（idf × 饱和项），idf ≤ 0 的词直接丢弃。"""
        vec = {}
        for token, freq in Counter(tokens).items():
            idf = self._bm25.idf.get(token, 0.0)
            if idf <= 0:
                continue
            vec[token] = idf * (freq * (_BM25_K1 + 1)) / (
                freq + _BM25_K1 * (1 - _BM25_B + _BM25_B * length / self._avgdl)
            )
        return vec

    def _weight_vectors(self, token_lists) -> list[tuple[dict, float]]:
        vectors = []
        for tokens in token_lists:
            vec = self._weight_vector(tokens, len(tokens))
            vectors.append((vec, math.sqrt(sum(w * w for w in vec.values()))))
        return vectors

    def scores(self, query: str) -> np.ndarray:
        if self.doc_count == 0:
            return np.zeros(0)
        qvec = self._weight_vector(tokenize(query), self._avgdl)
        qnorm = math.sqrt(sum(w * w for w in qvec.values()))
        if qnorm == 0:
            return np.zeros(self.doc_count)
        scores = np.zeros(self.doc_count)
        for i, (vec, norm) in enumerate(self._doc_vectors):
            if norm == 0:
                continue
            dot = sum(w * vec[t] for t, w in qvec.items() if t in vec)
            scores[i] = dot / (qnorm * norm) * _BM25_SCALE
        return np.clip(scores, 0.0, 1.0)


class KnowledgeStore:
    """知识库索引：优先「Chroma 向量 + BM25」混合检索，不可用时降级 BM25。

    - `retrieval_mode` 取值 `"hybrid"` 或 `"bm25"`（`/health` 直接读它）。
    - 无 embedding 配置时只走 BM25，**不发起任何网络请求**。
    - embedding 注入点：`KnowledgeStore(embedder=...)`，可离线跑通向量写入/查询。
    - 不在导入或构造时连接 Chroma；`data/chroma/` 不存在时正常启动并退回 BM25。
    """

    def __init__(
        self,
        knowledge_dir: Path | None = None,
        chroma_dir: Path | None = None,
        embedder=None,
        collection_name: str | None = None,
    ):
        self.knowledge_dir = Path(knowledge_dir) if knowledge_dir else settings.knowledge_dir
        self.entries = load_knowledge(self.knowledge_dir)
        # 条目级词法索引：降级路径与无 embedding 配置时的唯一打分依据
        self._entries_index = _Bm25Index([_lexical_text(e) for e in self.entries])

        # 条款级索引：把每个条目的具名判据条款单独索引，检索单元下沉到条款，
        # 用于挑选该条目内与本次查询最相关的判据文本（不参与条目排序打分）。
        self._entry_clause_span: list[tuple[int, int]] = []
        clause_texts: list[str] = []
        for entry in self.entries:
            start = len(clause_texts)
            clause_texts.extend(str(item.get("claim") or "") for item in entry.get("criteria") or [])
            self._entry_clause_span.append((start, len(clause_texts)))
        self._clause_index = _Bm25Index(clause_texts) if clause_texts else None

        self.chroma_dir = Path(chroma_dir) if chroma_dir else settings.chroma_dir
        self.collection_name = collection_name or settings.chroma_collection
        self._embedder = embedder
        self._embedder_provided = embedder is not None
        self._collection = None
        self._chunks: list[dict] = []
        self._chunks_index: _Bm25Index | None = None

        self._retrieval_mode = "bm25"
        self._degraded = False
        self._degraded_reason: str | None = None

        if self._embedder_provided or (settings.enable_hybrid_retrieval and settings.embedding_available):
            try:
                self._activate_hybrid()
            except Exception as exc:
                self._degrade(f"向量索引不可用，已退回 BM25：{type(exc).__name__}: {exc}")

    # -- 对外状态 ----------------------------------------------------------
    @property
    def retrieval_mode(self) -> str:
        return self._retrieval_mode

    @property
    def degraded(self) -> bool:
        """本该用混合检索（有 embedding 配置）但已降级为 BM25。"""
        return self._degraded

    @property
    def degraded_reason(self) -> str | None:
        return self._degraded_reason

    def _degrade(self, reason: str) -> None:
        self._retrieval_mode = "bm25"
        self._degraded = True
        self._degraded_reason = reason

    # -- 混合检索激活 ------------------------------------------------------
    def _activate_hybrid(self) -> None:
        from app.rag import ingest  # 延迟导入，避免 store / ingest 模块级循环依赖

        self._chunks = ingest.build_chunks(self.entries, self.knowledge_dir)
        if not self._chunks:
            raise RuntimeError(f"知识库为空：{self.knowledge_dir} 下没有可索引的文本块")
        self._chunks_index = _Bm25Index([chunk["text"] for chunk in self._chunks])

        if self._embedder is None:
            self._embedder = ingest.build_embedder()
            if self._embedder is None:
                raise RuntimeError(
                    "未配置 embedding 服务（MODEL_API_BASE / MODEL_API_KEY 为空），无法启用混合检索"
                )

        self._collection = ingest.open_collection(self.chroma_dir, self.collection_name)
        if self._collection.count() <= 0:
            raise RuntimeError(
                f"向量索引为空：{self.chroma_dir} 下集合 {self.collection_name} 尚无条目，"
                "请先执行 python -m app.rag.ingest --rebuild"
            )
        self._retrieval_mode = "hybrid"

    def _embed_texts(self, texts: list[str]) -> np.ndarray:
        """调用注入的 embedder 并做 L2 归一化。embedding 异常会向上抛出（由调用方降级）。"""
        if self._embedder is None:
            raise RuntimeError("没有可用的 embedding 回调")
        vectors = np.asarray(self._embedder(list(texts)), dtype=float)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vectors / norms

    def _chunk_vector_scores(self, query: str) -> np.ndarray:
        """用 Chroma 查询各文本块的向量相似度，归一化到 [0, 1]（未命中的块记 0）。"""
        scores = np.zeros(len(self._chunks))
        query_vector = self._embed_texts([query])[0]
        n_results = min(len(self._chunks), int(self._collection.count()))
        if n_results <= 0:
            return scores
        result = self._collection.query(
            query_embeddings=[query_vector.tolist()],
            n_results=n_results,
            include=["metadatas", "distances"],
        )
        index_by_id = {chunk["chunk_id"]: i for i, chunk in enumerate(self._chunks)}
        ids = (result.get("ids") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        for chunk_id, distance in zip(ids, distances):
            i = index_by_id.get(chunk_id)
            if i is None:
                continue
            # 集合按 cosine 距离建立，相似度 = 1 - distance，再裁到 [0, 1]
            scores[i] = float(np.clip(1.0 - float(distance), 0.0, 1.0))
        return scores

    # -- 打分与检索 --------------------------------------------------------
    def _clause_scores(self, query: str) -> np.ndarray:
        """条款级 BM25 分数（顺序与各条目 criteria 展开后的位置一致）。"""
        if self._clause_index is None:
            return np.zeros(0)
        return self._clause_index.scores(query)

    def _clause_hits(self, entry_index: int, clause_scores: np.ndarray, limit: int = 2) -> list[dict]:
        """该条目内与本次查询最相关的判据条款（条款级 BM25 取前 limit 条，0 分不取）。"""
        entry = self.entries[entry_index] if 0 <= entry_index < len(self.entries) else {}
        criteria = entry.get("criteria") or []
        if not criteria or clause_scores.size == 0:
            return []
        start, _ = self._entry_clause_span[entry_index]
        scored = [
            (float(clause_scores[start + offset]), item) for offset, item in enumerate(criteria)
        ]
        scored = [(score, item) for score, item in scored if score > 0]
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [
            {"id": item["id"], "claim": item.get("claim", ""), "score": round(score, 4)}
            for score, item in scored[:limit]
        ]

    def _pick_evidence(
        self, entry: dict, query: str, clause_hits: list[dict] | None = None
    ) -> str:
        """条款级证据：优先给该条目内最相关的判据条款，不足两条时用典型特征补齐。"""
        picked = [
            f"[{item['id']}] {item['claim']}" for item in (clause_hits or []) if item.get("claim")
        ]
        feats = entry.get("typical_features") or []
        if len(picked) < 2:
            q_tokens = set(tokenize(query))
            scored = []
            for feat in feats:
                f_tokens = set(tokenize(feat))
                overlap = len(q_tokens & f_tokens) / (len(f_tokens) or 1)
                scored.append((overlap, feat))
            scored.sort(key=lambda pair: pair[0], reverse=True)
            picked += [feat for overlap, feat in scored if overlap > 0][: 2 - len(picked)]
        if not picked:
            return entry.get("frequency_signature", "")
        return "；".join(picked)

    def _bm25_search(self, query: str, top_k: int) -> list[dict]:
        scores = self._entries_index.scores(query)
        clause_scores = self._clause_scores(query)
        order = sorted(range(len(self.entries)), key=lambda i: scores[i], reverse=True)
        results = []
        for i in order[:top_k]:
            entry = self.entries[i]
            source = entry["sources"][0] if entry["sources"] else {}
            clause_hits = self._clause_hits(i, clause_scores)
            results.append({
                "fault_type": entry["fault_type"],
                "title": entry["title"],
                "evidence": self._pick_evidence(entry, query, clause_hits),
                "source": source.get("name", ""),
                "source_url": source.get("url", ""),
                "locator": source.get("locator", ""),
                "frequency_family": entry.get("frequency_family"),
                "criteria": entry.get("criteria") or [],
                "clause_hits": clause_hits,
                "score": round(float(scores[i]), 4),
                "retrieval": "bm25",
            })
        return results

    def _hybrid_search(self, query: str, top_k: int) -> list[dict]:
        vector_scores = self._chunk_vector_scores(query)
        bm25_scores = self._chunks_index.scores(query)
        clause_scores = self._clause_scores(query)
        final_scores = np.clip(
            settings.rag_vector_weight * vector_scores + settings.rag_bm25_weight * bm25_scores,
            0.0,
            1.0,
        )
        # 每个条目只保留得分最高的那个文本块（避免同一故障类型占满候选位）
        best_chunk: dict[int, int] = {}
        for i, chunk in enumerate(self._chunks):
            entry_index = chunk["entry_index"]
            if entry_index not in best_chunk or final_scores[i] > final_scores[best_chunk[entry_index]]:
                best_chunk[entry_index] = i
        order = sorted(best_chunk.values(), key=lambda i: final_scores[i], reverse=True)[:top_k]

        results = []
        for i in order:
            chunk = self._chunks[i]
            entry = self.entries[chunk["entry_index"]]
            clause_hits = self._clause_hits(chunk["entry_index"], clause_scores)
            results.append({
                "fault_type": entry["fault_type"],
                "title": entry["title"],
                "evidence": self._pick_evidence(entry, query, clause_hits),
                "source": chunk["source"],
                "source_url": chunk["source_url"],
                "locator": chunk["locator"],
                "frequency_family": entry.get("frequency_family"),
                "criteria": entry.get("criteria") or [],
                "clause_hits": clause_hits,
                "score": round(float(final_scores[i]), 4),
                "retrieval": "hybrid",
                "chunk_id": chunk["chunk_id"],
                "document": chunk["document"],
                "vector_score": round(float(vector_scores[i]), 4),
                "bm25_score": round(float(bm25_scores[i]), 4),
            })
        return results

    def search(self, query: str, top_k: int | None = None) -> list[dict]:
        top_k = top_k or settings.rag_top_k
        if self._retrieval_mode == "hybrid":
            try:
                return self._hybrid_search(query, top_k)
            except Exception as exc:
                self._degrade(
                    f"embedding 调用或向量查询异常，已降级 BM25：{type(exc).__name__}: {exc}"
                )
        return self._bm25_search(query, top_k)
