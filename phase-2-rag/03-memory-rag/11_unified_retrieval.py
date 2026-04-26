"""
11_unified_retrieval.py — MQE + HyDE 统一检索框架

学习目标：
1. 将多查询扩展（MQE）和假设文档嵌入（HyDE）统一到一个检索函数
2. 实现候选池倍增器（candidate pool multiplier）提升召回率
3. 实现跨查询去重与最高分保留的融合排序
4. 理解中英文混合 token 估算方法
5. 实现嵌入降级链：API → 本地模型 → TF-IDF

核心概念：
- MQE：同一问题的多种表述，覆盖更多相关文档
- HyDE：用假设答案去检索，缩小查询与文档的语义鸿沟
- 统一框架：扩展 → 检索 → 去重 → 排序，一个函数搞定

统一检索流程：
    用户查询
       │
       ├──▶ MQE 扩展（LLM 生成 N 个变体）
       │       ├── 变体1 ──▶ 向量检索 ──┐
       │       ├── 变体2 ──▶ 向量检索 ──┤
       │       └── 变体3 ──▶ 向量检索 ──┤
       │                                 │
       ├──▶ HyDE（LLM 生成假设答案）     │
       │       └── 假设文档 ──▶ 向量检索 ─┤
       │                                 │
       └──▶ 原始查询 ──▶ 向量检索 ───────┤
                                         ▼
                                    候选池合并
                                         │
                                    去重 + 最高分保留
                                         │
                                    top_k 结果

运行方式：
    python 11_unified_retrieval.py
"""

import os
from typing import Optional

import chromadb
import litellm
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from sentence_transformers import SentenceTransformer

load_dotenv()

console = Console()
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek/deepseek-chat")


# ============================================================
# 1. 工具函数
# ============================================================

def call_llm(prompt: str, system: str = "") -> str:
    """调用 LLM（通过 litellm 统一接口）"""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    try:
        resp = litellm.completion(model=LLM_MODEL, messages=messages,
                                  temperature=0.7, max_tokens=512)
        return resp.choices[0].message.content.strip()
    except Exception as e:
        console.print(f"[yellow]LLM 调用失败: {e}[/yellow]")
        return ""


def estimate_tokens_mixed(text: str) -> int:
    """中英文混合 token 估算

    - CJK 字符：每个约 1 token
    - 英文/数字：按空格分词，每词约 1.3 token
    """
    cjk_count = sum(1 for ch in text if _is_cjk(ch))
    non_cjk = "".join(ch if not _is_cjk(ch) else " " for ch in text)
    word_count = len(non_cjk.split())
    return cjk_count + int(word_count * 1.3)


def _is_cjk(ch: str) -> bool:
    code = ord(ch)
    return (0x4E00 <= code <= 0x9FFF or 0x3400 <= code <= 0x4DBF
            or 0x20000 <= code <= 0x2A6DF or 0xF900 <= code <= 0xFAFF)


# ============================================================
# 2. 嵌入降级链
# ============================================================

class EmbeddingWithFallback:
    """嵌入降级链：本地 sentence-transformers → TF-IDF

    生产环境可在最前面加 API 嵌入（如百炼、OpenAI），
    这里为了教学简化，只演示本地模型和 TF-IDF 两级。
    """

    def __init__(self):
        self._local_model = None
        self._tfidf_vec = None
        self._tfidf_corpus: list[str] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        result = self._try_local(texts)
        if result is not None:
            return result
        return self._try_tfidf(texts)

    def embed_query(self, query: str) -> list[float]:
        return self.embed([query])[0]

    def _try_local(self, texts: list[str]) -> Optional[list[list[float]]]:
        try:
            if self._local_model is None:
                self._local_model = SentenceTransformer("all-MiniLM-L6-v2")
            vecs = self._local_model.encode(texts)
            return [v.tolist() for v in vecs]
        except Exception:
            return None

    def _try_tfidf(self, texts: list[str]) -> list[list[float]]:
        from sklearn.feature_extraction.text import TfidfVectorizer
        if self._tfidf_vec is None:
            self._tfidf_vec = TfidfVectorizer(max_features=384)
        all_texts = self._tfidf_corpus + texts
        matrix = self._tfidf_vec.fit_transform(all_texts)
        vecs = matrix[-len(texts):].toarray()
        return [v.tolist() for v in vecs]


# ============================================================
# 3. 统一检索器
# ============================================================

class UnifiedRetriever:
    """统一检索器 — MQE + HyDE + 候选池融合

    参数：
    - collection: ChromaDB 集合
    - use_mqe: 是否启用多查询扩展
    - use_hyde: 是否启用假设文档嵌入
    - pool_multiplier: 候选池倍增器（每个查询检索 top_k × multiplier）
    """

    def __init__(self, collection: chromadb.Collection,
                 embedder: Optional[EmbeddingWithFallback] = None):
        self.collection = collection
        self.embedder = embedder or EmbeddingWithFallback()

    def search(self, query: str, top_k: int = 5,
               use_mqe: bool = True, use_hyde: bool = True,
               mqe_count: int = 2, pool_multiplier: int = 3,
               ) -> list[dict]:
        """统一检索入口"""
        if self.collection.count() == 0:
            return []

        # 构建所有查询变体
        queries = [query]
        if use_mqe:
            queries.extend(self._expand_queries(query, n=mqe_count))
        if use_hyde:
            hyde_doc = self._generate_hypothetical_doc(query)
            if hyde_doc:
                queries.append(hyde_doc)

        # 去重
        seen = set()
        unique_queries = []
        for q in queries:
            if q and q not in seen:
                seen.add(q)
                unique_queries.append(q)

        # 每个查询检索候选
        per_query_k = max(top_k * pool_multiplier, 10)
        all_candidates: dict[str, dict] = {}

        for q in unique_queries:
            hits = self._search_single(q, per_query_k)
            for hit in hits:
                doc_id = hit["id"]
                if doc_id not in all_candidates or hit["score"] > all_candidates[doc_id]["score"]:
                    all_candidates[doc_id] = hit

        # 按分数排序返回
        results = sorted(all_candidates.values(),
                         key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def _search_single(self, query: str, top_k: int) -> list[dict]:
        q_emb = self.embedder.embed_query(query)
        n = min(top_k, self.collection.count())
        results = self.collection.query(
            query_embeddings=[q_emb], n_results=n,
        )
        hits = []
        for i, doc_id in enumerate(results["ids"][0]):
            dist = results["distances"][0][i] if results["distances"] else 0
            hits.append({
                "id": doc_id,
                "content": results["documents"][0][i],
                "score": max(0, 1 - dist),
                "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
            })
        return hits

    def _expand_queries(self, query: str, n: int = 2) -> list[str]:
        prompt = (f"原始查询：{query}\n"
                  f"请给出 {n} 个语义相同但表述不同的查询，每行一个，不要编号：")
        text = call_llm(prompt, system="你是检索查询扩展助手。生成简短的中文查询变体。")
        if not text:
            return []
        lines = [ln.strip("- \t·•") for ln in text.splitlines() if ln.strip()]
        return lines[:n]

    def _generate_hypothetical_doc(self, query: str) -> str:
        prompt = (f"问题：{query}\n"
                  "请直接写一段简短的答案性段落（50-100字），包含关键术语：")
        return call_llm(prompt, system="根据问题写一段可能的答案，用于辅助检索。")


# ============================================================
# 4. Demo
# ============================================================

SAMPLE_DOCS = [
    "RAG（检索增强生成）是一种结合信息检索和文本生成的 AI 技术，通过检索外部知识来增强大语言模型的回答准确性。",
    "ChromaDB 是一个开源的向量数据库，专为 AI 应用设计，支持高效的相似度搜索和元数据过滤。",
    "BM25 是经典的稀疏检索算法，基于词频和逆文档频率计算文档相关性，适合精确的关键词匹配。",
    "HyDE（假设文档嵌入）通过让 LLM 生成假设性答案，用答案的向量去检索真实文档，缩小查询与文档的语义鸿沟。",
    "多查询扩展（MQE）将用户的一个问题改写为多个不同角度的查询，扩大检索覆盖面，提升召回率。",
    "向量检索通过将文本转换为高维向量，利用余弦相似度等度量找到语义最相近的文档。",
    "混合检索结合稀疏检索（BM25）和稠密检索（向量），通过 RRF 等融合算法取长补短。",
    "RAGAS 是一个 RAG 评估框架，提供忠实度、答案相关性、上下文精确度等指标。",
    "Cross-Encoder 重排序模型对检索结果进行精细化排序，显著提升检索精度。",
    "LangChain 提供了丰富的文本分块策略，包括固定大小、递归字符、Markdown 感知等。",
]


def _build_demo_collection() -> chromadb.Collection:
    embedder = EmbeddingWithFallback()
    client = chromadb.Client()
    try:
        client.delete_collection("unified_demo")
    except Exception:
        pass
    col = client.create_collection("unified_demo",
                                   metadata={"hnsw:space": "cosine"})
    embeddings = embedder.embed(SAMPLE_DOCS)
    col.add(
        ids=[f"doc_{i}" for i in range(len(SAMPLE_DOCS))],
        embeddings=embeddings,
        documents=SAMPLE_DOCS,
        metadatas=[{"index": i} for i in range(len(SAMPLE_DOCS))],
    )
    return col


def demo_token_estimation():
    console.print(Panel("[bold]Demo 1: 中英文混合 Token 估算[/bold]"))

    texts = [
        "RAG 是检索增强生成技术",
        "Retrieval-Augmented Generation is a powerful technique",
        "RAG（检索增强生成）结合了 information retrieval 和 text generation",
    ]
    table = Table(title="Token 估算")
    table.add_column("文本", max_width=50)
    table.add_column("字符数", width=8)
    table.add_column("估算 Token", width=10)
    for t in texts:
        table.add_row(t, str(len(t)), str(estimate_tokens_mixed(t)))
    console.print(table)


def demo_unified_retrieval():
    console.print(Panel("[bold]Demo 2: 统一检索 vs 普通检索[/bold]"))

    col = _build_demo_collection()
    retriever = UnifiedRetriever(col)
    query = "如何提升 RAG 系统的检索效果"

    plain_results = retriever.search(query, top_k=3,
                                     use_mqe=False, use_hyde=False)
    mqe_results = retriever.search(query, top_k=3,
                                   use_mqe=True, use_hyde=False)
    hyde_results = retriever.search(query, top_k=3,
                                   use_mqe=False, use_hyde=True)
    unified_results = retriever.search(query, top_k=3,
                                       use_mqe=True, use_hyde=True)

    for label, results in [("普通检索", plain_results),
                           ("MQE", mqe_results),
                           ("HyDE", hyde_results),
                           ("统一(MQE+HyDE)", unified_results)]:
        table = Table(title=f"{label} — query: {query}")
        table.add_column("#", width=3)
        table.add_column("内容", max_width=50)
        table.add_column("得分", width=8)
        for i, r in enumerate(results, 1):
            table.add_row(str(i), r["content"][:50] + "...",
                          f"{r['score']:.3f}")
        console.print(table)


def demo_fallback_embedding():
    console.print(Panel("[bold]Demo 3: 嵌入降级链[/bold]"))

    embedder = EmbeddingWithFallback()
    texts = ["RAG 检索增强生成", "向量数据库"]
    vecs = embedder.embed(texts)
    console.print(f"  嵌入维度: {len(vecs[0])}")
    console.print(f"  文本数量: {len(vecs)}")

    import numpy as np
    v1, v2 = np.array(vecs[0]), np.array(vecs[1])
    sim = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    console.print(f"  '{texts[0]}' vs '{texts[1]}' 相似度: {sim:.3f}")


if __name__ == "__main__":
    console.print(Panel(
        "[bold]11 — MQE + HyDE 统一检索框架[/bold]\n"
        "多查询扩展 | 假设文档嵌入 | 候选池融合 | Token 估算",
        style="blue",
    ))
    demo_token_estimation()
    demo_fallback_embedding()
    demo_unified_retrieval()

