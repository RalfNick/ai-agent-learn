"""
09_memory_system.py — 认知科学启发的四类记忆系统

学习目标：
1. 理解人类记忆系统的层次结构（工作记忆、情景记忆、语义记忆）
2. 实现 WorkingMemory：纯内存 + TTL + 容量限制 + TF-IDF 混合检索
3. 实现 EpisodicMemory：SQLite 持久化 + ChromaDB 向量检索 + 会话追踪
4. 实现 SemanticMemory：实体/关系抽取 + 向量检索 + 知识图谱查询
5. 理解不同记忆类型的适用场景和评分机制

核心概念：
- 工作记忆（Working Memory）：短期、高速、容量有限，类似 CPU 缓存
- 情景记忆（Episodic Memory）：记录具体事件和经历，带时间戳
- 语义记忆（Semantic Memory）：抽象知识和概念，实体-关系结构

记忆系统架构：
    ┌─────────────────────────────────────────────────────┐
    │                   MemoryItem（统一数据结构）           │
    └──────────────────────┬──────────────────────────────┘
                           │
           ┌───────────────┼───────────────┐
           ▼               ▼               ▼
    ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
    │ WorkingMemory│ │EpisodicMemory│ │SemanticMemory│
    │  纯内存+TTL  │ │ SQLite+向量  │ │ 实体图谱+向量│
    │  TF-IDF检索  │ │ 会话追踪     │ │ 关系推理     │
    └─────────────┘ └─────────────┘ └─────────────┘

运行方式：
    python 09_memory_system.py
"""

import math
import os
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import chromadb
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from sentence_transformers import SentenceTransformer

console = Console()


# ============================================================
# 1. 统一数据结构：MemoryItem
# ============================================================

@dataclass
class MemoryItem:
    """统一的记忆数据结构，所有记忆类型共用"""
    content: str
    memory_type: str = "working"          # working / episodic / semantic
    importance: float = 0.5               # 0.0 ~ 1.0
    timestamp: datetime = field(default_factory=datetime.now)
    memory_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    metadata: dict[str, Any] = field(default_factory=dict)
    access_count: int = 0
    session_id: str = "default"


# ============================================================
# 2. 工作记忆：纯内存 + TTL + 容量限制 + TF-IDF 混合检索
# ============================================================

class WorkingMemory:
    """工作记忆 — 短期、高速、容量有限

    特点：
    - 纯内存存储，重启后丢失（类似 CPU 缓存）
    - TTL 自动过期（默认 60 分钟）
    - 容量限制（默认 50 条），超出时淘汰最低优先级
    - TF-IDF + 关键词混合检索
    """

    def __init__(self, capacity: int = 50, ttl_minutes: int = 60):
        self.capacity = capacity
        self.ttl_minutes = ttl_minutes
        self.memories: list[MemoryItem] = []
        self._tfidf = None

    def add(self, content: str, importance: float = 0.5,
            session_id: str = "default", **metadata) -> str:
        self._evict_expired()
        if len(self.memories) >= self.capacity:
            self._evict_lowest_priority()

        item = MemoryItem(
            content=content, memory_type="working",
            importance=importance, session_id=session_id,
            metadata=metadata,
        )
        self.memories.append(item)
        self._tfidf = None  # 缓存失效
        return item.memory_id

    def retrieve(self, query: str, top_k: int = 5) -> list[tuple[MemoryItem, float]]:
        """混合检索：TF-IDF 语义 + 关键词匹配 + 时间衰减 + 重要性"""
        self._evict_expired()
        if not self.memories:
            return []

        tfidf_scores = self._tfidf_scores(query)
        scored = []
        for i, mem in enumerate(self.memories):
            tfidf_sim = tfidf_scores.get(i, 0.0)
            kw_score = self._keyword_score(query, mem.content)
            base = tfidf_sim * 0.7 + kw_score * 0.3 if tfidf_sim > 0 else kw_score
            decay = self._time_decay(mem.timestamp)
            imp_weight = 0.8 + mem.importance * 0.4
            final = base * decay * imp_weight
            if final > 0:
                scored.append((mem, final))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    # --- 内部方法 ---

    def _evict_expired(self):
        cutoff = datetime.now() - timedelta(minutes=self.ttl_minutes)
        self.memories = [m for m in self.memories if m.timestamp > cutoff]

    def _evict_lowest_priority(self):
        if not self.memories:
            return
        worst = min(self.memories, key=lambda m: m.importance)
        self.memories.remove(worst)

    def _tfidf_scores(self, query: str) -> dict[int, float]:
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
        except ImportError:
            return {}
        corpus = [m.content for m in self.memories] + [query]
        vec = TfidfVectorizer()
        matrix = vec.fit_transform(corpus)
        sims = cosine_similarity(matrix[-1:], matrix[:-1])[0]
        return {i: float(s) for i, s in enumerate(sims)}

    @staticmethod
    def _keyword_score(query: str, content: str) -> float:
        q_tokens = set(query.lower().split())
        c_tokens = set(content.lower().split())
        if not q_tokens:
            return 0.0
        return len(q_tokens & c_tokens) / len(q_tokens)

    @staticmethod
    def _time_decay(ts: datetime, half_life_hours: float = 1.0) -> float:
        elapsed = (datetime.now() - ts).total_seconds() / 3600
        return math.exp(-0.693 * elapsed / half_life_hours)

    def size(self) -> int:
        return len(self.memories)

    def clear(self):
        self.memories.clear()
        self._tfidf = None


# ============================================================
# 3. 情景记忆：SQLite + ChromaDB + 会话追踪
# ============================================================

class EpisodicMemory:
    """情景记忆 — 记录具体事件和经历

    特点：
    - SQLite 持久化元数据，ChromaDB 存储向量
    - 按 session_id 追踪会话历史
    - 评分公式：(向量相似度 × 0.8 + 时间近因性 × 0.2) × 重要性权重
    """

    def __init__(self, db_path: str = "./memory_db/episodic.db",
                 collection_name: str = "episodic_memory"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_db()
        self._chroma = chromadb.Client()
        self._collection = self._chroma.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self._embedder = SentenceTransformer("all-MiniLM-L6-v2")

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS episodes (
                memory_id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                importance REAL DEFAULT 0.5,
                timestamp TEXT NOT NULL,
                session_id TEXT DEFAULT 'default',
                access_count INTEGER DEFAULT 0,
                metadata TEXT DEFAULT '{}'
            )
        """)
        conn.commit()
        conn.close()

    def add(self, content: str, importance: float = 0.5,
            session_id: str = "default", **metadata) -> str:
        item = MemoryItem(
            content=content, memory_type="episodic",
            importance=importance, session_id=session_id,
            metadata=metadata,
        )
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "INSERT INTO episodes VALUES (?, ?, ?, ?, ?, ?, ?)",
            (item.memory_id, item.content, item.importance,
             item.timestamp.isoformat(), item.session_id,
             item.access_count, str(item.metadata)),
        )
        conn.commit()
        conn.close()

        embedding = self._embedder.encode(content).tolist()
        self._collection.add(
            ids=[item.memory_id],
            embeddings=[embedding],
            documents=[content],
            metadatas=[{
                "importance": importance,
                "timestamp": item.timestamp.isoformat(),
                "session_id": session_id,
            }],
        )
        return item.memory_id

    def retrieve(self, query: str, top_k: int = 5) -> list[tuple[MemoryItem, float]]:
        if self._collection.count() == 0:
            return []
        q_emb = self._embedder.encode(query).tolist()
        results = self._collection.query(
            query_embeddings=[q_emb],
            n_results=min(top_k * 3, self._collection.count()),
        )
        scored = []
        for i, mid in enumerate(results["ids"][0]):
            dist = results["distances"][0][i] if results["distances"] else 0
            vec_sim = max(0, 1 - dist)
            meta = results["metadatas"][0][i]
            recency = self._recency_score(meta.get("timestamp", ""))
            importance = meta.get("importance", 0.5)
            base = vec_sim * 0.8 + recency * 0.2
            imp_weight = 0.8 + importance * 0.4
            final = base * imp_weight
            item = MemoryItem(
                content=results["documents"][0][i],
                memory_type="episodic",
                importance=importance,
                memory_id=mid,
                session_id=meta.get("session_id", "default"),
            )
            scored.append((item, final))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def get_session_history(self, session_id: str) -> list[MemoryItem]:
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute(
            "SELECT memory_id, content, importance, timestamp, session_id "
            "FROM episodes WHERE session_id = ? ORDER BY timestamp",
            (session_id,),
        ).fetchall()
        conn.close()
        return [
            MemoryItem(content=r[1], memory_type="episodic",
                       importance=r[2], memory_id=r[0], session_id=r[4])
            for r in rows
        ]

    @staticmethod
    def _recency_score(ts_str: str) -> float:
        try:
            ts = datetime.fromisoformat(ts_str)
            hours = (datetime.now() - ts).total_seconds() / 3600
            return math.exp(-0.1 * hours / 24)
        except (ValueError, TypeError):
            return 0.5

    def size(self) -> int:
        return self._collection.count()

    def clear(self):
        self._chroma.delete_collection(self._collection.name)
        self._collection = self._chroma.get_or_create_collection(
            name=self._collection.name,
            metadata={"hnsw:space": "cosine"},
        )
        conn = sqlite3.connect(self.db_path)
        conn.execute("DELETE FROM episodes")
        conn.commit()
        conn.close()


# ============================================================
# 4. 语义记忆：实体/关系 + 向量检索
# ============================================================

class SemanticMemory:
    """语义记忆 — 抽象知识和概念

    特点：
    - SQLite 存储实体和关系（轻量知识图谱）
    - ChromaDB 存储向量用于语义检索
    - 评分公式：(向量相似度 × 0.7 + 实体匹配 × 0.3) × 重要性权重
    """

    def __init__(self, db_path: str = "./memory_db/semantic.db",
                 collection_name: str = "semantic_memory"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_db()
        self._chroma = chromadb.Client()
        self._collection = self._chroma.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self._embedder = SentenceTransformer("all-MiniLM-L6-v2")

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS entities (
                entity_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                entity_type TEXT DEFAULT 'concept',
                memory_id TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS relations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                subject TEXT NOT NULL,
                predicate TEXT NOT NULL,
                object TEXT NOT NULL,
                memory_id TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS knowledge (
                memory_id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                importance REAL DEFAULT 0.5,
                timestamp TEXT NOT NULL,
                access_count INTEGER DEFAULT 0
            )
        """)
        conn.commit()
        conn.close()

    def add(self, content: str, importance: float = 0.5,
            entities: list[dict] | None = None,
            relations: list[dict] | None = None, **metadata) -> str:
        """添加语义记忆，可选传入预抽取的实体和关系"""
        item = MemoryItem(
            content=content, memory_type="semantic",
            importance=importance, metadata=metadata,
        )
        # 如果没有传入实体，用简单规则抽取
        if entities is None:
            entities = self._simple_entity_extract(content)
        if relations is None:
            relations = []

        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "INSERT OR REPLACE INTO knowledge VALUES (?, ?, ?, ?, ?)",
            (item.memory_id, content, importance,
             item.timestamp.isoformat(), 0),
        )
        for ent in entities:
            eid = uuid.uuid4().hex[:8]
            conn.execute(
                "INSERT OR REPLACE INTO entities VALUES (?, ?, ?, ?)",
                (eid, ent.get("name", ""), ent.get("type", "concept"),
                 item.memory_id),
            )
        for rel in relations:
            conn.execute(
                "INSERT INTO relations (subject, predicate, object, memory_id) "
                "VALUES (?, ?, ?, ?)",
                (rel["subject"], rel["predicate"], rel["object"],
                 item.memory_id),
            )
        conn.commit()
        conn.close()

        embedding = self._embedder.encode(content).tolist()
        self._collection.add(
            ids=[item.memory_id],
            embeddings=[embedding],
            documents=[content],
            metadatas=[{"importance": importance,
                        "timestamp": item.timestamp.isoformat()}],
        )
        return item.memory_id

    def retrieve(self, query: str, top_k: int = 5) -> list[tuple[MemoryItem, float]]:
        if self._collection.count() == 0:
            return []
        q_emb = self._embedder.encode(query).tolist()
        results = self._collection.query(
            query_embeddings=[q_emb],
            n_results=min(top_k * 3, self._collection.count()),
        )
        entity_boost = self._entity_match_scores(query)

        scored = []
        for i, mid in enumerate(results["ids"][0]):
            dist = results["distances"][0][i] if results["distances"] else 0
            vec_sim = max(0, 1 - dist)
            ent_score = entity_boost.get(mid, 0.0)
            meta = results["metadatas"][0][i]
            importance = meta.get("importance", 0.5)
            base = vec_sim * 0.7 + ent_score * 0.3
            imp_weight = 0.8 + importance * 0.4
            final = base * imp_weight
            item = MemoryItem(
                content=results["documents"][0][i],
                memory_type="semantic",
                importance=importance,
                memory_id=mid,
            )
            scored.append((item, final))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def get_entity_relations(self, entity_name: str) -> list[dict]:
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute(
            "SELECT subject, predicate, object FROM relations "
            "WHERE subject LIKE ? OR object LIKE ?",
            (f"%{entity_name}%", f"%{entity_name}%"),
        ).fetchall()
        conn.close()
        return [{"subject": r[0], "predicate": r[1], "object": r[2]}
                for r in rows]

    def _entity_match_scores(self, query: str) -> dict[str, float]:
        """查询中的词与记忆关联实体的匹配度"""
        q_lower = query.lower()
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute(
            "SELECT name, memory_id FROM entities"
        ).fetchall()
        conn.close()
        scores: dict[str, float] = {}
        for name, mid in rows:
            if name.lower() in q_lower:
                scores[mid] = scores.get(mid, 0) + 1.0
        if scores:
            max_s = max(scores.values())
            scores = {k: v / max_s for k, v in scores.items()}
        return scores

    @staticmethod
    def _simple_entity_extract(text: str) -> list[dict]:
        """简单的实体抽取（基于规则，不依赖 LLM）"""
        entities = []
        for token in text.replace("，", " ").replace("。", " ").split():
            token = token.strip()
            if len(token) >= 2 and not token.isdigit():
                entities.append({"name": token, "type": "concept"})
        return entities[:10]

    def size(self) -> int:
        return self._collection.count()

    def clear(self):
        self._chroma.delete_collection(self._collection.name)
        self._collection = self._chroma.get_or_create_collection(
            name=self._collection.name,
            metadata={"hnsw:space": "cosine"},
        )
        conn = sqlite3.connect(self.db_path)
        conn.execute("DELETE FROM entities")
        conn.execute("DELETE FROM relations")
        conn.execute("DELETE FROM knowledge")
        conn.commit()
        conn.close()


# ============================================================
# 5. Demo：三种记忆类型的对比演示
# ============================================================

def demo_working_memory():
    console.print(Panel("[bold]Demo 1: 工作记忆 — 短期高速缓存[/bold]"))

    wm = WorkingMemory(capacity=5, ttl_minutes=60)
    wm.add("用户正在学习 Python 的装饰器", importance=0.7)
    wm.add("用户问了关于闭包的问题", importance=0.5)
    wm.add("用户提到自己是后端开发者", importance=0.8)
    wm.add("讨论了 Python 的 GIL 问题", importance=0.6)
    wm.add("用户对异步编程感兴趣", importance=0.4)

    console.print(f"  记忆数量: {wm.size()}")

    results = wm.retrieve("Python 装饰器怎么用", top_k=3)
    table = Table(title="工作记忆检索结果")
    table.add_column("内容", style="cyan", max_width=40)
    table.add_column("得分", style="green", width=8)
    table.add_column("重要性", width=8)
    for mem, score in results:
        table.add_row(mem.content, f"{score:.3f}", f"{mem.importance:.1f}")
    console.print(table)

    # 测试容量淘汰
    wm.add("新的对话内容：讨论 FastAPI", importance=0.9)
    console.print(f"  添加第 6 条后记忆数量: {wm.size()} (容量限制=5)")
    wm.clear()


def demo_episodic_memory():
    console.print(Panel("[bold]Demo 2: 情景记忆 — 事件和经历[/bold]"))

    em = EpisodicMemory(
        db_path="./memory_db/demo_episodic.db",
        collection_name="demo_episodic",
    )
    em.clear()

    em.add("2024年3月，用户完成了第一个 RAG 项目", importance=0.9,
           session_id="session_001")
    em.add("用户在调试 ChromaDB 连接时遇到了超时问题", importance=0.6,
           session_id="session_001")
    em.add("用户学习了 BM25 和向量检索的混合策略", importance=0.8,
           session_id="session_002")
    em.add("讨论了 RAGAS 评估框架的使用方法", importance=0.7,
           session_id="session_002")

    results = em.retrieve("RAG 项目经验", top_k=3)
    table = Table(title="情景记忆检索结果")
    table.add_column("内容", style="cyan", max_width=40)
    table.add_column("得分", style="green", width=8)
    table.add_column("会话", width=14)
    for mem, score in results:
        table.add_row(mem.content, f"{score:.3f}", mem.session_id)
    console.print(table)

    # 会话历史
    history = em.get_session_history("session_001")
    console.print(f"  session_001 历史记录: {len(history)} 条")
    for h in history:
        console.print(f"    - {h.content[:50]}")
    em.clear()


def demo_semantic_memory():
    console.print(Panel("[bold]Demo 3: 语义记忆 — 知识和概念[/bold]"))

    sm = SemanticMemory(
        db_path="./memory_db/demo_semantic.db",
        collection_name="demo_semantic",
    )
    sm.clear()

    sm.add(
        "RAG 是检索增强生成技术，结合信息检索和文本生成",
        importance=0.9,
        entities=[
            {"name": "RAG", "type": "technology"},
            {"name": "检索增强生成", "type": "concept"},
        ],
        relations=[
            {"subject": "RAG", "predicate": "结合", "object": "信息检索"},
            {"subject": "RAG", "predicate": "结合", "object": "文本生成"},
        ],
    )
    sm.add(
        "ChromaDB 是一个开源的向量数据库，适合 RAG 应用",
        importance=0.7,
        entities=[
            {"name": "ChromaDB", "type": "tool"},
            {"name": "向量数据库", "type": "concept"},
        ],
        relations=[
            {"subject": "ChromaDB", "predicate": "是", "object": "向量数据库"},
            {"subject": "ChromaDB", "predicate": "适合", "object": "RAG"},
        ],
    )
    sm.add(
        "BM25 是经典的稀疏检索算法，基于词频统计",
        importance=0.6,
        entities=[
            {"name": "BM25", "type": "algorithm"},
        ],
    )

    results = sm.retrieve("什么是 RAG 技术", top_k=3)
    table = Table(title="语义记忆检索结果")
    table.add_column("内容", style="cyan", max_width=40)
    table.add_column("得分", style="green", width=8)
    table.add_column("重要性", width=8)
    for mem, score in results:
        table.add_row(mem.content, f"{score:.3f}", f"{mem.importance:.1f}")
    console.print(table)

    # 知识图谱查询
    rels = sm.get_entity_relations("RAG")
    console.print(f"\n  RAG 的关系网络:")
    for r in rels:
        console.print(f"    {r['subject']} --[{r['predicate']}]--> {r['object']}")
    sm.clear()


def cleanup():
    import shutil
    if os.path.exists("./memory_db"):
        shutil.rmtree("./memory_db")
        console.print("\n[dim]已清理 memory_db 目录[/dim]")


if __name__ == "__main__":
    console.print(Panel(
        "[bold]09 — 认知科学启发的四类记忆系统[/bold]\n"
        "工作记忆 | 情景记忆 | 语义记忆",
        style="blue",
    ))
    demo_working_memory()
    demo_episodic_memory()
    demo_semantic_memory()
    cleanup()

