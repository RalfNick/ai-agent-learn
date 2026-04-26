"""
10_memory_lifecycle.py — 记忆生命周期：整合、遗忘与评分

学习目标：
1. 理解记忆的完整生命周期：编码 → 存储 → 检索 → 整合 → 遗忘
2. 实现时间衰减评分（指数衰减模型，模拟艾宾浩斯遗忘曲线）
3. 实现记忆整合：工作记忆 → 情景记忆 → 语义记忆的逐级提升
4. 实现三种遗忘策略：基于重要性、基于时间、基于容量
5. 观察记忆系统在多轮交互中的动态演变

核心概念：
- 整合（Consolidation）：重要的短期记忆提升为长期记忆
- 遗忘（Forgetting）：清理低价值记忆，保持系统高效
- 评分（Scoring）：多因素综合评分决定记忆的检索优先级

记忆生命周期：
    编码 ──▶ 存储 ──▶ 检索 ──▶ 整合 ──▶ 遗忘
     │                  │        │        │
     │   importance     │  score  │ promote│ evict
     │   timestamp      │  rank   │  tier  │ clean
     ▼                  ▼        ▼        ▼
    MemoryItem      top_k结果  升级类型   释放空间

    整合路径：
    WorkingMemory ──(importance>0.7)──▶ EpisodicMemory
    EpisodicMemory ──(access>5)──▶ SemanticMemory

运行方式：
    python 10_memory_lifecycle.py
"""

import math
import os
import shutil
from datetime import datetime, timedelta

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# 从 09 导入记忆类
from importlib import import_module
_m09 = import_module("09_memory_system")
MemoryItem = _m09.MemoryItem
WorkingMemory = _m09.WorkingMemory
EpisodicMemory = _m09.EpisodicMemory
SemanticMemory = _m09.SemanticMemory

console = Console()


# ============================================================
# 1. 时间衰减与评分
# ============================================================

def time_decay(timestamp: datetime, half_life_hours: float = 24.0) -> float:
    """指数衰减模型，模拟艾宾浩斯遗忘曲线

    公式: decay = exp(-0.693 * elapsed_hours / half_life)
    - half_life=24h 时，24小时后衰减到 0.5，48小时后 0.25
    """
    elapsed = (datetime.now() - timestamp).total_seconds() / 3600
    return math.exp(-0.693 * max(0, elapsed) / half_life_hours)


class MemoryScorer:
    """多因素综合评分器

    最终得分 = α × 语义相似度 + β × 重要性 + γ × 时间近因性
    默认权重：α=0.5, β=0.3, γ=0.2
    """

    def __init__(self, alpha: float = 0.5, beta: float = 0.3,
                 gamma: float = 0.2):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def score(self, semantic_sim: float, importance: float,
              timestamp: datetime) -> float:
        recency = time_decay(timestamp)
        return (self.alpha * semantic_sim
                + self.beta * importance
                + self.gamma * recency)

    def score_item(self, item: MemoryItem, semantic_sim: float = 0.5) -> float:
        return self.score(semantic_sim, item.importance, item.timestamp)


# ============================================================
# 2. 记忆整合：逐级提升
# ============================================================

class MemoryConsolidator:
    """记忆整合器 — 将重要的短期记忆提升为长期记忆

    整合规则：
    - working → episodic: importance > importance_threshold (默认 0.7)
    - episodic → semantic: access_count > access_threshold (默认 5)
    """

    def __init__(self, importance_threshold: float = 0.7,
                 access_threshold: int = 5):
        self.importance_threshold = importance_threshold
        self.access_threshold = access_threshold

    def consolidate_working_to_episodic(
        self, working: WorkingMemory, episodic: EpisodicMemory,
        session_id: str = "default",
    ) -> int:
        """将重要的工作记忆提升为情景记忆"""
        promoted = 0
        to_remove = []
        for mem in working.memories:
            if mem.importance >= self.importance_threshold:
                episodic.add(
                    content=mem.content,
                    importance=mem.importance,
                    session_id=session_id,
                    source="consolidated_from_working",
                )
                to_remove.append(mem)
                promoted += 1
        for mem in to_remove:
            working.memories.remove(mem)
        return promoted

    def consolidate_episodic_to_semantic(
        self, episodic: EpisodicMemory, semantic: SemanticMemory,
    ) -> int:
        """将高频访问的情景记忆提升为语义记忆

        注意：简化实现，通过检索热门记忆来模拟 access_count
        """
        promoted = 0
        # 检索所有情景记忆中重要性最高的
        results = episodic.retrieve("", top_k=20)
        for mem, score in results:
            if mem.importance >= 0.8:
                semantic.add(
                    content=mem.content,
                    importance=mem.importance,
                    source="consolidated_from_episodic",
                )
                promoted += 1
                if promoted >= 5:
                    break
        return promoted


# ============================================================
# 3. 遗忘策略
# ============================================================

def forget_by_importance(memories: list[MemoryItem],
                         threshold: float = 0.2) -> tuple[list[MemoryItem], int]:
    """基于重要性遗忘：删除重要性低于阈值的记忆"""
    kept = [m for m in memories if m.importance >= threshold]
    forgotten = len(memories) - len(kept)
    return kept, forgotten


def forget_by_time(memories: list[MemoryItem],
                   max_age_days: int = 30) -> tuple[list[MemoryItem], int]:
    """基于时间遗忘：删除超过指定天数的记忆"""
    cutoff = datetime.now() - timedelta(days=max_age_days)
    kept = [m for m in memories if m.timestamp > cutoff]
    forgotten = len(memories) - len(kept)
    return kept, forgotten


def forget_by_capacity(memories: list[MemoryItem],
                       max_items: int = 100) -> tuple[list[MemoryItem], int]:
    """基于容量遗忘：超出容量时删除综合得分最低的记忆"""
    if len(memories) <= max_items:
        return memories, 0
    scorer = MemoryScorer()
    scored = [(m, scorer.score_item(m)) for m in memories]
    scored.sort(key=lambda x: x[1], reverse=True)
    kept = [m for m, _ in scored[:max_items]]
    return kept, len(memories) - max_items


# ============================================================
# 4. 生命周期管理器
# ============================================================

class MemoryLifecycleManager:
    """编排完整的记忆生命周期"""

    def __init__(self, db_dir: str = "./memory_db"):
        self.db_dir = db_dir
        self.working = WorkingMemory(capacity=20, ttl_minutes=60)
        self.episodic = EpisodicMemory(
            db_path=f"{db_dir}/lifecycle_episodic.db",
            collection_name="lifecycle_episodic",
        )
        self.semantic = SemanticMemory(
            db_path=f"{db_dir}/lifecycle_semantic.db",
            collection_name="lifecycle_semantic",
        )
        self.consolidator = MemoryConsolidator()
        self.scorer = MemoryScorer()

    def add_interaction(self, content: str, importance: float = 0.5,
                        session_id: str = "default"):
        """添加一次交互到工作记忆"""
        self.working.add(content, importance=importance,
                         session_id=session_id)

    def run_consolidation(self, session_id: str = "default") -> dict:
        """执行一轮整合"""
        w2e = self.consolidator.consolidate_working_to_episodic(
            self.working, self.episodic, session_id)
        e2s = self.consolidator.consolidate_episodic_to_semantic(
            self.episodic, self.semantic)
        return {"working_to_episodic": w2e, "episodic_to_semantic": e2s}

    def run_forgetting(self) -> dict:
        """执行一轮遗忘"""
        before = self.working.size()
        self.working.memories, _ = forget_by_importance(
            self.working.memories, threshold=0.2)
        self.working.memories, _ = forget_by_capacity(
            self.working.memories, max_items=20)
        after = self.working.size()
        return {"working_forgotten": before - after}

    def get_stats(self) -> dict:
        return {
            "working": self.working.size(),
            "episodic": self.episodic.size(),
            "semantic": self.semantic.size(),
        }

    def clear(self):
        self.working.clear()
        self.episodic.clear()
        self.semantic.clear()


# ============================================================
# 5. Demo：生命周期模拟
# ============================================================

def demo_scoring():
    console.print(Panel("[bold]Demo 1: 时间衰减与多因素评分[/bold]"))

    table = Table(title="时间衰减效果 (half_life=24h)")
    table.add_column("经过时间", width=12)
    table.add_column("衰减值", width=10)
    for hours in [0, 1, 6, 12, 24, 48, 72, 168]:
        ts = datetime.now() - timedelta(hours=hours)
        decay = time_decay(ts, half_life_hours=24)
        label = f"{hours}h" if hours < 24 else f"{hours//24}d"
        table.add_row(label, f"{decay:.4f}")
    console.print(table)

    scorer = MemoryScorer(alpha=0.5, beta=0.3, gamma=0.2)
    table2 = Table(title="多因素评分示例")
    table2.add_column("场景", width=20)
    table2.add_column("语义相似度", width=10)
    table2.add_column("重要性", width=8)
    table2.add_column("时间", width=8)
    table2.add_column("最终得分", width=10)
    cases = [
        ("高相关+高重要+新", 0.9, 0.9, 0),
        ("高相关+低重要+新", 0.9, 0.2, 0),
        ("低相关+高重要+新", 0.2, 0.9, 0),
        ("高相关+高重要+旧", 0.9, 0.9, 72),
        ("中等全面", 0.5, 0.5, 12),
    ]
    for name, sim, imp, hours in cases:
        ts = datetime.now() - timedelta(hours=hours)
        s = scorer.score(sim, imp, ts)
        table2.add_row(name, f"{sim:.1f}", f"{imp:.1f}",
                        f"{hours}h", f"{s:.3f}")
    console.print(table2)


def demo_consolidation():
    console.print(Panel("[bold]Demo 2: 记忆整合 — 短期 → 长期[/bold]"))

    mgr = MemoryLifecycleManager(db_dir="./memory_db")
    mgr.clear()

    interactions = [
        ("用户问了 Python 装饰器的用法", 0.5),
        ("用户是高级后端开发者", 0.9),
        ("讨论了 FastAPI 的依赖注入", 0.6),
        ("用户完成了 RAG 项目的部署", 0.85),
        ("闲聊了天气", 0.1),
        ("用户对 LangGraph 很感兴趣", 0.75),
    ]

    console.print("  [bold]添加 6 条工作记忆...[/bold]")
    for content, imp in interactions:
        mgr.add_interaction(content, importance=imp, session_id="s1")

    stats = mgr.get_stats()
    console.print(f"  整合前: working={stats['working']}, "
                  f"episodic={stats['episodic']}, semantic={stats['semantic']}")

    result = mgr.run_consolidation(session_id="s1")
    console.print(f"  整合结果: {result}")

    stats = mgr.get_stats()
    console.print(f"  整合后: working={stats['working']}, "
                  f"episodic={stats['episodic']}, semantic={stats['semantic']}")
    mgr.clear()


def demo_forgetting():
    console.print(Panel("[bold]Demo 3: 三种遗忘策略[/bold]"))

    memories = [
        MemoryItem("重要知识A", importance=0.9,
                   timestamp=datetime.now()),
        MemoryItem("重要知识B", importance=0.8,
                   timestamp=datetime.now() - timedelta(days=10)),
        MemoryItem("一般信息C", importance=0.4,
                   timestamp=datetime.now() - timedelta(days=5)),
        MemoryItem("低价值D", importance=0.1,
                   timestamp=datetime.now() - timedelta(days=40)),
        MemoryItem("过期信息E", importance=0.3,
                   timestamp=datetime.now() - timedelta(days=60)),
    ]

    table = Table(title="遗忘策略对比")
    table.add_column("策略", width=18)
    table.add_column("参数", width=16)
    table.add_column("遗忘数", width=8)
    table.add_column("保留", width=30)

    kept, n = forget_by_importance(memories, threshold=0.3)
    table.add_row("重要性过滤", "threshold=0.3", str(n),
                  ", ".join(m.content for m in kept))

    kept, n = forget_by_time(memories, max_age_days=30)
    table.add_row("时间过滤", "max_age=30d", str(n),
                  ", ".join(m.content for m in kept))

    kept, n = forget_by_capacity(memories, max_items=3)
    table.add_row("容量过滤", "max=3", str(n),
                  ", ".join(m.content for m in kept))

    console.print(table)


def demo_lifecycle_simulation():
    console.print(Panel("[bold]Demo 4: 完整生命周期模拟（5 轮会话）[/bold]"))

    mgr = MemoryLifecycleManager(db_dir="./memory_db")
    mgr.clear()

    sessions = {
        "s1": [("学习了 Python 基础语法", 0.5),
               ("用户是数据科学家", 0.9),
               ("讨论了 pandas 数据处理", 0.6)],
        "s2": [("学习了 RAG 的基本原理", 0.8),
               ("实践了 ChromaDB 向量存储", 0.7),
               ("闲聊", 0.1)],
        "s3": [("深入学习了混合检索策略", 0.85),
               ("完成了 RAGAS 评估实验", 0.75),
               ("讨论了 HyDE 查询改写", 0.7)],
        "s4": [("学习了记忆系统的设计", 0.9),
               ("实现了工作记忆模块", 0.8),
               ("临时笔记", 0.15)],
        "s5": [("整合了 Memory + RAG 系统", 0.95),
               ("部署了完整的问答助手", 0.85),
               ("测试了多轮对话效果", 0.7)],
    }

    table = Table(title="生命周期演变")
    table.add_column("会话", width=6)
    table.add_column("操作", width=12)
    table.add_column("Working", width=8)
    table.add_column("Episodic", width=8)
    table.add_column("Semantic", width=8)

    for sid, interactions in sessions.items():
        for content, imp in interactions:
            mgr.add_interaction(content, importance=imp, session_id=sid)
        stats = mgr.get_stats()
        table.add_row(sid, "添加后", str(stats["working"]),
                      str(stats["episodic"]), str(stats["semantic"]))

        mgr.run_consolidation(session_id=sid)
        mgr.run_forgetting()
        stats = mgr.get_stats()
        table.add_row("", "整合+遗忘", str(stats["working"]),
                      str(stats["episodic"]), str(stats["semantic"]))

    console.print(table)
    mgr.clear()


def cleanup():
    if os.path.exists("./memory_db"):
        shutil.rmtree("./memory_db")
        console.print("\n[dim]已清理 memory_db 目录[/dim]")


if __name__ == "__main__":
    console.print(Panel(
        "[bold]10 — 记忆生命周期：整合、遗忘与评分[/bold]\n"
        "时间衰减 | 多因素评分 | 整合提升 | 遗忘策略",
        style="blue",
    ))
    demo_scoring()
    demo_consolidation()
    demo_forgetting()
    demo_lifecycle_simulation()
    cleanup()

