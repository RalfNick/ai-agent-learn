"""
Phase 2 RAG benchmark dataset.

The corpus points at real learning notes and runnable scripts in this repo.
The labels are source-level relevance judgments: a retrieved chunk is relevant
when its source_id appears in the question's relevant_source_ids.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SourceSpec:
    source_id: str
    path: str
    title: str
    source_type: str


@dataclass(frozen=True)
class EvalQuestion:
    question: str
    relevant_source_ids: tuple[str, ...]
    ground_truth: str
    evidence: str


SOURCE_SPECS: tuple[SourceSpec, ...] = (
    SourceSpec("p1_react_article", "docs/phase-1/01-agent-react-fundamentals.md", "ReAct Agent fundamentals article", "markdown"),
    SourceSpec("p1_arch_article", "docs/phase-1/02-agent-architecture-deep-dive.md", "Agent architecture deep dive article", "markdown"),
    SourceSpec("p2_rag_overview", "docs/phase-2/01-rag-landscape-overview.md", "RAG landscape overview article", "markdown"),
    SourceSpec("p2_rag_pipeline", "docs/phase-2/02-rag-pipeline-deep-dive.md", "RAG pipeline deep dive article", "markdown"),
    SourceSpec("p2_hybrid_article", "docs/phase-2/03-hybrid-search-rerank-deep-dive.md", "Hybrid search and rerank article", "markdown"),
    SourceSpec("p3_langgraph_article", "docs/phase-3/legacy/01-langgraph-workflow-deep-dive.md", "LangGraph workflow article", "markdown"),
    SourceSpec("p3_multi_agent_article", "docs/phase-3/legacy/02-multi-agent-paradigms.md", "Multi-agent paradigms article", "markdown"),
    SourceSpec("p3_framework_article", "docs/phase-3/legacy/03-framework-comparison-insights.md", "Framework comparison article", "markdown"),
    SourceSpec("minimal_hello_agent", "phase-1-fundamentals/01-minimal-agent/01_hello_agent.py", "Minimal hello agent script", "python"),
    SourceSpec("minimal_tools", "phase-1-fundamentals/01-minimal-agent/02_custom_tools.py", "Custom tools script", "python"),
    SourceSpec("code_vs_toolcalling", "phase-1-fundamentals/01-minimal-agent/03_code_vs_toolcalling.py", "Code agent vs tool calling script", "python"),
    SourceSpec("minimal_multi_agent", "phase-1-fundamentals/01-minimal-agent/04_multi_agent.py", "Minimal multi-agent script", "python"),
    SourceSpec("smol_internals", "phase-1-fundamentals/02-smolagents-deep-dive/05_agent_internals.py", "smolagents internals script", "python"),
    SourceSpec("smol_tools", "phase-1-fundamentals/02-smolagents-deep-dive/06_tool_ecosystem.py", "smolagents tool ecosystem script", "python"),
    SourceSpec("smol_planning", "phase-1-fundamentals/02-smolagents-deep-dive/07_planning_reflection.py", "Planning and reflection script", "python"),
    SourceSpec("smol_gradio", "phase-1-fundamentals/02-smolagents-deep-dive/08_gradio_interactive.py", "Gradio interactive agent script", "python"),
    SourceSpec("smol_sandbox", "phase-1-fundamentals/02-smolagents-deep-dive/09_sandboxed_execution.py", "Sandboxed execution script", "python"),
    SourceSpec("arch_engine", "phase-1-fundamentals/03-agent-architecture/10_execution_engine.py", "Execution engine script", "python"),
    SourceSpec("arch_tool_system", "phase-1-fundamentals/03-agent-architecture/11_tool_system.py", "Tool system architecture script", "python"),
    SourceSpec("arch_prompt_engine", "phase-1-fundamentals/03-agent-architecture/12_prompt_engine.py", "Prompt engine script", "python"),
    SourceSpec("arch_supervisor", "phase-1-fundamentals/03-agent-architecture/13_supervisor_orchestrator.py", "Supervisor orchestrator script", "python"),
    SourceSpec("arch_memory", "phase-1-fundamentals/03-agent-architecture/14_memory_context.py", "Memory and context script", "python"),
    SourceSpec("arch_mini_agent", "phase-1-fundamentals/03-agent-architecture/15_mini_agent.py", "Integrated mini agent script", "python"),
    SourceSpec("rag_loading", "phase-2-rag/01-basic-rag/01_document_loading.py", "Document loading script", "python"),
    SourceSpec("rag_chunking", "phase-2-rag/01-basic-rag/02_text_chunking.py", "Text chunking script", "python"),
    SourceSpec("rag_embedding_store", "phase-2-rag/01-basic-rag/03_embedding_vectorstore.py", "Embedding and vector store script", "python"),
    SourceSpec("rag_naive", "phase-2-rag/01-basic-rag/04_naive_rag.py", "Naive RAG script", "python"),
    SourceSpec("rag_hybrid_basic", "phase-2-rag/02-advanced-rag/05_hybrid_search.py", "Hybrid search basics script", "python"),
    SourceSpec("rag_reranking", "phase-2-rag/02-advanced-rag/06_reranking.py", "Reranking script", "python"),
    SourceSpec("rag_query_transform", "phase-2-rag/02-advanced-rag/07_query_transformation.py", "Query transformation script", "python"),
    SourceSpec("rag_ragas_intro", "phase-2-rag/02-advanced-rag/08_ragas_evaluation.py", "RAGAS introduction script", "python"),
    SourceSpec("rag_hybrid_pipeline", "phase-2-rag/03-hybrid-search/01_hybrid_retrieval_pipeline.py", "Hybrid retrieval pipeline script", "python"),
    SourceSpec("rag_rerank_pipeline", "phase-2-rag/03-hybrid-search/02_rerank_pipeline.py", "Two-stage rerank pipeline script", "python"),
    SourceSpec("rag_full_pipeline", "phase-2-rag/03-hybrid-search/03_full_rag_pipeline.py", "Full RAG pipeline script", "python"),
    SourceSpec("memory_system", "phase-2-rag/03-memory-rag/09_memory_system.py", "Memory system script", "python"),
    SourceSpec("memory_lifecycle", "phase-2-rag/03-memory-rag/10_memory_lifecycle.py", "Memory lifecycle script", "python"),
    SourceSpec("unified_retrieval", "phase-2-rag/03-memory-rag/11_unified_retrieval.py", "Unified retrieval script", "python"),
    SourceSpec("memory_enhanced_rag", "phase-2-rag/03-memory-rag/12_memory_enhanced_rag.py", "Memory enhanced RAG script", "python"),
    SourceSpec("pdf_learning_assistant", "phase-2-rag/03-memory-rag/13_pdf_learning_assistant.py", "PDF learning assistant script", "python"),
    SourceSpec("ragas_from_scratch", "phase-2-rag/04-rag-evaluation/01_ragas_metrics_from_scratch.py", "RAGAS metrics from scratch script", "python"),
    SourceSpec("rag_eval_pipeline", "phase-2-rag/04-rag-evaluation/02_evaluation_pipeline.py", "Automated evaluation pipeline script", "python"),
    SourceSpec("rag_optimization_lab", "phase-2-rag/04-rag-evaluation/03_rag_optimization_lab.py", "RAG optimization lab script", "python"),
)


EVAL_QUESTIONS: tuple[EvalQuestion, ...] = (
    EvalQuestion(
        "ReAct Agent 和普通 LLM 对话的核心区别是什么？",
        ("p1_react_article", "minimal_hello_agent"),
        "Agent 不只是生成文本，还能围绕任务进行多步推理、调用工具、观察结果并继续行动；普通 LLM 对话通常只完成一次文本生成。",
        "phase1 ReAct 文章解释了 LLM 与 Agent 的能力、状态、决策和边界差异；minimal agent 脚本展示了可运行 Agent。",
    ),
    EvalQuestion(
        "一个 Agent 框架通常可以拆成哪几层架构？",
        ("p1_arch_article", "arch_mini_agent", "arch_engine"),
        "通用 Agent 框架可拆为 LLM 适配、状态与记忆、工具系统、执行引擎、Prompt 引擎和多 Agent 编排等层。",
        "架构文章给出六层抽象；mini agent 和 execution engine 脚本实现了核心层的组合。",
    ),
    EvalQuestion(
        "工具系统为什么要使用 schema 驱动注册与调用？",
        ("p1_arch_article", "arch_tool_system", "minimal_tools"),
        "Schema 驱动让工具的名称、描述、参数和返回类型可被统一注册、注入 prompt、校验参数并执行调用，降低工具扩展成本。",
        "tool system 脚本展示 Tool/ToolRegistry；架构文章解释工具层职责。",
    ),
    EvalQuestion(
        "Prompt 引擎在 Agent 中主要解决什么问题？",
        ("p1_arch_article", "arch_prompt_engine"),
        "Prompt 引擎负责模板管理、变量注入、消息历史组织和工具描述拼接，使同一 Agent 能适配不同任务和工具集合。",
        "架构文章和 prompt engine 脚本都围绕模板系统、动态注入和消息管理展开。",
    ),
    EvalQuestion(
        "Supervisor/Worker 多 Agent 模式适合解决什么问题？",
        ("p1_arch_article", "arch_supervisor", "minimal_multi_agent", "p3_multi_agent_article"),
        "Supervisor/Worker 适合把复杂任务拆给多个专长 Agent，由监督者分派、汇总和控制流程。",
        "supervisor_orchestrator 和多 Agent 文章描述中心化调度、子任务分发和结果汇总。",
    ),
    EvalQuestion(
        "RAG 主要解决 LLM 的哪几个根本限制？",
        ("p2_rag_overview", "p2_rag_pipeline", "rag_naive"),
        "RAG 通过外部检索缓解知识过期、缺少私有数据和幻觉问题，让模型基于检索到的资料回答。",
        "RAG 全景和 pipeline 文章都以知识截止、私有数据、幻觉为动机；naive RAG 脚本展示检索加生成。",
    ),
    EvalQuestion(
        "RAG 的离线索引阶段包含哪些步骤？",
        ("p2_rag_overview", "p2_rag_pipeline", "rag_loading", "rag_chunking", "rag_embedding_store"),
        "离线阶段通常包括文档加载与清洗、文本分块、Embedding 向量化、写入向量数据库并建立索引。",
        "文档加载、分块、embedding/vectorstore 脚本分别实现索引链路的关键步骤。",
    ),
    EvalQuestion(
        "RAG 的在线查询阶段包含哪些步骤？",
        ("p2_rag_overview", "p2_rag_pipeline", "rag_naive", "rag_full_pipeline"),
        "在线阶段包括问题向量化、检索相关文档、构造带上下文的 prompt、调用 LLM 生成带依据的回答。",
        "overview/pipeline 文章和 naive/full RAG 脚本展示 query -> retrieval -> prompt -> generation。",
    ),
    EvalQuestion(
        "为什么 chunk_size 和 chunk_overlap 会影响 RAG 效果？",
        ("p2_rag_pipeline", "rag_chunking", "p2_hybrid_article"),
        "块太大会降低检索精度，块太小会丢上下文；overlap 可以减少边界截断，但过多会增加冗余和成本。",
        "pipeline 文章和 chunking 脚本讨论分块质量；hybrid 文章也提到分块对 precision/recall 的影响。",
    ),
    EvalQuestion(
        "文档加载和清洗为什么是 RAG 的上限因素？",
        ("p2_rag_pipeline", "rag_loading"),
        "加载和清洗决定进入索引的数据质量；PDF 乱序、网页噪声、编码和控制字符等问题会直接污染检索与生成。",
        "pipeline 文章强调垃圾进垃圾出；document_loading 脚本展示多格式加载和清洗。",
    ),
    EvalQuestion(
        "Chroma 向量库在基础 RAG 中承担什么职责？",
        ("rag_embedding_store", "rag_naive", "p2_rag_pipeline"),
        "Chroma 存储文本块向量和元数据，支持按查询向量进行相似度检索，返回相关文档块给生成阶段。",
        "embedding_vectorstore 和 naive_rag 脚本使用 Chroma 构建 collection、add documents、query。",
    ),
    EvalQuestion(
        "Naive RAG 的主要局限性有哪些？",
        ("rag_naive", "p2_rag_pipeline", "p2_hybrid_article"),
        "Naive RAG 可能检索遗漏、排序不准、上下文拼接粗糙、查询理解弱，难以处理关键词和语义混合需求。",
        "naive_rag 脚本列出局限；hybrid/rerank 文章解释为什么需要高级检索。",
    ),
    EvalQuestion(
        "BM25 和向量检索分别擅长什么？",
        ("p2_hybrid_article", "rag_hybrid_basic", "rag_hybrid_pipeline"),
        "BM25 擅长关键词、专有名词和精确匹配；向量检索擅长语义相似和同义表达，两者互补。",
        "hybrid search 文章和脚本对 BM25 sparse retrieval 与 dense retrieval 做了对比。",
    ),
    EvalQuestion(
        "混合检索为什么通常比单一检索更稳？",
        ("p2_hybrid_article", "rag_hybrid_basic", "rag_hybrid_pipeline"),
        "混合检索结合 BM25 与 dense retrieval，可同时捕获精确关键词和语义相关内容，降低单一路径失误风险。",
        "hybrid article 和 pipeline 使用 BM25 + Dense + RRF 融合。",
    ),
    EvalQuestion(
        "RRF 排序融合的作用是什么？",
        ("p2_hybrid_article", "rag_hybrid_basic", "rag_hybrid_pipeline", "rag_full_pipeline"),
        "RRF 通过倒数排名加权融合多个检索器的结果，不依赖原始分数归一化，常用于混合检索结果合并。",
        "hybrid 文章和多个 pipeline 脚本都实现 reciprocal_rank_fusion。",
    ),
    EvalQuestion(
        "Cross-Encoder 重排序为什么更准但更慢？",
        ("p2_hybrid_article", "rag_reranking", "rag_rerank_pipeline", "rag_full_pipeline"),
        "Cross-Encoder 同时输入 query 和 document pair 计算相关性，交互建模更充分，因此精度高；但每个候选都要单独打分，延迟更高。",
        "reranking 脚本和文章解释 Bi-Encoder 与 Cross-Encoder 的差异。",
    ),
    EvalQuestion(
        "两阶段检索管道通常如何组织？",
        ("p2_hybrid_article", "rag_rerank_pipeline", "rag_full_pipeline"),
        "第一阶段用 BM25/Dense/Hybrid 粗召回较多候选，第二阶段用 Cross-Encoder rerank 精排后保留 Top-K。",
        "two-stage rerank pipeline 和 full RAG pipeline 展示 first_stage_k -> rerank -> final_k。",
    ),
    EvalQuestion(
        "HyDE 查询改写的基本思想是什么？",
        ("p2_hybrid_article", "rag_query_transform", "rag_full_pipeline"),
        "HyDE 先让 LLM 为问题生成假设性答案文档，再用这个假设文档做向量检索，以弥合短查询和文档表达之间的语义差距。",
        "query_transformation 和 full pipeline 脚本实现 hyde_transform。",
    ),
    EvalQuestion(
        "Multi-Query 扩展如何提升召回率？",
        ("p2_hybrid_article", "rag_query_transform", "rag_full_pipeline"),
        "Multi-Query 把一个问题改写成多个不同角度的查询，并行检索后合并结果，从而覆盖更多相关文档。",
        "query_transformation 和 hybrid article 都描述 multi-query expansion。",
    ),
    EvalQuestion(
        "RAGAS 的四个核心指标是什么？",
        ("p2_hybrid_article", "rag_ragas_intro", "ragas_from_scratch", "rag_optimization_lab"),
        "RAGAS 常用 Faithfulness、Answer Relevancy、Context Precision 和 Context Recall 评估生成忠实度、答案相关性与检索上下文质量。",
        "ragas evaluation 脚本和文章系统解释四个指标。",
    ),
    EvalQuestion(
        "Faithfulness 低通常意味着什么，应该怎么优化？",
        ("p2_hybrid_article", "ragas_from_scratch", "rag_optimization_lab"),
        "Faithfulness 低意味着回答包含上下文无法支撑的声明，可通过更严格 prompt、降低 temperature、提升检索质量和减少无关上下文优化。",
        "评估文章和优化实验室把 Faithfulness 与幻觉、prompt/检索优化关联起来。",
    ),
    EvalQuestion(
        "Context Precision 和 Context Recall 分别衡量什么？",
        ("p2_hybrid_article", "rag_ragas_intro", "ragas_from_scratch", "rag_eval_pipeline"),
        "Context Precision 衡量检索结果中有多少真正相关；Context Recall 衡量回答所需信息是否被检索覆盖。",
        "RAGAS 脚本和评估 pipeline 都包含 precision/recall 相关解释。",
    ),
    EvalQuestion(
        "为什么要用 Precision@K、Recall@K、MRR 和 NDCG 评估检索？",
        ("rag_eval_pipeline", "rag_optimization_lab", "p2_hybrid_article"),
        "这些 IR 指标可量化前 K 个结果的准确性、覆盖率、首个相关文档排名和整体排序质量，避免凭感觉调参。",
        "evaluation_pipeline 和 optimization_lab 实现并解释这些指标。",
    ),
    EvalQuestion(
        "参数扫描在 RAG 优化中有什么价值？",
        ("rag_optimization_lab", "p2_hybrid_article"),
        "参数扫描可以系统比较 retriever_type、first_stage_k、final_k、rerank 等配置对质量、延迟和成本的影响，找到证据支持的配置。",
        "optimization_lab 脚本是参数扫描和指标驱动调优实验室。",
    ),
    EvalQuestion(
        "Agent 记忆系统可以分成哪些类型？",
        ("memory_system", "memory_lifecycle", "memory_enhanced_rag"),
        "记忆系统可包含工作记忆、情节记忆、语义记忆，以及带生命周期管理的长期记忆。",
        "memory_system 脚本实现 working/episodic/semantic memory；lifecycle 脚本处理衰减、巩固和遗忘。",
    ),
    EvalQuestion(
        "记忆生命周期管理通常包含哪些机制？",
        ("memory_lifecycle", "memory_system"),
        "记忆生命周期管理包括重要性评分、时间衰减、访问频率、记忆巩固、遗忘和清理。",
        "memory_lifecycle 脚本实现 scoring、consolidation、forgetting 和 simulation。",
    ),
    EvalQuestion(
        "统一检索如何把知识库和记忆结合起来？",
        ("unified_retrieval", "memory_enhanced_rag", "pdf_learning_assistant"),
        "统一检索把文档知识、会话记忆和长期记忆放入统一候选集合，根据查询选择相关上下文供 RAG 生成使用。",
        "unified_retrieval 和 memory_enhanced_rag 脚本展示知识与记忆的联合检索。",
    ),
    EvalQuestion(
        "PDF 学习助手如何结合 RAG 和记忆？",
        ("pdf_learning_assistant", "memory_enhanced_rag", "rag_loading", "rag_chunking"),
        "PDF 学习助手先解析和分块 PDF，再基于检索回答问题，同时记录学习进度、用户偏好和历史上下文。",
        "pdf_learning_assistant 脚本包含 PDF chunking、学习会话和 memory 对比。",
    ),
    EvalQuestion(
        "LangGraph 为什么适合复杂 Agent 工作流？",
        ("p3_langgraph_article", "p3_framework_article"),
        "LangGraph 使用状态图、节点、边和条件路由表达可循环、可分支、可持久化的复杂 Agent 工作流。",
        "phase3 LangGraph 文章介绍 StateGraph、条件路由、人机交互和持久化。",
    ),
    EvalQuestion(
        "不同 Agent 框架应该从哪些维度对比？",
        ("p3_framework_article", "p3_langgraph_article", "p3_multi_agent_article"),
        "框架对比可从状态管理、工作流可控性、多 Agent 表达、调试可观测性、开发复杂度和生产适配性等维度展开。",
        "framework comparison 和 phase3 文章围绕 LangGraph、CrewAI、Claude SDK 的取舍展开。",
    ),
)


def validate_dataset() -> None:
    source_ids = {source.source_id for source in SOURCE_SPECS}
    if len(SOURCE_SPECS) < 20:
        raise ValueError("Benchmark corpus must include at least 20 sources.")
    if not 30 <= len(EVAL_QUESTIONS) <= 50:
        raise ValueError("Benchmark must include 30-50 evaluation questions.")
    for item in EVAL_QUESTIONS:
        missing = set(item.relevant_source_ids) - source_ids
        if missing:
            raise ValueError(f"Question has unknown relevant sources: {item.question} -> {missing}")
