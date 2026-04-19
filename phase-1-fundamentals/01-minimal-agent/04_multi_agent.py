"""
04_multi_agent.py — 多 Agent 协作系统

学习目标：
1. 理解多 Agent 架构：Manager（管理者）+ Worker（执行者）
2. 理解 Agent 之间如何通信和协作
3. 观察 Manager 如何将任务分发给合适的 Worker

架构图：
    ┌──────────────┐
    │  Manager     │  ← 接收用户任务，决定分发给谁
    │  Agent       │
    └──────┬───────┘
           │ 分发任务
    ┌──────┴───────┐
    │              │
    ▼              ▼
┌────────┐  ┌────────────┐
│ Search │  │ Analyst    │
│ Agent  │  │ Agent      │
│ 搜索信息│  │ 分析推理    │
└────────┘  └────────────┘

运行方式：
    python 04_multi_agent.py
"""

import os
from dotenv import load_dotenv
from smolagents import CodeAgent, LiteLLMModel, tool

load_dotenv()

model = LiteLLMModel(model_id="deepseek/deepseek-chat", temperature=0.3)


# ============================================================
# 定义 Worker Agent 的专属工具
# ============================================================

@tool
def search_tech_news(query: str) -> str:
    """
    搜索科技新闻（模拟数据）。

    Args:
        query: 搜索关键词
    """
    mock_news = {
        "AI": [
            "OpenAI 发布 GPT-5，推理能力大幅提升",
            "Google DeepMind 发布 Gemini 2.5 Pro",
            "Anthropic Claude 4 系列模型发布，Agent 能力增强",
        ],
        "芯片": [
            "NVIDIA H200 GPU 开始量产",
            "AMD MI350 AI 加速器发布",
            "台积电 2nm 工艺进入量产阶段",
        ],
        "机器人": [
            "Figure 02 人形机器人进入工厂测试",
            "波士顿动力 Atlas 电动版商用化",
            "特斯拉 Optimus 第二代原型机亮相",
        ],
    }
    for key, news in mock_news.items():
        if key.lower() in query.lower():
            return "\n".join(f"- {n}" for n in news)
    return f"未找到与 '{query}' 相关的新闻"


@tool
def search_company_info(company: str) -> str:
    """
    搜索公司信息（模拟数据）。

    Args:
        company: 公司名称
    """
    data = {
        "OpenAI": "成立于2015年，CEO Sam Altman，估值超1500亿美元，主要产品 GPT 系列和 ChatGPT",
        "Anthropic": "成立于2021年，CEO Dario Amodei，估值约600亿美元，主要产品 Claude 系列",
        "Google": "Alphabet 子公司，AI 部门 DeepMind，主要 AI 产品 Gemini 系列",
    }
    return data.get(company, f"暂无 {company} 的信息")


# ============================================================
# 创建 Worker Agents（专业执行者）
# ============================================================

# Worker 1：搜索 Agent — 负责信息检索
search_agent = CodeAgent(
    tools=[search_tech_news, search_company_info],
    model=model,
    name="search_agent",
    description="搜索 Agent：负责搜索科技新闻和公司信息。当需要查找事实性信息时，交给它。",
)

# Worker 2：分析 Agent — 负责数据分析和推理
# 没有搜索工具，但有 Python 计算能力
analyst_agent = CodeAgent(
    tools=[],
    model=model,
    name="analyst_agent",
    description="分析 Agent：负责数据分析、趋势判断和撰写报告。当需要对信息进行深度分析时，交给它。",
    add_base_tools=True,
)

# ============================================================
# 创建 Manager Agent（管理者）
# ============================================================
# Manager 自己没有工具，但管理两个 Worker
# 它的职责是：理解任务 → 拆分子任务 → 分发给合适的 Worker → 汇总结果

manager = CodeAgent(
    tools=[],
    model=model,
    managed_agents=[search_agent, analyst_agent],
)

# ============================================================
# 运行多 Agent 系统
# ============================================================
print("=" * 60)
print("多 Agent 协作：科技行业分析")
print("=" * 60)
print("Manager 会将任务拆分，分发给 Search Agent 和 Analyst Agent\n")

result = manager.run(
    "请帮我分析当前 AI 行业的竞争格局。"
    "先搜索 AI 领域的最新新闻和主要公司（OpenAI、Anthropic、Google）的信息，"
    "然后基于这些信息写一份简短的行业分析报告。"
)

print(f"\n{'=' * 60}")
print("最终报告：")
print("=" * 60)
print(result)
