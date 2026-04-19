"""
03_code_vs_toolcalling.py — CodeAgent vs ToolCallingAgent 对比

学习目标：
1. 理解 smolagents 两种 Agent 的本质区别
2. 观察同一任务下两种 Agent 的不同执行方式
3. 理解为什么 CodeAgent 通常更高效

核心区别：
┌─────────────────┬──────────────────────────────────────┐
│  CodeAgent      │ LLM 生成 Python 代码，直接执行        │
│                 │ 支持循环、条件、变量 → 更灵活          │
│                 │ 研究表明比 JSON 方式少 30% 步骤        │
├─────────────────┼──────────────────────────────────────┤
│ ToolCallingAgent│ LLM 生成 JSON 格式的工具调用           │
│                 │ 类似 OpenAI function calling          │
│                 │ 更安全可控，但表达力有限               │
└─────────────────┴──────────────────────────────────────┘

运行方式：
    python 03_code_vs_toolcalling.py
"""

import os
from dotenv import load_dotenv
from smolagents import CodeAgent, ToolCallingAgent, LiteLLMModel, tool

load_dotenv()

model = LiteLLMModel(model_id="deepseek/deepseek-chat", temperature=0.3)


# 定义一个简单工具供两种 Agent 使用
@tool
def lookup_population(country: str) -> str:
    """
    查询国家的人口数据（模拟数据）。

    Args:
        country: 国家名称，如 "中国", "美国", "日本"
    """
    data = {
        "中国": "14.1 亿",
        "美国": "3.3 亿",
        "日本": "1.25 亿",
        "印度": "14.4 亿",
        "巴西": "2.15 亿",
        "德国": "0.84 亿",
    }
    return data.get(country, f"暂无 {country} 的人口数据")


@tool
def lookup_gdp(country: str) -> str:
    """
    查询国家的 GDP 数据（模拟数据，单位：万亿美元）。

    Args:
        country: 国家名称
    """
    data = {
        "中国": "17.8 万亿美元",
        "美国": "25.5 万亿美元",
        "日本": "4.2 万亿美元",
        "印度": "3.7 万亿美元",
        "巴西": "1.9 万亿美元",
        "德国": "4.1 万亿美元",
    }
    return data.get(country, f"暂无 {country} 的 GDP 数据")


tools = [lookup_population, lookup_gdp]
task = "比较中国、美国和日本的人口和 GDP，哪个国家的人均 GDP 最高？"

# ============================================================
# CodeAgent：LLM 写代码来解决
# ============================================================
print("=" * 60)
print("CodeAgent 执行")
print("=" * 60)
print("（LLM 会写 Python 代码，可以用循环批量查询，用变量计算）\n")

code_agent = CodeAgent(tools=tools, model=model)
result_code = code_agent.run(task)
print(f"\nCodeAgent 答案: {result_code}")
print(f"CodeAgent 步骤数: {len([s for s in code_agent.logs if hasattr(s, 'model_output')])}")

# ============================================================
# ToolCallingAgent：LLM 用 JSON 调用工具
# ============================================================
print("\n" + "=" * 60)
print("ToolCallingAgent 执行")
print("=" * 60)
print("（LLM 每次只能调用一个工具，需要更多轮次）\n")

tc_agent = ToolCallingAgent(tools=tools, model=model)
result_tc = tc_agent.run(task)
print(f"\nToolCallingAgent 答案: {result_tc}")
print(f"ToolCallingAgent 步骤数: {len([s for s in tc_agent.logs if hasattr(s, 'model_output')])}")

# ============================================================
# 对比总结
# ============================================================
print("\n" + "=" * 60)
print("对比总结")
print("=" * 60)
print("""
CodeAgent 优势：
  - 可以在一步内用循环查询多个国家（减少 LLM 调用次数）
  - 可以直接用 Python 做数学计算（不需要额外的计算器工具）
  - 可以用变量存储中间结果，逻辑更清晰

ToolCallingAgent 优势：
  - 不执行任意代码，更安全
  - 输出格式固定（JSON），更容易解析和监控
  - 适合对安全性要求高的生产环境

选择建议：
  - 学习/原型阶段 → CodeAgent（更灵活高效）
  - 生产环境 → ToolCallingAgent（更安全可控）
  - 或者 CodeAgent + 沙箱执行（如 Docker/E2B）
""")
