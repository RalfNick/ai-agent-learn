"""
01_hello_agent.py — 最简 smolagents Agent

学习目标：
1. 理解 Agent 的基本构成：模型 + 工具 + 循环
2. 理解 CodeAgent 的工作方式：LLM 写 Python 代码来完成任务
3. 观察 Agent 的思考-行动-观察循环

运行方式：
    cp .env.example .env  # 填入你的 DeepSeek API Key
    pip install -r requirements.txt
    python 01_hello_agent.py
"""

import os
from dotenv import load_dotenv
from smolagents import CodeAgent, LiteLLMModel

load_dotenv()

# ============================================================
# 第一步：创建模型
# ============================================================
# LiteLLMModel 是 smolagents 的通用模型适配器
# 通过 "deepseek/deepseek-chat" 指定使用 DeepSeek 的对话模型
# 它会自动读取环境变量 DEEPSEEK_API_KEY
model = LiteLLMModel(
    model_id="deepseek/deepseek-chat",
    temperature=0.7,
)

# ============================================================
# 第二步：创建 Agent
# ============================================================
# CodeAgent 是 smolagents 的核心 — LLM 会写 Python 代码来完成任务
# tools=[] 表示没有外部工具，Agent 只能用纯 Python 推理
# add_base_tools=True 会添加一些内置工具（如 PythonInterpreterTool）
agent = CodeAgent(
    tools=[],
    model=model,
    add_base_tools=True,  # 添加内置基础工具
)

# ============================================================
# 第三步：运行 Agent
# ============================================================
# agent.run() 启动 Agent 循环：
#   1. LLM 收到任务，思考如何解决
#   2. LLM 生成 Python 代码
#   3. 代码被执行，结果作为"观察"反馈给 LLM
#   4. LLM 决定是否需要继续，或调用 final_answer() 结束
# print("=" * 60)
# print("示例 1：简单计算（Agent 会写 Python 代码来算）")
# print("=" * 60)

# result = agent.run("计算从 1 到 100 的所有质数之和")
# print(f"\n最终答案: {result}")

# ============================================================
# 第四步：查看 Agent 的执行日志
# ============================================================
# agent.logs 记录了每一步的详细信息
# 这是理解 Agent 内部运作的关键
# print("\n" + "=" * 60)
# print("Agent 执行日志（观察思考-行动-观察循环）")
# print("=" * 60)

# for i, step in enumerate(agent.logs):
#     if hasattr(step, "model_output"):
#         print(f"\n--- Step {i} ---")
#         # model_output 是 LLM 的原始输出（包含思考过程和代码）
#         print(f"LLM 输出: {step.model_output[:200]}...")
#     if hasattr(step, "observations"):
#         print(f"观察结果: {step.observations[:200]}...")

# ============================================================
# 第五步：多轮对话 — 理解 Agent 的记忆
# ============================================================
print("\n" + "=" * 60)
print("示例 2：Agent 的推理能力")
print("=" * 60)

result2 = agent.run(
    "斐波那契数列的第 20 项是多少？请同时告诉我计算过程。"
)
print(f"\n最终答案: {result2}")
