"""
05_agent_internals.py — 深入 Agent 内部机制

学习目标：
1. 理解 ReAct 循环的内部实现：Thought → Code → Observation
2. 掌握 Agent 的日志系统和记忆机制
3. 学会通过 system prompt、instructions、max_steps 调控 Agent 行为

核心概念：
- agent.logs：每一步的详细执行记录
- agent.write_memory_to_messages()：将记忆转为 LLM 可读的消息格式
- prompt_templates：Agent 的 system prompt 模板（可定制）
- max_steps：控制 Agent 最大推理步数，防止无限循环
- instructions：向 Agent 注入自定义指令

运行方式：
    cd phase-1-fundamentals/02-smolagents-deep-dive
    pip install -r requirements.txt
    python 05_agent_internals.py
"""

import os
from dotenv import load_dotenv
from smolagents import CodeAgent, LiteLLMModel, tool

load_dotenv()

model = LiteLLMModel(model_id="deepseek/deepseek-chat", temperature=0.3)


# ============================================================
# 1. 查看 Agent 的 System Prompt — 理解 LLM 看到了什么
# ============================================================
print("=" * 60)
print("1. Agent 的 System Prompt 结构")
print("=" * 60)

agent = CodeAgent(tools=[], model=model, add_base_tools=True)

# Agent 的 system prompt 是一个 Jinja2 模板
# 初始化时，工具描述、managed_agents 描述会被注入
system_prompt = agent.prompt_templates["system_prompt"]
print(f"System prompt 长度: {len(system_prompt)} 字符")
print(f"前 500 字符预览:\n{system_prompt[:500]}")
print("...")

# ============================================================
# 2. 使用 instructions 定制 Agent 行为
# ============================================================
# instructions 会被追加到 system prompt 末尾
# 这是最简单的定制方式，不需要修改模板
print("\n" + "=" * 60)
print("2. 使用 instructions 定制 Agent")
print("=" * 60)

custom_agent = CodeAgent(
    tools=[],
    model=model,
    add_base_tools=True,
    instructions="你是一个数学教授。回答问题时，请用通俗易懂的方式解释推理过程，就像在给学生上课一样。",
)

result = custom_agent.run("为什么 0.1 + 0.2 不等于 0.3？请解释原因。")
print(f"答案: {result}")


# ============================================================
# 3. 深入分析 agent.logs — 理解每一步发生了什么
# ============================================================
print("\n" + "=" * 60)
print("3. 深入分析 Agent 执行日志")
print("=" * 60)

@tool
def fibonacci(n: int) -> str:
    """
    计算斐波那契数列的第 n 项。

    Args:
        n: 要计算的项数（从第 1 项开始）
    """
    if n <= 0:
        return "n 必须是正整数"
    a, b = 0, 1
    for _ in range(n - 1):
        a, b = b, a + b
    return f"斐波那契数列第 {n} 项是 {b}"


log_agent = CodeAgent(tools=[fibonacci], model=model, max_steps=5)
result = log_agent.run("斐波那契数列第 10 项和第 20 项分别是多少？它们的比值接近什么数？")
print(f"\n最终答案: {result}")


# ============================================================
# 4. write_memory_to_messages() — 查看 Agent 的记忆
# ============================================================
print("\n" + "=" * 60)
print("4. Agent 的记忆（消息格式）")
print("=" * 60)

messages = log_agent.write_memory_to_messages()
print(f"总消息数: {len(messages)}")
for msg in messages:
    content = msg.content
    print(f"content: {content}...")


# ============================================================
# 5. max_steps 控制 — 防止 Agent 陷入无限循环
# ============================================================
print("\n" + "=" * 60)
print("5. max_steps 控制演示")
print("=" * 60)

# 设置很小的 max_steps，观察 Agent 被截断的行为
limited_agent = CodeAgent(
    tools=[fibonacci],
    model=model,
    max_steps=2,  # 只允许 2 步
)

print("给 Agent 一个需要多步的任务，但只允许 2 步：")
try:
    result = limited_agent.run(
        "计算斐波那契数列第 5、10、15、20、25、30 项，然后画出增长趋势。"
    )
    print(f"答案: {result}")
except Exception as e:
    print(f"Agent 在 max_steps 内未完成: {e}")
