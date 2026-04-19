"""
08_gradio_interactive.py — Gradio UI 与多轮对话

学习目标：
1. 使用 GradioUI 创建可视化的 Agent 交互界面
2. 理解 reset=False 实现多轮对话的机制
3. 观察 Agent 的思考过程在 UI 中的展示

核心概念：
- GradioUI：smolagents 内置的 Web 界面，可视化 Agent 的思考和执行过程
- reset=False：保留 Agent 记忆，实现连续对话
- agent.interrupt()：中断正在运行的 Agent

运行方式：
    python 08_gradio_interactive.py
    # 然后在浏览器打开 http://localhost:7860
"""

import os
from dotenv import load_dotenv
from smolagents import CodeAgent, LiteLLMModel, tool, GradioUI

load_dotenv()

model = LiteLLMModel(model_id="deepseek/deepseek-chat", temperature=0.3)


# ============================================================
# 定义一组有趣的工具
# ============================================================

@tool
def search_knowledge(topic: str) -> str:
    """
    搜索知识库中的信息（模拟数据）。

    Args:
        topic: 要搜索的主题，如 "Python", "机器学习", "深度学习"
    """
    knowledge = {
        "Python": "Python 是一种高级编程语言，由 Guido van Rossum 于 1991 年创建。"
                  "它以简洁的语法和丰富的生态系统著称，广泛用于 Web 开发、数据科学、AI 等领域。",
        "机器学习": "机器学习是 AI 的子领域，让计算机从数据中学习模式而无需显式编程。"
                   "主要分为监督学习、无监督学习和强化学习三大类。",
        "深度学习": "深度学习是机器学习的子集，使用多层神经网络处理复杂模式。"
                   "代表架构包括 CNN（图像）、RNN/Transformer（序列）、GAN（生成）等。",
        "Agent": "AI Agent 是能自主感知环境、做出决策并执行行动的智能系统。"
                "核心组件包括：规划（Planning）、记忆（Memory）、工具（Tools）、行动（Action）。"
                "经典范式有 ReAct、Plan-and-Execute、Reflexion 等。",
        "RAG": "RAG（检索增强生成）结合了信息检索和文本生成。"
              "流程：用户提问 → 检索相关文档 → 将文档作为上下文 → LLM 生成答案。"
              "关键技术：向量检索、混合搜索、Rerank 重排序、RAGAS 评估。",
    }
    for key, value in knowledge.items():
        if key.lower() in topic.lower() or topic.lower() in key.lower():
            return value
    return f"知识库中暂无关于 '{topic}' 的信息。可搜索的主题：{', '.join(knowledge.keys())}"


@tool
def create_quiz(topic: str, difficulty: str = "中等") -> str:
    """
    根据主题生成测验题目（模拟）。

    Args:
        topic: 测验主题
        difficulty: 难度级别，可选 "简单", "中等", "困难"
    """
    quizzes = {
        "Python": {
            "简单": "Q: Python 中用什么关键字定义函数？\nA: def",
            "中等": "Q: Python 的 GIL 是什么？它如何影响多线程？\nA: 全局解释器锁，同一时刻只允许一个线程执行 Python 字节码",
            "困难": "Q: 解释 Python 的 MRO（方法解析顺序）和 C3 线性化算法。\nA: MRO 决定多继承时方法的查找顺序，C3 算法保证单调性和局部优先级",
        },
        "机器学习": {
            "简单": "Q: 监督学习和无监督学习的主要区别是什么？\nA: 监督学习有标签数据，无监督学习没有",
            "中等": "Q: 什么是过拟合？如何防止？\nA: 模型在训练集上表现好但泛化差。防止方法：正则化、Dropout、数据增强、早停",
            "困难": "Q: 推导 SVM 的对偶问题，并解释核技巧的数学原理。\nA: 通过拉格朗日乘子法将原问题转为对偶问题，核函数在高维空间计算内积而无需显式映射",
        },
    }
    for key in quizzes:
        if key.lower() in topic.lower():
            return quizzes[key].get(difficulty, quizzes[key]["中等"])
    return f"暂无关于 '{topic}' 的测验题目"


# ============================================================
# 创建 Agent
# ============================================================
agent = CodeAgent(
    tools=[search_knowledge, create_quiz],
    model=model,
    add_base_tools=True,
    instructions=(
        "你是一个 AI 学习助手。你可以帮用户搜索知识、生成测验题目。"
        "回答时请用中文，语气友好专业。"
        "如果用户想测试自己的知识，主动用 create_quiz 工具生成题目。"
    ),
)


# ============================================================
# 方式一：GradioUI（推荐，可视化）
# ============================================================
print("=" * 60)
print("启动 Gradio UI")
print("=" * 60)
print()
print("即将启动 Web 界面，你可以：")
print("  1. 在浏览器中与 Agent 对话")
print("  2. 观察 Agent 的思考过程（Thought → Code → Observation）")
print("  3. 进行多轮对话（Agent 会记住上下文）")
print()
print("试试这些对话：")
print('  - "帮我了解一下什么是 RAG"')
print('  - "给我出一道关于 Python 的困难题目"')
print('  - "刚才说的 RAG 中，Rerank 是什么意思？"（测试多轮记忆）')
print()
print("按 Ctrl+C 退出")
print()

# 启动 Gradio 界面
# 底层实现：用户每次输入时调用 agent.run(user_input, reset=False)
# reset=False 保留了 Agent 的记忆，实现多轮对话
GradioUI(agent).launch()


# ============================================================
# 方式二：代码中实现多轮对话（不启动 UI）
# ============================================================
# 如果你不想启动 Gradio，可以用下面的代码实现多轮对话：
#
# # 第一轮
# result1 = agent.run("什么是 Agent？")
# print(f"回答 1: {result1}")
#
# # 第二轮 — reset=False 保留记忆
# result2 = agent.run("它和普通 LLM 有什么区别？", reset=False)
# print(f"回答 2: {result2}")
#
# # 第三轮 — Agent 记得之前的对话
# result3 = agent.run("给我出一道相关的测验题", reset=False)
# print(f"回答 3: {result3}")
