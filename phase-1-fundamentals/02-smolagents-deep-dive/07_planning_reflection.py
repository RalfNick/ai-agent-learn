"""
07_planning_reflection.py — 规划与反思机制

学习目标：
1. 理解 planning_interval 如何让 Agent 定期"停下来想想"
2. 使用 final_answer_checks 验证 Agent 的输出质量
3. 通过 additional_args 传递复杂上下文给 Agent
4. 掌握构建高质量 Agent 的最佳实践

核心概念：
- planning_interval：每 N 步触发一次规划，Agent 会更新事实列表和下一步计划
- final_answer_checks：在 Agent 给出最终答案前进行自定义验证
- additional_args：向 Agent 传递额外的结构化数据
- 工具设计原则：好的工具描述 = 好的 Agent 表现

运行方式：
    python 07_planning_reflection.py
"""

import os
import json
from dotenv import load_dotenv
from smolagents import CodeAgent, LiteLLMModel, tool

load_dotenv()

model = LiteLLMModel(model_id="deepseek/deepseek-chat", temperature=0.3)


# ============================================================
# 1. planning_interval — 让 Agent 定期反思
# ============================================================
print("=" * 60)
print("1. Planning Interval — 定期规划与反思")
print("=" * 60)

# planning_interval=N 表示每 N 步，Agent 会暂停执行，
# 进入一个"规划步骤"：
#   - 更新已知事实列表
#   - 反思当前进展
#   - 规划接下来的步骤
# 这个步骤不调用工具，纯粹是 LLM 的思考


@tool
def search_database(query: str) -> str:
    """
    搜索产品数据库（模拟数据）。

    Args:
        query: 搜索关键词
    """
    products = {
        "手机": [
            {"name": "iPhone 16 Pro", "price": 8999, "rating": 4.8},
            {"name": "Samsung S25 Ultra", "price": 9499, "rating": 4.7},
            {"name": "Pixel 9 Pro", "price": 6999, "rating": 4.6},
        ],
        "笔记本": [
            {"name": "MacBook Pro M4", "price": 14999, "rating": 4.9},
            {"name": "ThinkPad X1 Carbon", "price": 10999, "rating": 4.5},
            {"name": "Dell XPS 15", "price": 11999, "rating": 4.6},
        ],
        "耳机": [
            {"name": "AirPods Pro 3", "price": 1899, "rating": 4.7},
            {"name": "Sony WH-1000XM6", "price": 2499, "rating": 4.8},
        ],
    }
    for key, items in products.items():
        if key in query:
            return json.dumps(items, ensure_ascii=False)
    return f"未找到与 '{query}' 相关的产品"


@tool
def get_user_budget(user_id: str) -> str:
    """
    获取用户的预算信息。

    Args:
        user_id: 用户 ID
    """
    budgets = {
        "user_001": {"total": 20000, "spent": 5000, "remaining": 15000},
        "user_002": {"total": 50000, "spent": 30000, "remaining": 20000},
    }
    info = budgets.get(user_id)
    if info:
        return json.dumps(info, ensure_ascii=False)
    return f"未找到用户 {user_id} 的预算信息"


# 启用 planning_interval=2：每 2 步反思一次
planning_agent = CodeAgent(
    tools=[search_database, get_user_budget],
    model=model,
    planning_interval=2,  # 关键参数！
    max_steps=8,
)

print("任务：帮用户做购物推荐（Agent 会每 2 步反思一次）\n")
result = planning_agent.run(
    "我是 user_001，想买一部手机和一副耳机。"
    "请根据我的预算推荐最佳组合，要求总价不超过剩余预算。"
)
print(f"\n推荐结果: {result}")


# ============================================================
# 2. final_answer_checks — 输出质量验证
# ============================================================
print("\n" + "=" * 60)
print("2. Final Answer Checks — 验证输出质量")
print("=" * 60)


# 定义验证函数：检查最终答案是否是数字
def must_be_number(final_answer, agent_memory=None) -> bool:
    """验证最终答案必须是数字"""
    try:
        float(str(final_answer))
        return True
    except (ValueError, TypeError):
        return False


# 定义验证函数：检查答案长度
def must_be_detailed(final_answer, agent_memory=None) -> bool:
    """验证最终答案必须超过 20 个字符（足够详细）"""
    return len(str(final_answer)) > 20


# 使用数字验证的 Agent
print("测试 1：要求答案必须是数字")
math_agent = CodeAgent(
    tools=[],
    model=model,
    add_base_tools=True,
    final_answer_checks=[must_be_number],
    max_steps=5,
)

result = math_agent.run("计算 2 的 10 次方")
print(f"答案: {result} (类型: {type(result).__name__})")

# 使用详细度验证的 Agent
print("\n测试 2：要求答案必须足够详细")
detail_agent = CodeAgent(
    tools=[search_database],
    model=model,
    final_answer_checks=[must_be_detailed],
    max_steps=5,
)

result = detail_agent.run("推荐一款评分最高的手机")
print(f"答案: {result}")


# ============================================================
# 3. additional_args — 传递复杂上下文
# ============================================================
print("\n" + "=" * 60)
print("3. Additional Args — 传递复杂上下文")
print("=" * 60)

# additional_args 让你可以传递任何 Python 对象给 Agent
# Agent 可以在生成的代码中直接使用这些变量

context_agent = CodeAgent(tools=[], model=model, add_base_tools=True)

# 传递一个数据表给 Agent 分析
sales_data = {
    "Q1": {"revenue": 1200000, "cost": 800000, "customers": 450},
    "Q2": {"revenue": 1500000, "cost": 900000, "customers": 520},
    "Q3": {"revenue": 1100000, "cost": 750000, "customers": 380},
    "Q4": {"revenue": 1800000, "cost": 1000000, "customers": 600},
}

result = context_agent.run(
    "分析这份销售数据，找出利润率最高的季度和客户增长最快的季度。",
    additional_args={"sales_data": sales_data},
)
print(f"分析结果: {result}")


# ============================================================
# 4. 工具设计最佳实践
# ============================================================
print("\n" + "=" * 60)
print("4. 工具设计最佳实践对比")
print("=" * 60)

# ❌ 差的工具设计
@tool
def bad_weather(loc: str, dt: str) -> str:
    """
    返回天气。

    Args:
        loc: 地点
        dt: 时间
    """
    return str([28.0, 0.35, 0.85])


# ✅ 好的工具设计
@tool
def good_weather(location: str, date_time: str) -> str:
    """
    获取指定地点和时间的天气预报。

    Args:
        location: 地点名称，格式如 "北京市海淀区" 或 "Tokyo, Japan"
        date_time: 日期时间，格式为 "YYYY-MM-DD HH:MM"，如 "2025-03-15 14:00"
    """
    # 模拟数据
    temperature = 28.0
    rain_risk = 0.35
    wave_height = 0.85

    return (
        f"{location} 在 {date_time} 的天气预报：\n"
        f"  温度: {temperature}°C\n"
        f"  降雨概率: {rain_risk * 100:.0f}%\n"
        f"  浪高: {wave_height}m\n"
        f"  建议: {'适合户外活动' if rain_risk < 0.5 else '建议携带雨具'}"
    )


print("差的工具设计问题：")
print("  - 参数名不清晰（loc, dt）")
print("  - 描述太简略，LLM 不知道格式要求")
print("  - 返回原始数组，LLM 不知道每个值的含义")
print()
print("好的工具设计要点：")
print("  - 参数名语义明确（location, date_time）")
print("  - 描述包含格式要求和示例")
print("  - 返回人类可读的格式化字符串")
print("  - 包含错误处理和有用的建议")
print()
print("核心原则：想象你是一个第一次使用这个工具的人，")
print("工具的描述是否足够让你正确使用它？")
