"""
02_custom_tools.py — 自定义工具系统

学习目标：
1. 理解工具（Tool）在 Agent 中的角色：扩展 LLM 的能力边界
2. 掌握两种定义工具的方式：@tool 装饰器 和 Tool 子类
3. 观察 Agent 如何自主选择和调用工具

核心概念：
- LLM 本身只能生成文本，工具让它能"做事"
- Agent 根据任务描述自动选择合适的工具
- 工具的 docstring 和类型注解是 LLM 理解工具的关键

运行方式：
    python 02_custom_tools.py
"""

import os
import json
import math
from datetime import datetime
from dotenv import load_dotenv
from smolagents import CodeAgent, LiteLLMModel, tool, Tool

load_dotenv()

model = LiteLLMModel(model_id="deepseek/deepseek-chat", temperature=0.3)

# ============================================================
# 方式一：@tool 装饰器（推荐，简单直接）
# ============================================================
# 关键点：
# - 函数名就是工具名
# - docstring 是 LLM 理解工具用途的依据（写清楚！）
# - 类型注解告诉 LLM 参数类型和返回类型
# - smolagents 会自动从这些信息生成工具描述


@tool
def get_current_time(timezone: str = "Asia/Shanghai") -> str:
    """
    获取指定时区的当前时间。

    Args:
        timezone: 时区名称，如 "Asia/Shanghai", "US/Eastern", "Europe/London"
    """
    from datetime import datetime, timezone as tz
    import zoneinfo

    try:
        zone = zoneinfo.ZoneInfo(timezone)
        now = datetime.now(zone)
        return now.strftime("%Y-%m-%d %H:%M:%S %Z")
    except Exception as e:
        return f"无法获取时区 {timezone} 的时间: {e}"


@tool
def calculator(expression: str) -> str:
    """
    计算数学表达式。支持基本运算和常用数学函数。

    Args:
        expression: 数学表达式，如 "2**10", "math.sqrt(144)", "math.pi * 5**2"
    """
    import ast
    import operator

    # 安全的数学计算 — 使用 ast.literal_eval 处理简单表达式
    # 对于包含函数调用的表达式，使用受限的命名空间
    allowed_names = {
        "math": math,
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
        "sum": sum,
        "pow": pow,
    }
    try:
        # 注意：这里仅用于学习演示，生产环境应使用更安全的表达式解析器
        code = compile(expression, "<string>", "eval")
        result = eval(code, {"__builtins__": {}}, allowed_names)  # noqa: S307 — sandboxed for learning demo
        return str(result)
    except Exception as e:
        return f"计算错误: {e}"


@tool
def weather_lookup(city: str) -> str:
    """
    查询城市的天气信息（模拟数据，用于演示工具调用）。

    Args:
        city: 城市名称，如 "北京", "上海", "东京"
    """
    # 模拟天气数据 — 实际项目中会调用真实 API
    mock_weather = {
        "北京": {"temp": 22, "condition": "晴", "humidity": 35},
        "上海": {"temp": 26, "condition": "多云", "humidity": 65},
        "东京": {"temp": 20, "condition": "小雨", "humidity": 78},
        "纽约": {"temp": 18, "condition": "阴", "humidity": 55},
    }
    if city in mock_weather:
        w = mock_weather[city]
        return f"{city}: {w['condition']}, 温度 {w['temp']}°C, 湿度 {w['humidity']}%"
    return f"暂无 {city} 的天气数据"


# ============================================================
# 方式二：Tool 子类（更灵活，适合复杂工具）
# ============================================================
# 当工具需要初始化状态、连接外部服务时，用子类更合适


class UnitConverter(Tool):
    name = "unit_converter"
    description = "单位换算工具，支持长度、重量、温度的常见单位转换"
    inputs = {
        "value": {"type": "number", "description": "要转换的数值"},
        "from_unit": {"type": "string", "description": "原始单位，如 km, mile, kg, lb, celsius, fahrenheit"},
        "to_unit": {"type": "string", "description": "目标单位"},
    }
    output_type = "string"

    # 转换表
    conversions = {
        ("km", "mile"): lambda v: v * 0.621371,
        ("mile", "km"): lambda v: v * 1.60934,
        ("kg", "lb"): lambda v: v * 2.20462,
        ("lb", "kg"): lambda v: v * 0.453592,
        ("celsius", "fahrenheit"): lambda v: v * 9 / 5 + 32,
        ("fahrenheit", "celsius"): lambda v: (v - 32) * 5 / 9,
        ("m", "ft"): lambda v: v * 3.28084,
        ("ft", "m"): lambda v: v * 0.3048,
    }

    def forward(self, value: float, from_unit: str, to_unit: str) -> str:
        key = (from_unit.lower(), to_unit.lower())
        if key in self.conversions:
            result = self.conversions[key](value)
            return f"{value} {from_unit} = {result:.4f} {to_unit}"
        return f"不支持从 {from_unit} 到 {to_unit} 的转换"


# ============================================================
# 创建 Agent 并注册所有工具
# ============================================================
agent = CodeAgent(
    tools=[
        get_current_time,
        calculator,
        weather_lookup,
        UnitConverter(),  # 子类需要实例化
    ],
    model=model,
)

# ============================================================
# 测试：Agent 自主选择工具
# ============================================================
# 关键观察点：Agent 会根据问题自动选择合适的工具
# 你不需要告诉它用哪个工具 — 这就是 Agent 的"自主决策"能力

print("=" * 60)
print("测试 1：需要时间工具")
print("=" * 60)
result1 = agent.run("现在北京时间几点了？")
print(f"答案: {result1}\n")

print("=" * 60)
print("测试 2：需要计算器 + 单位转换（组合使用工具）")
print("=" * 60)
result2 = agent.run(
    "一个圆的半径是 5 公里，它的面积是多少平方公里？"
    "另外，这个半径换算成英里是多少？"
)
print(f"答案: {result2}\n")

print("=" * 60)
print("测试 3：需要天气工具 + 推理")
print("=" * 60)
result3 = agent.run("北京和东京今天哪个城市更适合户外活动？请给出理由。")
print(f"答案: {result3}\n")
