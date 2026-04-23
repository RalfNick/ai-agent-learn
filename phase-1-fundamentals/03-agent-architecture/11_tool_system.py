"""
11_tool_system.py — Schema-Driven 工具系统

从 smolagents 和 langchain 源码中提炼工具系统的核心设计模式。
展示：Schema 驱动、装饰器注册、类继承注册、动态管理、Prompt 生成。

对应文章第 3 章：工具系统
"""

from __future__ import annotations
import inspect
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, get_type_hints


# ── 1. 工具基类（Tool Abstraction）──────────────────────────────
# smolagents: Tool 类 → name/description/inputs/output_type
# langchain:  BaseTool → name/description/args_schema
# 共性：Schema-Driven，LLM 通过 schema 理解工具能力

@dataclass
class ToolSchema:
    """工具的元数据描述，LLM 通过它理解工具"""
    name: str
    description: str
    parameters: dict[str, dict[str, str]]  # {param_name: {type, description}}
    return_type: str = "string"


class Tool(ABC):
    """工具基类 — 类继承方式（适合复杂/有状态的工具）"""
    name: str
    description: str
    parameters: dict[str, dict[str, str]]
    return_type: str = "string"

    @abstractmethod
    def forward(self, **kwargs) -> Any:
        ...

    def __call__(self, **kwargs) -> Any:
        return self.forward(**kwargs)

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(self.name, self.description, self.parameters, self.return_type)


# ── 2. 装饰器注册（Decorator Pattern）──────────────────────────
# smolagents: @tool 装饰器 → 从函数签名和 docstring 自动提取 schema
# langchain:  @tool 装饰器 → 类似机制

def tool(func: Callable) -> Tool:
    """从函数自动生成 Tool 实例（装饰器模式）"""
    hints = get_type_hints(func)
    sig = inspect.signature(func)
    doc = inspect.getdoc(func) or ""

    # 从 docstring 提取参数描述（简化版）
    param_descriptions = {}
    for line in doc.split("\n"):
        line = line.strip()
        if ":" in line and not line.startswith("Returns"):
            parts = line.split(":", 1)
            if len(parts) == 2:
                param_descriptions[parts[0].strip()] = parts[1].strip()

    # 构建参数 schema
    type_map = {str: "string", int: "integer", float: "number", bool: "boolean"}
    parameters = {}
    for param_name, param in sig.parameters.items():
        hint = hints.get(param_name, str)
        parameters[param_name] = {
            "type": type_map.get(hint, "string"),
            "description": param_descriptions.get(param_name, ""),
        }

    # 创建 Tool 子类实例（必须实现 forward 以满足 ABC 约束）
    class SimpleTool(Tool):
        def forward(self, **kwargs) -> Any:
            return func(**kwargs)

    instance = SimpleTool()
    instance.name = func.__name__
    instance.description = doc.split("\n")[0]
    instance.parameters = parameters
    instance.return_type = type_map.get(hints.get("return", str), "string")
    return instance


# ── 3. 工具注册表（Tool Registry）──────────────────────────────
# smolagents: self.tools = {tool.name: tool for tool in tools}
# langchain:  ToolNode 管理工具集合

class ToolRegistry:
    """工具注册表 — 支持动态增删"""

    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, t: Tool) -> None:
        self._tools[t.name] = t

    def register_batch(self, tools: list[Tool]) -> None:
        for t in tools:
            self._tools[t.name] = t

    def has(self, name: str) -> bool:
        return name in self._tools

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def execute(self, name: str, **kwargs) -> Any:
        t = self.get(name)
        if not t:
            raise ValueError(f"工具 '{name}' 未注册")
        return t(**kwargs)

    def to_prompt(self, style: str = "code") -> str:
        """生成工具描述，注入到 System Prompt 中"""
        lines = []
        for t in self._tools.values():
            if style == "code":
                # smolagents CodeAgent 风格：生成函数签名
                args = ", ".join(f"{k}: {v['type']}" for k, v in t.parameters.items())
                lines.append(f"def {t.name}({args}) -> {t.return_type}:")
                lines.append(f'    """{t.description}"""')
            else:
                # langchain ToolCallingAgent 风格：JSON 描述
                lines.append(f"{t.name}: {t.description}")
                lines.append(f"  参数: {json.dumps(t.parameters, ensure_ascii=False)}")
            lines.append("")
        return "\n".join(lines)

    @property
    def tool_names(self) -> list[str]:
        return list(self._tools.keys())


# ── 4. 演示 ─────────────────────────────────────────────────────

# 方式一：装饰器注册
@tool
def weather_lookup(location: str) -> str:
    """查询指定城市的天气信息
    location: 城市名称，如 '北京'
    """
    mock_data = {"北京": "晴，25°C", "上海": "多云，22°C", "深圳": "阵雨，28°C"}
    return mock_data.get(location, f"{location}: 暂无数据")


# 方式二：类继承注册
class UnitConverter(Tool):
    name = "unit_converter"
    description = "单位换算工具，支持长度和重量"
    parameters = {
        "value": {"type": "number", "description": "数值"},
        "from_unit": {"type": "string", "description": "源单位"},
        "to_unit": {"type": "string", "description": "目标单位"},
    }
    return_type = "string"

    CONVERSIONS = {
        ("km", "mile"): 0.621371,
        ("kg", "lb"): 2.20462,
        ("m", "ft"): 3.28084,
    }

    def forward(self, value: float, from_unit: str, to_unit: str) -> str:
        key = (from_unit, to_unit)
        if key in self.CONVERSIONS:
            result = value * self.CONVERSIONS[key]
            return f"{value} {from_unit} = {result:.2f} {to_unit}"
        return f"不支持 {from_unit} → {to_unit} 的换算"


if __name__ == "__main__":
    registry = ToolRegistry()
    registry.register(weather_lookup)
    registry.register(UnitConverter())

    print("=== 已注册工具 ===")
    print(f"工具列表: {registry.tool_names}\n")

    print("=== Code 风格 Prompt（smolagents CodeAgent）===")
    print(registry.to_prompt(style="code"))

    print("=== JSON 风格 Prompt（langchain ToolCallingAgent）===")
    print(registry.to_prompt(style="json"))

    print("=== 工具调用 ===")
    print(registry.execute("weather_lookup", location="北京"))
    print(registry.execute("unit_converter", value=10, from_unit="km", to_unit="mile"))

    print("\n=== 动态管理：移除工具 ===")
    registry.unregister("weather_lookup")
    print(f"剩余工具: {registry.tool_names}")
