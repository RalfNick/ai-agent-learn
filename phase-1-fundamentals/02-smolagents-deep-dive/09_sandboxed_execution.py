"""
09_sandboxed_execution.py — 安全沙箱执行

学习目标：
1. 理解 CodeAgent 的安全风险：LLM 生成的代码可能有害
2. 了解 smolagents 的本地安全执行器（LocalPythonExecutor）
3. 掌握四种沙箱方案：E2B、Docker、Blaxel、WASM
4. 理解 additional_authorized_imports 的作用

核心概念：
- LocalPythonExecutor：基于 AST 的安全 Python 解释器（默认）
- executor_type：选择代码执行环境（"local"/"e2b"/"docker"/"blaxel"/"wasm"）
- additional_authorized_imports：允许 Agent 使用的额外 Python 模块
- 安全风险：Prompt 注入、供应链攻击、恶意代码执行

运行方式：
    python 09_sandboxed_execution.py
"""

import os
from dotenv import load_dotenv
from smolagents import CodeAgent, LiteLLMModel

load_dotenv()

model = LiteLLMModel(model_id="deepseek/deepseek-chat", temperature=0.3)


# ============================================================
# 1. 理解默认的安全执行器
# ============================================================
print("=" * 60)
print("1. LocalPythonExecutor — 默认的安全执行器")
print("=" * 60)

# smolagents 不使用标准 Python 解释器执行 LLM 生成的代码
# 而是使用自研的 LocalPythonExecutor：
#   - 解析代码的 AST（抽象语法树）
#   - 逐操作执行，检查每一步是否安全
#   - 默认禁止所有 import（除了安全白名单）
#   - 禁止访问子模块（防止通过合法模块访问危险子模块）
#   - 限制最大操作数（防止无限循环）

from smolagents.local_python_executor import LocalPythonExecutor

executor = LocalPythonExecutor(["numpy"])

print("测试安全执行器的防护能力：\n")

# 测试 1：非法命令
print("测试 1 — 非法 shell 命令：")
try:
    executor("!echo 'hacked'")
except Exception as e:
    print(f"  已拦截: {type(e).__name__}")

# 测试 2：未授权的 import
print("\n测试 2 — 未授权的 import：")
try:
    executor("import subprocess")
except Exception as e:
    print(f"  已拦截: {e}")

# 测试 3：无限循环
print("\n测试 3 — 无限循环：")
try:
    executor("while True: pass")
except Exception as e:
    print(f"  已拦截: {type(e).__name__}")

# 测试 4：合法操作
print("\n测试 4 — 合法的 numpy 操作：")
result = executor("import numpy as np; result = np.mean([1, 2, 3, 4, 5]); result")
print(f"  正常执行: np.mean([1,2,3,4,5]) = {result}")


# ============================================================
# 2. additional_authorized_imports
# ============================================================
print("\n" + "=" * 60)
print("2. additional_authorized_imports — 授权额外模块")
print("=" * 60)

# 默认安全白名单包括：math, random, re, datetime, collections, itertools 等
# 如果 Agent 需要使用其他模块，必须显式授权

agent_with_imports = CodeAgent(
    tools=[],
    model=model,
    additional_authorized_imports=["json", "statistics"],
    add_base_tools=True,
)

print("授权了 json 和 statistics 模块")
result = agent_with_imports.run(
    "用 statistics 模块计算 [23, 45, 67, 89, 12, 34, 56, 78] 的均值、中位数和标准差"
)
print(f"结果: {result}")

print("\n通配符授权示例：")
print('  additional_authorized_imports=["numpy.*"]')
print("  → 允许 numpy、numpy.random、numpy.linalg 等所有子模块")


# ============================================================
# 3. 沙箱方案对比
# ============================================================
print("\n" + "=" * 60)
print("3. 四种沙箱方案对比")
print("=" * 60)

comparison = """
┌──────────────┬────────────┬──────────┬──────────┬──────────────┐
│ 方案          │ 安全级别    │ 设置难度  │ 多Agent  │ 适用场景      │
├──────────────┼────────────┼──────────┼──────────┼──────────────┤
│ Local        │ **        │ 零配置    │ 支持     │ 开发/学习     │
│ (默认)       │ AST 沙箱   │          │          │              │
├──────────────┼────────────┼──────────┼──────────┼──────────────┤
│ E2B          │ ****      │ 简单      │ 需额外   │ 云端生产环境   │
│              │ 云端隔离    │ 需 API Key│ 配置     │              │
├──────────────┼────────────┼──────────┼──────────┼──────────────┤
│ Docker       │ ****      │ 中等      │ 需额外   │ 本地生产环境   │
│              │ 容器隔离    │ 需 Docker │ 配置     │              │
├──────────────┼────────────┼──────────┼──────────┼──────────────┤
│ Blaxel       │ ****      │ 简单      │ 需额外   │ 低延迟生产    │
│              │ 云端 VM     │ 需账号    │ 配置     │ (<25ms 启动)  │
├──────────────┼────────────┼──────────┼──────────┼──────────────┤
│ WASM         │ ***       │ 简单      │ 不支持   │ 浏览器/边缘   │
│              │ WebAssembly │ 需 Deno  │          │              │
└──────────────┴────────────┴──────────┴──────────┴──────────────┘
"""
print(comparison)


# ============================================================
# 4. 各沙箱方案的使用方式
# ============================================================
print("=" * 60)
print("4. 各沙箱方案的使用代码")
print("=" * 60)

print("""
# === E2B 云端沙箱 ===
# pip install 'smolagents[e2b]'
# 设置环境变量 E2B_API_KEY

with CodeAgent(model=model, tools=[], executor_type="e2b") as agent:
    agent.run("计算斐波那契数列第 100 项")
# with 语句确保沙箱资源被正确清理


# === Docker 容器沙箱 ===
# pip install 'smolagents[docker]'
# 需要本地安装 Docker

with CodeAgent(model=model, tools=[], executor_type="docker") as agent:
    agent.run("计算斐波那契数列第 100 项")


# === Blaxel 云端 VM ===
# pip install 'smolagents[blaxel]'
# 设置 BL_API_KEY 和 BL_WORKSPACE

with CodeAgent(model=model, tools=[], executor_type="blaxel") as agent:
    agent.run("计算斐波那契数列第 100 项")


# === WASM (WebAssembly) ===
# 需要安装 Deno

agent = CodeAgent(model=model, tools=[], executor_type="wasm")
agent.run("计算斐波那契数列第 100 项")
""")


# ============================================================
# 5. 安全威胁模型
# ============================================================
print("=" * 60)
print("5. Agent 代码执行的安全威胁")
print("=" * 60)

print("""
Agent 执行 LLM 生成的代码面临四类威胁：

1. LLM 自身错误
   - LLM 可能无意中生成有害命令
   - 风险较低，但确实发生过

2. 供应链攻击
   - 使用被篡改的 LLM 模型
   - 使用知名模型 + 可信推理服务可降低风险

3. Prompt 注入
   - Agent 浏览网页时遇到恶意指令
   - 恶意网站在内容中嵌入对抗性指令
   - 这是最常见的攻击向量

4. 公开暴露的 Agent
   - 将 Agent 暴露给公网用户
   - 恶意用户构造对抗性输入
   - 必须使用沙箱 + 输入验证

防御策略：
  开发阶段 → Local executor（默认）足够
  生产阶段 → E2B/Docker/Blaxel 沙箱 + 输入验证 + 输出过滤
  高安全场景 → 整个 Agent 系统运行在沙箱中
""")

print("=" * 60)
print("总结")
print("=" * 60)
print("""
学习阶段的建议：
  1. 使用默认的 Local executor，足够安全
  2. 通过 additional_authorized_imports 按需开放模块
  3. 不要授权 subprocess, shutil 等危险模块

生产阶段的建议：
  1. 必须使用 E2B/Docker/Blaxel 沙箱
  2. 设置资源限制（内存、CPU、执行时间）
  3. 实施输入验证和输出过滤
  4. 监控和审计所有代码执行
""")
