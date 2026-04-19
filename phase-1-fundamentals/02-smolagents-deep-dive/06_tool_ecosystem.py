"""
06_tool_ecosystem.py — smolagents 工具生态系统

学习目标：
1. 从 HuggingFace Hub 加载社区共享的工具（演示 API，网络不可用时跳过）
2. 将 Gradio Space 作为工具使用（演示 API，网络不可用时跳过）
3. 导入 LangChain 工具到 smolagents
4. 连接 MCP Server 获取工具（演示 API，需要真实 MCP Server）
5. 使用 ToolCollection 批量加载工具
6. 动态管理 Agent 工具箱

运行方式：
    python 06_tool_ecosystem.py
"""

import os
import threading
from dotenv import load_dotenv
from smolagents import CodeAgent, LiteLLMModel, tool, Tool

# 静默后台线程异常（MCP client 在 uvx 进程失败时会产生线程噪音）
threading.excepthook = lambda args: None

load_dotenv()

model = LiteLLMModel(model_id="deepseek/deepseek-chat", temperature=0.3)


# ============================================================
# 1. 从 Hub 加载工具（需要网络访问 HuggingFace）
# ============================================================
print("=" * 60)
print("1. 从 HuggingFace Hub 加载工具")
print("=" * 60)

try:
    from smolagents import load_tool

    image_tool = load_tool("m-ric/text-to-image", trust_remote_code=True)
    print(f"工具名: {image_tool.name}")
    print(f"工具描述: {image_tool.description}")
except Exception as e:
    print(f"跳过 Hub 工具加载（需要网络访问）: {type(e).__name__}")
    print("API 用法：")
    print("  tool = load_tool('username/tool-name', trust_remote_code=True)")
    print("  agent = CodeAgent(tools=[tool], model=model)")
    print("  my_tool.push_to_hub('your-username/tool-name')  # 推送自己的工具")


# ============================================================
# 2. 将 Gradio Space 作为工具（需要网络访问）
# ============================================================
print("\n" + "=" * 60)
print("2. 将 Gradio Space 变成工具")
print("=" * 60)

try:
    image_tool = Tool.from_space(
        "black-forest-labs/FLUX.1-schnell",
        name="image_generator",
        description="根据文本描述生成图片",
    )
    print(f"工具名: {image_tool.name}")
    print(f"工具描述: {image_tool.description}")
    print("Tool.from_space() 加载成功！")
except Exception as e:
    print(f"跳过 Gradio Space 工具（需要网络访问）: {type(e).__name__}")
    print("API 用法：")
    print("  tool = Tool.from_space('space-owner/space-name', name='my_tool', description='...')")
    print("  agent = CodeAgent(tools=[tool], model=model)")


# ============================================================
# 3. 导入 LangChain 工具（本地可用，实际运行）
# ============================================================
print("\n" + "=" * 60)
print("3. 从 LangChain 导入工具（实际运行）")
print("=" * 60)

try:
    from langchain_community.tools import WikipediaQueryRun
    from langchain_community.utilities import WikipediaAPIWrapper

    lc_wiki_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
    wiki_tool = Tool.from_langchain(lc_wiki_tool)

    print(f"转换后的工具名: {wiki_tool.name}")
    print(f"工具描述: {wiki_tool.description[:100]}...")

    agent = CodeAgent(tools=[wiki_tool], model=model)
    result = agent.run("用 Wikipedia 查一下图灵奖是什么，用一句话概括。")
    print(f"\n答案: {result}")

except ImportError:
    print("需要安装: pip install langchain-community wikipedia")
    print("Tool.from_langchain() 用法：")
    print("  wiki_tool = Tool.from_langchain(WikipediaQueryRun(...))")
    print("  agent = CodeAgent(tools=[wiki_tool], model=model)")


# ============================================================
# 4. 连接 MCP Server（需要真实 MCP Server）
# ============================================================
print("\n" + "=" * 60)
print("4. 连接 MCP Server 获取工具")
print("=" * 60)

# 演示用本地 stdio MCP server（filesystem server，uvx 可直接安装）
try:
    from smolagents import MCPClient
    from mcp import StdioServerParameters

    # mcp-server-filesystem 是官方提供的真实 MCP server
    # mcp-server-fetch 是官方提供的真实 MCP server，支持 HTTP 抓取
    server_params = StdioServerParameters(
        command="uvx",
        args=["mcp-server-fetch"],
        env={"UV_PYTHON": "3.12", **os.environ},
    )

    with MCPClient(server_params) as tools:
        print(f"MCP 工具列表: {[t.name for t in tools]}")
        agent = CodeAgent(tools=tools, model=model)
        result = agent.run("抓取 https://example.com 页面的标题。")
        print(f"结果: {result}")
except Exception as e:
    print(f"跳过 MCP Server 连接: {type(e).__name__}: {e}")
    print("API 用法（stdio 方式）：")
    print("  server_params = StdioServerParameters(command='uvx', args=['some-mcp-server'])")
    print("  with MCPClient(server_params) as tools:")
    print("      agent = CodeAgent(tools=tools, model=model)")

print("\nHTTP 方式连接 MCP Server：")
print("  MCPClient({'url': 'http://127.0.0.1:8000/mcp', 'transport': 'streamable-http'})")


# ============================================================
# 5. ToolCollection — 批量管理工具
# ============================================================
print("\n" + "=" * 60)
print("5. ToolCollection 批量管理工具")
print("=" * 60)

print("从 Hub Collection 加载（需要网络）：")
try:
    from smolagents import ToolCollection

    # 注意：Hub collection URL 可能失效，这里仅演示 API
    tool_collection = ToolCollection.from_hub(
        collection_slug="huggingface-tools/diffusion-tools-6630bb19a942c2306a2cdb6f",
        trust_remote_code=True,
    )
    print(f"加载工具数量: {len(tool_collection.tools)}")
    for t in tool_collection.tools:
        print(f"  - {t.name}: {t.description[:60]}...")
except Exception as e:
    print(f"跳过 ToolCollection Hub 加载（需要网络访问）: {type(e).__name__}")
    print("API 用法：")
    print("  tc = ToolCollection.from_hub('owner/collection-slug', trust_remote_code=True)")
    print("  agent = CodeAgent(tools=[*tc.tools], model=model)")

print("\n从 MCP Server 加载（需要真实 MCP Server）：")
try:
    from smolagents import ToolCollection
    from mcp import StdioServerParameters

    server_params = StdioServerParameters(
        command="uvx",
        args=["mcp-server-fetch"],
        env={"UV_PYTHON": "3.12", **os.environ},
    )
    with ToolCollection.from_mcp(server_params, trust_remote_code=True) as tc:
        print(f"MCP 工具数量: {len(tc.tools)}")
        print(f"工具列表: {[t.name for t in tc.tools]}")
except Exception as e:
    print(f"跳过 ToolCollection MCP 加载: {type(e).__name__}: {e}")
    print("API 用法：")
    print("  with ToolCollection.from_mcp(server_params) as tc:")
    print("      agent = CodeAgent(tools=[*tc.tools], model=model)")


# ============================================================
# 6. 动态管理 Agent 的工具箱（本地，实际运行）
# ============================================================
print("\n" + "=" * 60)
print("6. 动态管理 Agent 工具箱（实际运行）")
print("=" * 60)


@tool
def add_numbers(a: float, b: float) -> str:
    """
    将两个数字相加。

    Args:
        a: 第一个数字
        b: 第二个数字
    """
    return str(a + b)


@tool
def multiply_numbers(a: float, b: float) -> str:
    """
    将两个数字相乘。

    Args:
        a: 第一个数字
        b: 第二个数字
    """
    return str(a * b)


agent = CodeAgent(tools=[add_numbers], model=model)
print(f"初始工具: {list(agent.tools.keys())}")

# 动态添加工具 — agent.tools 是普通字典
agent.tools[multiply_numbers.name] = multiply_numbers
print(f"添加后工具: {list(agent.tools.keys())}")

result = agent.run("计算 (3 + 7) × 5 的结果")
print(f"答案: {result}")

# 动态移除工具
del agent.tools["add_numbers"]
print(f"移除后工具: {list(agent.tools.keys())}")

print("\n提示：不要给 Agent 太多工具，会让弱模型困惑。精选最相关的工具即可。")
