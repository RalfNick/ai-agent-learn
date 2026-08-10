import asyncio
import unittest


class McpAdapterTests(unittest.TestCase):
    def test_tool_discovery_and_shared_output(self):
        try:
            from mcp.server.mcpserver import MCPServer  # noqa: F401
        except ImportError:
            self.skipTest("MCP Python SDK 2.x is not installed")

        from agent_cli_lab.mcp_server import server

        tools = asyncio.run(server.list_tools())
        self.assertEqual({tool.name for tool in tools}, {"list_runs", "get_run", "export_report"})
        result = asyncio.run(server.call_tool("get_run", {"run_id": "run-003"}))
        self.assertFalse(result.is_error)
        self.assertEqual(result.structured_content["id"], "run-003")


if __name__ == "__main__":
    unittest.main()
