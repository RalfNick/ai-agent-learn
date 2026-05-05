"""
03_autonomous_coder.py — Claude SDK 自主代码审查 Agent：读代码 → 分析 → 生成修复建议

设计思考：
前面两个文件展示了 Claude SDK 的基础循环和 Guardrails/Handoff 协议。
这个文件展示 Claude SDK 最独特的价值：作为"自主行动者"与环境交互。

Phase 1 的 Agent 只能调用预定义的工具函数（calculate、search 等）。
LangGraph 用图结构管理工具调用的控制流。
CrewAI 在角色定义中指定 Agent 能做什么。

本文件实现一个自主代码审查 Agent，包含：

1. AST 静态分析引擎 — 无需 LLM 的结构级分析
   - 圈复杂度计算
   - 函数/类统计
   - Import 分析
   - 安全问题模式匹配（危险函数、硬编码密钥）
   - 反模式检测（裸 except、可变默认参数）

2. LLM 驱动的代码审查 — 语义级分析
   - 命名质量、设计模式建议、潜在 bug 检测

3. 自主工作流
   - 扫描目录 → AST 分析 → 按需 LLM 审查 → 生成报告

核心洞察：AI Agent 在代码审查中的角色不是"替代人"，而是"第一道过滤器"——
把机械性检查（复杂度、规范、安全模式）自动化，让人专注于架构和设计。

运行方式：
    # 仅 AST 静态分析（不需要 API key）
    python 03_autonomous_coder.py .

    # 包含 LLM 深度审查（需要 ANTHROPIC_API_KEY）
    python 03_autonomous_coder.py /path/to/project
"""

from __future__ import annotations

import ast
import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from anthropic import Anthropic
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

load_dotenv()
console = Console()
ANTHROPIC_MODEL = "claude-sonnet-4-6"


class IssueSeverity(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class CodeIssue:
    file: str
    line: int
    severity: IssueSeverity
    category: str
    title: str
    description: str
    suggestion: str = ""
    code_snippet: str = ""


@dataclass
class FileMetrics:
    file: str
    total_lines: int = 0
    code_lines: int = 0
    comment_lines: int = 0
    blank_lines: int = 0
    num_functions: int = 0
    num_classes: int = 0
    num_imports: int = 0
    max_function_length: int = 0
    max_complexity: int = 0
    avg_complexity: float = 0.0
    maintainability_index: float = 0.0


class ASTAnalyzer:
    """基于 AST 的代码静态分析器

    不依赖 LLM，纯 AST 遍历 + 规则匹配。
    快速、零 token 开销、100% 确定性。
    """

    DANGEROUS_CALLS: dict[str, IssueSeverity] = {
        "exec": IssueSeverity.CRITICAL,
        "__import__": IssueSeverity.HIGH,
        "os.system": IssueSeverity.CRITICAL,
        "os.popen": IssueSeverity.CRITICAL,
        "subprocess.call": IssueSeverity.HIGH,
        "subprocess.Popen": IssueSeverity.HIGH,
        "pickle.loads": IssueSeverity.HIGH,
        "yaml.load": IssueSeverity.MEDIUM,
    }

    HARDCODED_PATTERNS: list[tuple[str, str]] = [
        ("api_key", "API Key"),
        ("password", "密码"),
        ("secret", "密钥"),
        ("token", "Token"),
        ("database_url", "数据库连接字符串"),
    ]

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.source: str = ""
        self.tree: ast.AST | None = None
        self.issues: list[CodeIssue] = []
        self._load()

    def _load(self):
        try:
            with open(self.filepath, "r", encoding="utf-8") as f:
                self.source = f.read()
            self.tree = ast.parse(self.source)
        except SyntaxError as e:
            self.issues.append(CodeIssue(
                file=self.filepath, line=e.lineno or 0, severity=IssueSeverity.CRITICAL,
                category="syntax", title=f"语法错误: {e.msg}", description=str(e),
            ))
        except Exception as e:
            self.issues.append(CodeIssue(
                file=self.filepath, line=0, severity=IssueSeverity.HIGH,
                category="io", title=f"文件读取失败: {e}", description=str(e),
            ))

    def analyze(self) -> list[CodeIssue]:
        if self.tree is None:
            return self.issues
        self._analyze_imports()
        self._analyze_functions()
        self._analyze_security()
        self._analyze_patterns()
        self._analyze_naming()
        return self.issues

    def compute_metrics(self) -> FileMetrics:
        metrics = FileMetrics(file=self.filepath)
        if not self.source:
            return metrics
        lines = self.source.split("\n")
        metrics.total_lines = len(lines)
        metrics.code_lines = sum(1 for l in lines if l.strip() and not l.strip().startswith("#"))
        metrics.comment_lines = sum(1 for l in lines if l.strip().startswith("#"))
        metrics.blank_lines = sum(1 for l in lines if not l.strip())
        if self.tree:
            total_cpx = 0
            for node in ast.walk(self.tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    metrics.num_functions += 1
                    func_len = (node.end_lineno - node.lineno + 1) if node.end_lineno else 0
                    metrics.max_function_length = max(metrics.max_function_length, func_len)
                    cpx = self._cyclomatic_complexity(node)
                    metrics.max_complexity = max(metrics.max_complexity, cpx)
                    total_cpx += cpx
                elif isinstance(node, ast.ClassDef):
                    metrics.num_classes += 1
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    metrics.num_imports += 1
            if metrics.num_functions > 0:
                metrics.avg_complexity = total_cpx / metrics.num_functions
                metrics.maintainability_index = max(
                    0, 100 - metrics.avg_complexity * 5 - metrics.max_function_length * 0.1
                )
        return metrics

    def _analyze_imports(self):
        for node in ast.walk(self.tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                for alias in node.names:
                    if alias.name == "*":
                        self.issues.append(CodeIssue(
                            file=self.filepath, line=node.lineno, severity=IssueSeverity.LOW,
                            category="import", title="通配符 import: from X import *",
                            description="import * 会污染命名空间，IDE 和静态分析工具无法追踪来源。",
                            suggestion=f"明确列出: from {node.module} import X, Y",
                        ))

    def _analyze_functions(self):
        for node in ast.walk(self.tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            complexity = self._cyclomatic_complexity(node)
            func_len = (node.end_lineno - node.lineno + 1) if node.end_lineno else 0
            nesting = self._max_nesting_depth(node)
            code = self._get_node_source(node)

            if complexity > 15:
                self.issues.append(CodeIssue(
                    file=self.filepath, line=node.lineno, severity=IssueSeverity.HIGH,
                    category="complexity",
                    title=f"圈复杂度过高: {node.name}() (复杂度={complexity})",
                    description=f"圈复杂度 {complexity} > 15，测试和维护困难。",
                    suggestion="提取子函数，或将复杂条件链替换为策略模式。",
                    code_snippet=code[:200],
                ))
            elif complexity > 10:
                self.issues.append(CodeIssue(
                    file=self.filepath, line=node.lineno, severity=IssueSeverity.MEDIUM,
                    category="complexity",
                    title=f"圈复杂度偏高: {node.name}() (复杂度={complexity})",
                    description=f"圈复杂度 {complexity} > 10，建议简化。",
                    suggestion="考虑提取辅助函数。",
                ))

            if func_len > 50:
                self.issues.append(CodeIssue(
                    file=self.filepath, line=node.lineno, severity=IssueSeverity.MEDIUM,
                    category="length",
                    title=f"函数过长: {node.name}() ({func_len} 行)",
                    description=f"函数有 {func_len} 行，超过推荐的 50 行。",
                    suggestion="按职责拆分为更小的函数。",
                ))

            if nesting > 4:
                self.issues.append(CodeIssue(
                    file=self.filepath, line=node.lineno, severity=IssueSeverity.MEDIUM,
                    category="nesting",
                    title=f"嵌套深度过大: {node.name}() (深度={nesting})",
                    description=f"嵌套深度 {nesting} > 4，使用提前返回减少嵌套。",
                    suggestion="使用 guard clause（提前 return / continue）减少嵌套。",
                    code_snippet=code[:150],
                ))

            if len(node.args.args) > 5:
                self.issues.append(CodeIssue(
                    file=self.filepath, line=node.lineno, severity=IssueSeverity.LOW,
                    category="params",
                    title=f"参数过多: {node.name}() ({len(node.args.args)} 个)",
                    description="过多参数意味着函数职责不单一。",
                    suggestion="使用 dataclass 封装相关参数为配置对象。",
                ))

            if not ast.get_docstring(node):
                if not node.name.startswith("_") and not (
                    node.name.startswith("__") and node.name.endswith("__")
                ):
                    self.issues.append(CodeIssue(
                        file=self.filepath, line=node.lineno, severity=IssueSeverity.LOW,
                        category="doc",
                        title=f"缺少 docstring: {node.name}()",
                        description="公开函数缺少文档字符串。",
                        suggestion='添加简短 docstring: """简短描述。"""',
                    ))

    def _analyze_security(self):
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Call):
                call_name = self._get_full_call_name(node)
                for dangerous, severity in self.DANGEROUS_CALLS.items():
                    if dangerous in call_name:
                        is_ast_ctx = call_name == "eval" and self._is_ast_utility_usage()
                        actual = IssueSeverity.LOW if is_ast_ctx else severity
                        self.issues.append(CodeIssue(
                            file=self.filepath, line=node.lineno, severity=actual,
                            category="security",
                            title=f"危险函数: {call_name}()",
                            description=(
                                "如果用于 AST 解析则安全，需注释说明用途。"
                                if is_ast_ctx else
                                f"{call_name}() 可执行任意代码，存在安全风险。"
                            ),
                            suggestion="安全替代: ast.literal_eval / subprocess.run(shell=False) / yaml.safe_load()",
                            code_snippet=self._get_node_source(node)[:200],
                        ))
                        break

            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        for pattern, label in self.HARDCODED_PATTERNS:
                            if pattern in target.id.lower():
                                if isinstance(node.value, ast.Constant):
                                    val = str(node.value.value)
                                    if len(val) > 8:
                                        self.issues.append(CodeIssue(
                                            file=self.filepath, line=node.lineno,
                                            severity=IssueSeverity.CRITICAL,
                                            category="security",
                                            title=f"硬编码{label}: {target.id}",
                                            description=f"{label}硬编码在源码中，应用环境变量。",
                                            suggestion=f'{target.id} = os.environ.get("{target.id.upper()}", "")',
                                            code_snippet=f"{target.id} = '{val[:30]}...'",
                                        ))

    def _analyze_patterns(self):
        for node in ast.walk(self.tree):
            if isinstance(node, ast.ExceptHandler):
                if node.type is None:
                    self.issues.append(CodeIssue(
                        file=self.filepath, line=node.lineno, severity=IssueSeverity.HIGH,
                        category="pattern", title="裸 except 捕获",
                        description="except: 会捕获 KeyboardInterrupt 和 SystemExit。",
                        suggestion="改为明确类型: except Exception as e:",
                    ))
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for default in node.args.defaults:
                    if isinstance(default, (ast.List, ast.Dict, ast.Set)):
                        self.issues.append(CodeIssue(
                            file=self.filepath, line=node.lineno, severity=IssueSeverity.HIGH,
                            category="pattern",
                            title=f"可变默认参数: {node.name}(...)",
                            description="list/dict/set 作为默认参数在所有调用间共享。",
                            suggestion=f"def {node.name}(..., items=None):\n    items = items or []",
                            code_snippet=self._get_node_source(node)[:150],
                        ))

    def _analyze_naming(self):
        for node in ast.walk(self.tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name.isupper() and "_" not in node.name:
                    self.issues.append(CodeIssue(
                        file=self.filepath, line=node.lineno, severity=IssueSeverity.LOW,
                        category="naming",
                        title=f"函数命名: {node.name}（全大写应为常量）",
                        suggestion=f"改为 snake_case: {node.name.lower()}()",
                    ))

    @staticmethod
    def _cyclomatic_complexity(func_node: ast.AST) -> int:
        complexity = 1
        for node in ast.walk(func_node):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.AsyncFor,
                                 ast.ExceptHandler, ast.comprehension)):
                complexity += 1
            elif isinstance(node, (ast.And, ast.Or)):
                complexity += 1
        return complexity

    @staticmethod
    def _max_nesting_depth(func_node: ast.AST) -> int:
        max_depth = 0
        for node in ast.walk(func_node):
            if isinstance(node, (ast.If, ast.For, ast.While, ast.Try, ast.With,
                                 ast.AsyncFor, ast.AsyncWith, ast.ExceptHandler)):
                current = 1
                for _ in ast.iter_child_nodes(node):
                    pass
                max_depth = max(max_depth, current)
        return max_depth

    def _get_node_source(self, node: ast.AST) -> str:
        if hasattr(node, "lineno") and hasattr(node, "end_lineno") and node.end_lineno:
            lines = self.source.split("\n")
            start = max(0, node.lineno - 1)
            end = min(len(lines), node.end_lineno)
            return "\n".join(lines[start:end])
        return ""

    @staticmethod
    def _get_full_call_name(call_node: ast.Call) -> str:
        func = call_node.func
        if isinstance(func, ast.Name):
            return func.id
        if isinstance(func, ast.Attribute):
            parts = [func.attr]
            current = func.value
            while isinstance(current, ast.Attribute):
                parts.append(current.attr)
                current = current.value
            if isinstance(current, ast.Name):
                parts.append(current.id)
            return ".".join(reversed(parts))
        return "unknown"

    def _is_ast_utility_usage(self) -> bool:
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "ast":
                        return True
            elif isinstance(node, ast.ImportFrom):
                if node.module == "ast":
                    return True
        return False


class LLMCodeReviewer:
    """LLM 驱动的语义级代码审查

    AST 分析检查结构性问题。LLM 审查检查语义问题：
    边缘情况、设计模式、错误处理、代码意图。

    策略：AST 先过滤 → 只在发现严重问题时调用 LLM（节省约 80% token）。
    """

    def __init__(self, client: Anthropic, model: str = ANTHROPIC_MODEL):
        self.client = client
        self.model = model

    REVIEW_PROMPT = (
        "你是严格的代码审查专家。分析代码：\n"
        "1. 潜在 bug 或逻辑错误\n"
        "2. 错误处理不完善处\n"
        "3. 安全漏洞\n"
        "4. 性能和可读性问题\n"
        "5. 更好的设计选择\n\n"
        "回复严格 JSON：\n"
        '{"issues": [{"severity": "high|medium|low", "title": "...", '
        '"description": "...", "suggestion": "..."}], '
        '"overall": "excellent|good|fair|poor", "summary": "..."}'
    )

    def review_function(self, filepath: str, func_name: str, code: str) -> list[CodeIssue]:
        try:
            response = self.client.messages.create(
                model=self.model, max_tokens=1024, system=self.REVIEW_PROMPT,
                messages=[{
                    "role": "user",
                    "content": f"文件: {filepath}\n函数: {func_name}()\n\n```python\n{code}\n```",
                }],
            )
            result_text = response.content[0].text if response.content else "{}"
            json_str = result_text
            if "```" in json_str:
                parts = json_str.split("```")
                json_str = parts[1] if len(parts) > 1 else parts[0]
                if json_str.startswith("json"):
                    json_str = json_str[4:]
            json_str = json_str.strip()
            result = json.loads(json_str)
            issues: list[CodeIssue] = []
            for item in result.get("issues", []):
                issues.append(CodeIssue(
                    file=filepath, line=0,
                    severity=IssueSeverity(item.get("severity", "medium")),
                    category="llm_review",
                    title=item.get("title", ""),
                    description=item.get("description", ""),
                    suggestion=item.get("suggestion", ""),
                ))
            return issues
        except (json.JSONDecodeError, Exception) as e:
            return [CodeIssue(
                file=filepath, line=0, severity=IssueSeverity.INFO,
                category="llm_review",
                title=f"LLM 审查失败: {str(e)[:80]}",
                description=result_text[:200] if 'result_text' in dir() else "",
            )]

    def review_file_summary(
        self, filepath: str, metrics: FileMetrics, issues: list[CodeIssue]
    ) -> str:
        issue_summary = "\n".join(
            f"- [{i.severity.value}] {i.title}" for i in issues[:10]
        ) or "无严重问题"
        response = self.client.messages.create(
            model=self.model, max_tokens=512,
            system="你是代码审查专家。基于分析结果，给文件整体评价（100字以内）。",
            messages=[{"role": "user", "content": (
                f"文件: {filepath}\n指标: {metrics.total_lines} 行, "
                f"{metrics.num_functions} 函数, {metrics.num_classes} 类, "
                f"最高复杂度 {metrics.max_complexity}\n问题:\n{issue_summary}\n\n整体评价:"
            )}],
        )
        return response.content[0].text if response.content else ""


@dataclass
class ReviewReport:
    directory: str = ""
    files_scanned: int = 0
    files_with_issues: int = 0
    total_issues: int = 0
    issues_by_severity: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    issues_by_category: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    file_metrics: list[FileMetrics] = field(default_factory=list)
    all_issues: list[CodeIssue] = field(default_factory=list)
    summaries: dict[str, str] = field(default_factory=dict)

    def add_issues(self, issues: list[CodeIssue]):
        for issue in issues:
            self.all_issues.append(issue)
            self.issues_by_severity[issue.severity.value] += 1
            self.issues_by_category[issue.category] += 1
        self.total_issues += len(issues)


class AutonomousCodeReviewer:
    """自主代码审查 Agent

    工作流程：扫描目录 → AST 分析 → 按需 LLM 审查 → 生成报告

    三种框架实现对比：
    - LangGraph: State + 4 个 Node + 循环边
    - CrewAI: 3-4 个 Agent 角色 + Task 链
    - Claude SDK: 一个类，方法调用，最直接（本实现）
    """

    EXCLUDE_DIRS = {".venv", "venv", "__pycache__", ".git", ".tox",
                    "node_modules", ".mypy_cache", ".pytest_cache"}

    def __init__(self, api_key: str | None = None, use_llm: bool = True):
        self.use_llm = use_llm
        if use_llm and api_key:
            self.client = Anthropic(api_key=api_key)
            self.reviewer = LLMCodeReviewer(self.client)
        else:
            self.client = None
            self.reviewer = None
        self.report = ReviewReport()

    def scan_directory(self, directory: str, pattern: str = "**/*.py") -> list[str]:
        files = []
        base = Path(directory).resolve()
        for py_file in base.rglob(pattern):
            if set(py_file.parts) & self.EXCLUDE_DIRS:
                continue
            files.append(str(py_file))
        self.report.directory = str(base)
        return sorted(files)

    def review_file(self, filepath: str) -> tuple[list[CodeIssue], FileMetrics]:
        rel_path = os.path.relpath(filepath, self.report.directory)
        console.print(f"  [cyan]分析:[/cyan] {rel_path}")
        analyzer = ASTAnalyzer(filepath)
        all_issues = analyzer.analyze()
        metrics = analyzer.compute_metrics()

        if self.use_llm and self.reviewer:
            high_sev = [i for i in all_issues
                        if i.severity in (IssueSeverity.CRITICAL, IssueSeverity.HIGH)]
            if high_sev and analyzer.tree:
                console.print(f"    [yellow]{len(high_sev)} 个严重问题，LLM 深度审查...[/yellow]")
                for node in ast.walk(analyzer.tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        code = analyzer._get_node_source(node)
                        if code and len(code) > 20:
                            llm_issues = self.reviewer.review_function(filepath, node.name, code)
                            all_issues.extend(llm_issues)
            if all_issues:
                summary = self.reviewer.review_file_summary(filepath, metrics, all_issues)
                self.report.summaries[filepath] = summary
                console.print(f"    [dim]{summary[:120]}[/dim]")
        return all_issues, metrics

    def run(self, directory: str) -> ReviewReport:
        self.report = ReviewReport(directory=directory)
        files = self.scan_directory(directory)
        if not files:
            console.print("[yellow]未找到 Python 文件[/yellow]")
            return self.report
        console.print(f"\n[bold]找到 {len(files)} 个 Python 文件[/bold]\n")
        with Progress(
            SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("审查代码中...", total=len(files))
            for filepath in files:
                progress.update(task, description=f"分析: {os.path.basename(filepath)}")
                issues, metrics = self.review_file(filepath)
                self.report.file_metrics.append(metrics)
                self.report.files_scanned += 1
                if issues:
                    self.report.files_with_issues += 1
                    self.report.add_issues(issues)
                progress.advance(task)
        return self.report

    def print_report(self):
        r = self.report
        console.print(Panel(
            f"[bold]目录:[/bold] {r.directory}\n"
            f"[bold]文件:[/bold] {r.files_scanned} 扫描, {r.files_with_issues} 有问题\n"
            f"[bold]问题:[/bold] {r.total_issues} 个",
            title="代码审查总览", border_style="blue",
        ))
        if r.total_issues == 0:
            console.print("[green]未发现问题。[/green]")
            return

        sev_table = Table(title="问题严重性分布")
        sev_table.add_column("严重性", style="cyan")
        sev_table.add_column("数量", style="magenta")
        sev_table.add_column("占比")
        for sev in ["critical", "high", "medium", "low", "info"]:
            count = r.issues_by_severity.get(sev, 0)
            if count > 0:
                pct = f"{count / max(r.total_issues, 1) * 100:.1f}%"
                sev_table.add_row(sev, str(count), pct)
        console.print(sev_table)

        cat_table = Table(title="问题分类分布")
        cat_table.add_column("分类", style="cyan")
        cat_table.add_column("数量", style="magenta")
        for cat, count in sorted(r.issues_by_category.items(), key=lambda x: -x[1]):
            cat_table.add_row(cat, str(count))
        console.print(cat_table)

        serious = [i for i in r.all_issues
                   if i.severity in (IssueSeverity.CRITICAL, IssueSeverity.HIGH)]
        if serious:
            console.print(f"\n[bold red]{len(serious)} 个严重问题[/bold red]")
            for issue in serious[:15]:
                console.print(Panel(
                    f"[bold]{issue.title}[/bold]\n"
                    f"文件: {issue.file}\n行号: {issue.line}\n分类: {issue.category}\n\n"
                    f"{issue.description}\n\n[green]建议:[/green] {issue.suggestion}",
                    title=f"[{issue.severity.value.upper()}] {os.path.basename(issue.file)}:{issue.line}",
                    border_style="red" if issue.severity == IssueSeverity.CRITICAL else "yellow",
                ))

        if r.file_metrics:
            metric_table = Table(title="文件质量指标")
            metric_table.add_column("文件", style="cyan", width=25)
            metric_table.add_column("行数", style="dim")
            metric_table.add_column("函数", style="green")
            metric_table.add_column("类", style="green")
            metric_table.add_column("最高复杂度", style="yellow")
            metric_table.add_column("可维护性", style="magenta")
            for m in sorted(r.file_metrics, key=lambda x: -x.maintainability_index)[:15]:
                mi = m.maintainability_index
                mi_color = "green" if mi > 70 else "yellow" if mi > 40 else "red"
                metric_table.add_row(
                    os.path.basename(m.file)[:23], str(m.total_lines),
                    str(m.num_functions), str(m.num_classes),
                    str(m.max_complexity), f"[{mi_color}]{mi:.0f}[/{mi_color}]",
                )
            console.print(metric_table)


def run_demo():
    console.print(Panel(
        "[bold]Claude SDK: 自主代码审查 Agent[/bold]\n"
        "AST 静态分析（零 token）+ LLM 语义审查（按需启用）\n"
        "自主扫描 → 分析 → 排序 → 报告，全自动代码审查流水线",
        title="03 Autonomous Coder", border_style="blue",
    ))

    api_key = os.getenv("ANTHROPIC_API_KEY")
    use_llm = bool(api_key)
    if not use_llm:
        console.print("[yellow]未设置 ANTHROPIC_API_KEY，仅使用 AST 静态分析[/yellow]")
        console.print("[dim]设置后可启用 LLM 深度语义审查[/dim]")

    target = sys.argv[1] if len(sys.argv) > 1 else "."
    if not os.path.isdir(target):
        console.print(f"[red]目录不存在: {target}[/red]")
        return

    abs_target = os.path.abspath(target)
    console.print(f"\n[bold]目标:[/bold] {abs_target}")

    agent = AutonomousCodeReviewer(api_key=api_key, use_llm=use_llm)
    report = agent.run(abs_target)
    if report.files_scanned > 0:
        agent.print_report()

    console.print(Panel(
        "[bold]自主代码审查的架构洞察[/bold]\n\n"
        "双层分析策略：\n"
        "  AST 层: 确定性规则、零 token 开销、秒级响应\n"
        "  LLM 层: 语义理解、有 token 开销、按需启用\n"
        "  组合: AST 过滤 → 只在严重问题时调用 LLM（节省 ~80% token）\n\n"
        "三种框架实现同一功能：\n"
        "  LangGraph: State + 4 Node + 循环边，清晰但状态管理复杂\n"
        "  CrewAI: 3-4 Agent 角色 + Task 链，角色分明但过度设计\n"
        "  Claude SDK: 一个类，方法调用链，最直接但需手写\n\n"
        "选择：简单流程 → 手写 / 需分支重试 → LangGraph / 多角色 → CrewAI",
        title="设计思考", border_style="green",
    ))


if __name__ == "__main__":
    run_demo()
