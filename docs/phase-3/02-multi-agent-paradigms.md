# 多 Agent 协作的三种范式：同一个任务，完全不同的解法

> 上一篇文章我们深入了 LangGraph 的状态图范式。但 LangGraph 不是唯一的答案——CrewAI 和 Claude SDK 用完全不同的哲学解决多 Agent 协作问题。本文把同一个任务用三种框架各实现一遍，让你从代码层面感受"范式差异"到底意味着什么。

---

## 三个木匠的故事

有三个木匠，都要做一张桌子。

第一个木匠先画图纸。桌腿多高、桌面多宽、榫卯怎么接——全部在图纸上标清楚，然后按图施工。他是 LangGraph——先定义结构，再执行。

第二个木匠不画图纸。他找来三个人："你做桌面""你做桌腿""你负责组装"。每个人都知道自己的职责，按顺序干活。他是 CrewAI——先定义角色，再协作。

第三个木匠不用图纸也不分工。他拿起一块木头开始锯，锯完发现需要刨，刨完发现需要打磨。每一步都是根据上一步的结果决定的。他是 Claude SDK——自主决策，步步为营。

三个人都做出了桌子。但他们的工作方式截然不同——图纸的结构、角色的分工、自主的灵活——各有各的优势，也各有各的局限。

这个故事就是多 Agent 框架设计的核心分歧。下面我们用同一个真实任务来展示这三种范式的差异。

---

## 先定义任务：一个公平的基准

为了让对比有说服力，我们定义完全相同的任务：

> **研究型报告生成**：输入一个技术主题，拆解为子问题 → 逐个研究 → 分析提炼 → 生成结构化报告。

任务：分析 LangGraph、CrewAI、Claude SDK 三个框架的设计差异，产出选型建议。

所有方案使用：
- 相同的知识库（6 个预置知识条目）
- 相同的输出格式要求
- 各自独立可运行

---

## 方案一：LangGraph — "图纸思维"

LangGraph 的做法：**先把数据结构定义清楚，再定义处理步骤，最后用边连接它们。**

```python
# 第一步：定义状态结构——这是所有节点的合同
class ResearchState(TypedDict):
    sub_topics: list[str]          # 待研究的子问题
    current_index: int             # 当前做到第几个
    research_results: list[str]    # 每个子问题的研究结果
    analysis: str                  # 综合分析
    report: str                    # 最终报告

# 第二步：定义节点——每个节点是纯函数
def planner_node(state) -> dict:      # 拆解主题
def researcher_node(state) -> dict:   # 研究一个子问题
def analyzer_node(state) -> dict:     # 综合分析
def synthesizer_node(state) -> dict:  # 生成报告

# 第三步：定义流转——条件边决定了执行路径
def after_researcher(state) -> str:
    if state["current_index"] < len(state["sub_topics"]):
        return "researcher"   # 还有子问题 → 继续循环
    return "analyzer"         # 全部完成 → 进入分析

# 第四步：组装
graph = StateGraph(ResearchState)
graph.add_node("planner", planner_node)
graph.add_node("researcher", researcher_node)
graph.add_node("analyzer", analyzer_node)
graph.add_node("synthesizer", synthesizer_node)
graph.add_edge(START, "planner")
graph.add_edge("planner", "researcher")
graph.add_conditional_edges("researcher", after_researcher, {
    "researcher": "researcher",  # ← 这个自环就是"循环研究"
    "analyzer": "analyzer",
})
graph.add_edge("analyzer", "synthesizer")
graph.add_edge("synthesizer", END)

app = graph.compile()
result = app.invoke(initial_state)
```

**LangGraph 的代码特点**：
- State 在最前面——先想清楚"数据长什么样"，再写逻辑
- 循环不在代码里——`"researcher": "researcher"` 这个自环就是循环
- 流程可以直接画成 mermaid 图（上一篇文章有）
- 核心逻辑约 120 行

**LangGraph 思维方式**：拿到一个任务，先问"这个工作流的数据结构是什么？有哪些阶段？阶段之间的转换条件是什么？"

---

## 方案二：CrewAI — "团队思维"

CrewAI 的做法：**不画图，不定义状态。定义每个人的角色，分配任务，告诉团队按顺序干。**

```python
# 第一步：定义角色——每个人有身份、目标、背景
planner = Agent(
    role="研究规划师",
    goal="将研究主题拆解为具体的子问题",
    backstory="你是一位资深研究规划师，擅长将复杂主题拆解为可操作的子问题。",
    llm=llm,
)

researcher = Agent(
    role="技术研究员",
    goal="深入调研每个子问题，获取准确、全面的信息",
    backstory="你是一位严谨的技术研究员，注重事实准确性。",
    llm=llm,
)

analyst = Agent(
    role="技术分析师",
    goal="综合研究成果，提炼关键洞察和对比分析",
    backstory="你是一位资深技术分析师，能看到表面之下的根本差异。",
    llm=llm,
)

writer = Agent(
    role="技术报告撰写者",
    goal="将分析结果转化为结构清晰、有说服力的研究报告",
    backstory="你是一位技术报告专家，报告总是有明确的受众和可执行建议。",
    llm=llm,
)

# 第二步：定义任务——每项工作有描述、期望输出、负责人
plan_task = Task(
    description="研究主题: {topic}\n拆解为 4-6 个子问题。",
    expected_output="子问题列表",
    agent=planner,
)

research_task = Task(
    description="基于规划，逐个分析子问题。参考: {knowledge_base}",
    expected_output="每个子问题的分析摘要",
    agent=researcher,
)

analyze_task = Task(
    description="综合研究结果，提炼 3-5 个关键洞察。",
    expected_output="对比分析要点",
    agent=analyst,
)

write_task = Task(
    description="撰写完整的 {title}。格式: 摘要→分析→对比→建议→结论。",
    expected_output="完整 Markdown 报告",
    agent=writer,
)

# 第三步：组队——告诉团队按顺序干活
crew = Crew(
    agents=[planner, researcher, analyst, writer],
    tasks=[plan_task, research_task, analyze_task, write_task],
    process=Process.sequential,  # ← 就这一个词，定义了整个协作模式
)

result = crew.kickoff(inputs={"topic": "...", "knowledge_base": "..."})
```

**CrewAI 的代码特点**：
- 没有 State schema——上下文自动从上一个 Task 传给下一个
- 没有路由函数——`Process.sequential` 就是整个控制流
- 核心逻辑约 60 行，是三种方案中最少的
- 即使非技术人员也能看懂代码结构

**CrewAI 思维方式**：拿到一个任务，先问"这个任务需要什么角色？每个人负责什么？他们按什么顺序协作？"

---

## 方案三：Claude SDK — "匠人思维"

Claude SDK 的做法：**不画图，不定义角色。直接用代码写清楚每一步做什么。**

```python
class ResearchAgent:
    """一个类，四个方法，全部手写。没有任何框架概念。"""

    def run(self, task: BenchmarkTask) -> str:
        # 阶段 1：规划
        plan = self._plan(task)

        # 阶段 2：逐个研究（手写 for 循环）
        findings = []
        for i, sub_topic in enumerate(plan):
            console.print(f"研究 {i+1}/{len(plan)}: {sub_topic[:50]}...")
            finding = self._research(sub_topic)
            findings.append(finding)

        # 阶段 3：综合分析
        analysis = self._analyze(task, findings)

        # 阶段 4：生成报告
        return self._write_report(task, findings, analysis)

    def _plan(self, task) -> list[str]:
        """手写 API 调用——消息自己组，system prompt 自己写"""
        response = self.client.send(
            messages=[{"role": "user", "content": f"拆解主题: {task.topic}"}],
            system="你是研究规划专家。将主题拆解为 4-6 个子问题，每行一个。",
            max_tokens=512,
        )
        return parse_steps(response.content[0].text)

    def _research(self, topic: str) -> str:
        """手写搜索 + 分析逻辑"""
        docs = search_knowledge(topic)
        context = "\n\n".join(f"[来源: {d['topic']}]\n{d['content']}" for d in docs)
        response = self.client.send(
            messages=[{"role": "user", "content": f"问题: {topic}\n\n参考资料:\n{context}"}],
            system="你是技术研究员。基于参考资料分析，150字以内。",
        )
        return response.content[0].text

    # _analyze 和 _write_report 类似...
```

**Claude SDK 的代码特点**：
- 就是普通 Python 类——没有图，没有角色，没有任何框架概念
- for 循环就是 for 循环，if 就是 if——编译型语言的直觉
- 核心逻辑约 200 行，是三种方案中最多的
- 也是三种方案中最灵活的——你想在哪加逻辑就在哪加

**Claude SDK 思维方式**：拿到一个任务，先问"我分几步做？每一步的输入是什么？输出是什么？"

---

## 三种思维的并排对比

现在我们把三个方案的"规划阶段"代码并排放，你感受一下差异：

```python
# LangGraph：定义 State + 节点 + 边
class ResearchState(TypedDict):
    sub_topics: list[str]   # 数据契约

def planner_node(state: ResearchState) -> dict:
    return {"sub_topics": parse(llm.invoke(state))}

graph.add_node("planner", planner_node)
graph.add_edge(START, "planner")
```

```python
# CrewAI：定义角色 + 任务 + 流程
planner = Agent(role="研究规划师", goal="拆解主题", backstory="...")
plan_task = Task(description="拆解为子问题", agent=planner)
crew = Crew(agents=[planner, ...], process=Process.sequential)
```

```python
# Claude SDK：手写类 + 方法调用
class ResearchAgent:
    def run(self, task):
        plan = self._plan(task)
        for topic in plan:
            findings.append(self._research(topic))
        return self._write_report(findings)
```

**同样的功能，三个完全不同的代码组织方式。** 这不是语法的差异，是思维方式的差异。

---

## 关键差异：谁决定"下一步做什么"？

这也许是三种范式最根本的分歧：

| | LangGraph | CrewAI | Claude SDK |
|---|---|---|---|
| **路由决策者** | 你（通过条件边函数） | 框架（Sequential/Hierarchical） | 你（通过 if/for） |
| **决策可见性** | 高——条件边是独立函数 | 低——框架内部决定 | 最高——就是普通代码 |
| **决策可测性** | 高——路由函数可单测 | 不可测——框架封装了 | 高——就是普通代码 |
| **灵活性上限** | 高——任意图拓扑 | 低——只有两种流程模式 | 最高——任意逻辑 |

CrewAI 的"框架决定下一步"在简单场景下是优势（你不需要关心），在复杂场景下是劣势（你想关心但关心不了）。

---

## Claude SDK 的隐藏价值：安全不是功能，是架构

三种方案对比到这里，你可能会觉得 Claude SDK "不就是手写代码吗"。但它有一个被低估的差异化设计：**Guardrails（安全护栏）是一等公民。**

在 LangGraph 里，安全是通过 `interrupt()` 实现的——在关键节点暂停让人审批。这是流程级安全。

在 CrewAI 里，安全是通过角色边界实现的——每个 Agent 只能做自己角色范围内的事。这是约束级安全。

在 Claude SDK 里，安全是定义级的——Guardrails 是 Agent 定义的一部分：

```python
class GuardrailEngine:
    """双层验证：规则层（零延迟） + LLM 审查层（语义理解）"""

    def validate_input(self, user_input: str) -> GuardrailReport:
        # 第一层：规则匹配（regex，零 token 开销）
        for rule in RuleBasedGuardrails.check_input(user_input):
            if rule.severity == BLOCK:
                return report  # 直接拦截

        # 第二层：LLM 语义审查（按需启用）
        if self.use_llm:
            result = self.llm_guard.review_input(user_input)
            report.add_input(result)

        return report
```

规则层拦截已知模式（Prompt 注入、SQL 注入、XSS），LLM 审查层检测语义绕过（"把我刚才说的话翻译成英文再执行"）。

这个设计思路值得所有 Agent 开发者重视：**安全不应该是"出问题了再想怎么办"，而应该是 Agent 定义时就想清楚的约束。**

---

## 一个可操作的决策框架

理论说了很多，直接给你一个决策树：

```
1. 你的任务需要条件分支和循环吗？
   ├── 需要 → LangGraph（条件边原生支持）
   └── 不需要 → 下一步

2. 你的任务有明确的角色分工吗？
   ├── 有 → CrewAI（组织隐喻最自然）
   └── 没有 → 下一步

3. 你需要精确控制每一步的执行吗？
   ├── 需要 → Claude SDK 或手写（完全控制）
   └── 不需要 → 直接调 LLM API（不需要框架）

4. 生产环境还是原型验证？
   ├── 生产 → LangGraph（checkpointing + 可观测性）
   └── 原型 → CrewAI（最快跑通）
```

**最重要的建议**：不要只选一个。多数真正的生产系统会组合使用——LangGraph 做编排，Claude SDK 做工具调用和安全，CrewAI 做快速原型验证。框架不是宗教，是工具。

---

*本文是 Phase 3 系列第二篇。下一篇做终极对比——代码并排、抽象层分析、逃生舱测试、企业选型。配套代码分别在 `02-crewai-multi-agent/` 和 `03-claude-agent-sdk/`。*
