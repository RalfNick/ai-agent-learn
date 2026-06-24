# Phase6 04 Web UI

这是 Phase6 capstone 的前端界面：一个可直接使用的企业知识库 Agent 控制台。

它不是 landing page，打开后就是工作台：

- 左侧：问题输入、回答、mode、API 状态。
- 右侧：Sources / Trace 切换视图。
- 顶部：review status。

## 安装

```bash
cd phase-6-capstone/04-web-ui
npm install
```

## 运行测试

```bash
npm test
```

## 构建

```bash
npm run build
```

## 启动

```bash
npm run dev -- --port 3020
```

打开：

```text
http://127.0.0.1:3020
```

## 连接后端

默认 API 地址：

```text
http://127.0.0.1:8010
```

可以通过环境变量覆盖：

```bash
NEXT_PUBLIC_API_BASE_URL=http://127.0.0.1:8010 npm run dev -- --port 3020
```

如果后端不可用，UI 会展示内置 demo response，方便独立查看布局和交互。

## 当前边界

- 还没有登录和权限。
- 还没有流式输出。
- 还没有 eval dashboard。
- 当前 trace 是后端返回的结构化步骤，不做复杂可视化。

下一步进入 release/eval 集成，把后端、runtime、UI 放进同一个可复现启动方式里。
