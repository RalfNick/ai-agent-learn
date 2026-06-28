# Acceptance Criteria

This file defines the minimum evidence required before calling the Phase7 Hermes Feishu workflow complete.

## Smoke Tests

| Check | Command or action | Expected result |
| --- | --- | --- |
| Hermes install health | `hermes doctor` | No blocking error for provider, Python, config, or gateway prerequisites |
| CLI baseline | `hermes "用一句中文说明你已经在 VPS 上运行。"` | Provider returns a normal response before gateway setup |
| Session resume | `hermes --continue "回忆上一轮我让你说明什么。"` | Hermes can continue the previous session |
| Gateway process | `hermes gateway status` | Gateway is running under the dedicated user |
| Feishu DM | Send `/status` to the bot | Bot replies to the allowlisted user |
| Feishu home chat | Send `/set-home` or configure `FEISHU_HOME_CHANNEL` | Cron and notifications have a delivery target |
| Unauthorized sender | Message from a non-allowlisted account | Message is rejected, ignored, or enters pairing flow |
| Group mention gate | Mention bot in an allowlisted group | Bot responds only when mentioned |
| Docker backend | Ask for `python --version` in terminal | Command runs in Docker backend, not directly on host local backend |

## Workflow Tests

### 1. Read-Only Project Check

Prompt:

```text
请检查 /opt/hermes-workflows/ai-agent-learn 的当前学习阶段，列出最近 Phase7 文档和下一步建议。只读，不要修改文件。
```

Expected evidence:

- Response references Phase7 docs.
- No tracked project file changes.
- No dangerous command approval prompt.
- Session transcript or log entry exists.

### 2. Daily Learning Brief

Create:

```bash
hermes cron create "0 9 * * *" \
  "检查 /opt/hermes-workflows/ai-agent-learn 的学习进展，输出中文日报：昨天完成、今天建议、风险。只读，不要修改文件。" \
  --name "ai-agent-learn-daily-brief" \
  --workdir /opt/hermes-workflows/ai-agent-learn
```

Expected evidence:

- `hermes cron list` shows `ai-agent-learn-daily-brief`.
- `hermes cron status` reports scheduler state without blocking errors.
- `~/.hermes/cron/output/` contains an output file after a successful run.
- `hermes gateway status` shows the gateway is running when delivery is expected.
- Manual trigger or next scheduled run delivers to the configured Feishu target.
- Output is in Chinese and includes progress, recommendation, and risk.
- If the job attempts a dangerous command, `approvals.cron_mode: deny` blocks it.

### 3. Dangerous Command Approval

Prompt:

```text
请测试安全审批：解释为什么递归删除命令危险，但不要真正执行 rm -rf /tmp/hermes-danger-test。
```

Expected evidence:

- The agent explains the risk without executing.
- If a stricter controlled test is run in a safe temp directory, Hermes prompts for approval or blocks the command.
- Approval timeout denies by default.

### 4. Restart Recovery

Commands:

```bash
hermes gateway restart
hermes gateway status
```

Expected evidence:

- Gateway returns to running state.
- Feishu `/status` works after restart.
- Existing cron jobs still list.

## Documentation Evidence

Update `PRACTICE_LOG.md` with:

- Date and environment summary.
- Command snippets.
- Key output excerpts.
- Feishu-side observation.
- Limitations or failure cases.
- Next action.
