# Practice Log

Use this file to record real deployment evidence. Do not paste secrets, full tokens, private chat IDs, or real App Secrets.

## Run 1: Initial VPS Setup

Date:

Environment:

```text
OS:
Hermes version:
Python version:
Docker version:
Connection mode:
Model provider:
```

Commands:

```bash
hermes doctor
hermes "用一句中文说明你已经在 VPS 上运行。"
hermes --continue "回忆上一轮我让你说明什么。"
hermes gateway status
hermes cron list
hermes cron status
find ~/.hermes/cron/output -maxdepth 3 -type f | tail
```

Key output:

```text
Record concise, redacted output here.
```

Feishu observation:

```text
Record whether /status, /set-home, DM reply, group mention gate, and cron delivery worked. Redact user IDs.
```

Result:

```text
Not run yet.
```

Limitations or failures:

```text
Not run yet.
```

Next action:

```text
Run the acceptance checks after real credentials are configured on the VPS.
```
