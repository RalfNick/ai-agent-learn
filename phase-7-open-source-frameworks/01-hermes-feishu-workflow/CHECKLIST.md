# Deployment Checklist

Use this checklist on the VPS. Keep real secrets outside this repository.

## 1. Host Baseline

- [ ] Create a dedicated OS user, for example `hermes-agent`.
- [ ] Create a workspace root, for example `/opt/hermes-workflows`.
- [ ] Install Docker and confirm the `hermes-agent` user can run containers through the approved host policy.
- [ ] Confirm outbound network access to the selected model provider and Feishu/Lark.
- [ ] Do not expose a public Feishu webhook endpoint for the first version.

## 2. Hermes Installation

- [ ] Install Hermes as the dedicated user.
- [ ] Configure the model provider with private credentials.
- [ ] Run `hermes doctor`.
- [ ] Confirm `hermes "用一句中文说明你已经在 VPS 上运行。"` returns a response.
- [ ] Confirm `hermes --continue "回忆上一轮我让你说明什么。"` can resume the session.
- [ ] Stop here and fix Hermes/provider issues before configuring Feishu.

## 3. Safe Configuration

- [ ] Copy `config/hermes-config.example.yaml` to the private Hermes config path.
- [ ] Keep `approvals.mode: manual`.
- [ ] Keep `approvals.cron_mode: deny`.
- [ ] Keep `terminal.backend: docker`.
- [ ] Keep Docker env passthrough empty unless a specific skill requires a named variable.
- [ ] Keep memory enabled, but review important memory writes during early practice.

## 4. Feishu / Lark

- [ ] Create or scan-to-create a Feishu/Lark app through `hermes gateway setup`.
- [ ] Use WebSocket mode.
- [ ] Configure required scopes: `im:message`, `im:message:send_as_bot`, `im:resource`, `im:chat`, `im:chat:readonly`.
- [ ] Fill `FEISHU_APP_ID`, `FEISHU_APP_SECRET`, `FEISHU_DOMAIN`, `FEISHU_CONNECTION_MODE`.
- [ ] Fill `FEISHU_ALLOWED_USERS` with approved Open IDs.
- [ ] Keep `FEISHU_GROUP_POLICY=allowlist`.
- [ ] Keep `FEISHU_REQUIRE_MENTION=true`.
- [ ] Set the Feishu home chat with `/set-home` or private `FEISHU_HOME_CHANNEL`.
- [ ] Publish the app version in the Feishu/Lark developer console.

## 5. Gateway Service

- [ ] First run `hermes gateway` in foreground and confirm Feishu DM works.
- [ ] Run `hermes gateway install`.
- [ ] On VPS/headless hosts, run `sudo loginctl enable-linger hermes-agent` if using the user service.
- [ ] Run `hermes gateway start`.
- [ ] Run `hermes gateway status`.
- [ ] Compare the generated service with `deploy/hermes-gateway.service.example` only if you need to audit or customize service behavior.

## 6. Workflow Verification

- [ ] In Feishu DM, run `/status`.
- [ ] Ask for a read-only project check.
- [ ] Create the daily learning brief cron job.
- [ ] Confirm the job appears in `hermes cron list`.
- [ ] Run `hermes cron status`.
- [ ] Check `~/.hermes/cron/output/` after a manual or scheduled run.
- [ ] Confirm gateway is running when testing cron delivery.
- [ ] Trigger a controlled dangerous-command approval test.
- [ ] Restart the gateway and confirm Feishu responds again.

## 7. Evidence

- [ ] Record commands and key output in `PRACTICE_LOG.md`.
- [ ] Record any failure and the fix.
- [ ] Keep screenshots or copied Feishu message excerpts outside the repo if they contain private IDs.
