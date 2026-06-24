# Phase6 Web UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a usable Next.js UI for the Phase6 capstone that lets a reader ask questions and inspect answer, sources, trace, and review status.

**Architecture:** Create `phase-6-capstone/04-web-ui` as an independent Next.js app. The UI calls `/api/v1/answer` through a configurable API base URL and falls back to a local demo response so the frontend can be reviewed without running Python services.

**Tech Stack:** Next.js, React, TypeScript, lucide-react, Node built-in test runner for static contract checks.

## Global Constraints

- First viewport must be the working chat experience, not a landing page.
- Show sources, trace, review status, and mode as first-class UI.
- Keep the UI dense and operational, not marketing-style.
- Use stable layout dimensions and avoid nested cards.
- Do not require the backend to run for basic UI demo mode.

---

### Task 1: UI Contract Tests

**Files:**
- Create: `phase-6-capstone/04-web-ui/tests/ui-contract.test.mjs`

- [x] **Step 1: Write failing tests**

Tests assert:
- `app/page.tsx` exists and references sources, trace, review status.
- `lib/demo-response.mjs` exports `demoAnswerResponse` with answer, sources, trace.
- `lib/format.mjs` maps `evidence_supported` and `abstained` to display labels.

- [x] **Step 2: Verify RED**

Run:

```bash
cd phase-6-capstone/04-web-ui && npm test
```

Expected: fail because app files do not exist.

### Task 2: Next.js App

**Files:**
- Create: `phase-6-capstone/04-web-ui/package.json`
- Create: `phase-6-capstone/04-web-ui/next.config.mjs`
- Create: `phase-6-capstone/04-web-ui/tsconfig.json`
- Create: `phase-6-capstone/04-web-ui/app/layout.tsx`
- Create: `phase-6-capstone/04-web-ui/app/page.tsx`
- Create: `phase-6-capstone/04-web-ui/app/globals.css`
- Create: `phase-6-capstone/04-web-ui/lib/demo-response.mjs`
- Create: `phase-6-capstone/04-web-ui/lib/format.mjs`

- [x] **Step 1: Implement UI**

Build:
- Chat input and submit button.
- Answer panel with mode and review status.
- Sources panel with title, score, snippet, path.
- Trace panel with step, detail, latency.
- Demo fallback when API call fails.

- [x] **Step 2: Verify tests**

Run `npm test`.

### Task 3: Documentation And Review

**Files:**
- Create: `phase-6-capstone/04-web-ui/README.md`
- Create: `docs/phase-6/04-web-ui.md`
- Modify: `docs/phase-6/README.md`
- Modify: `phase-6-capstone/README.md`

- [x] **Step 1: Document usage**

Include:
- install
- dev server
- API base URL
- demo fallback
- current limitations

- [x] **Step 2: Round review**

Review:
- UI exposes sources/trace/review.
- frontend does not hide backend uncertainty.
- next step into release eval / docker integration.

### Task 4: Verification

- [x] **Step 1: Install dependencies**

```bash
cd phase-6-capstone/04-web-ui && npm install
```

- [x] **Step 2: Run tests**

```bash
cd phase-6-capstone/04-web-ui && npm test
```

- [x] **Step 3: Build**

```bash
cd phase-6-capstone/04-web-ui && npm run build
```

- [x] **Step 4: Start dev server smoke**

```bash
cd phase-6-capstone/04-web-ui && npm run dev -- --port 3020
```
