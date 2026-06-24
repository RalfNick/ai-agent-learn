# Phase5 Docker Deploy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Containerize the Phase5 FastAPI wrapper so the Phase4 integrated runtime can be started with Docker Compose and checked through `/health`.

**Architecture:** Keep `01-fastapi-backend` as the application source and add `02-docker-deploy` as a deployment wrapper. The Docker image copies the FastAPI app plus the Phase4 runtime dependencies into `/app`, then starts `uvicorn app.main:app` from the backend workdir. Compose builds from the repository root, publishes port `8000`, persists `.memory` in a named volume, and repeats the same `/health` check at the orchestration layer.

**Tech Stack:** Docker, Docker Compose, Python 3.12 slim image, FastAPI, Uvicorn, standard-library `unittest`.

---

### Task 1: File Contract Tests

**Files:**
- Create: `phase-5-production/02-docker-deploy/tests/test_docker_deploy_files.py`

- [x] **Step 1: Write the failing test**

```python
class DockerDeployFilesTest(unittest.TestCase):
    def test_dockerfile_packages_fastapi_and_phase4_runtime(self) -> None:
        dockerfile = read_deploy_file("Dockerfile")
        self.assertIn("FROM python:3.12-slim", dockerfile)
        self.assertIn("phase-4-advanced/05-agent-runtime-integration", dockerfile)
        self.assertIn("http://127.0.0.1:8000/health", dockerfile)

    def test_compose_exposes_service_and_healthcheck(self) -> None:
        compose = read_deploy_file("docker-compose.yml")
        self.assertIn("phase5-agent-api:", compose)
        self.assertIn('"8000:8000"', compose)
        self.assertIn("healthcheck:", compose)
```

- [x] **Step 2: Run test to verify it fails**

Run: `PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s phase-5-production/02-docker-deploy/tests`

Expected: FAIL because `Dockerfile`, `docker-compose.yml`, and `README.md` do not exist yet.

### Task 2: Docker Deployment Files

**Files:**
- Create: `phase-5-production/02-docker-deploy/Dockerfile`
- Create: `phase-5-production/02-docker-deploy/docker-compose.yml`
- Create: `phase-5-production/02-docker-deploy/README.md`
- Create: `.dockerignore`

- [ ] **Step 1: Add Dockerfile**

Use `python:3.12-slim`, install backend requirements, copy the FastAPI app, `docs/`, Phase2/Phase3 benchmark outputs, and Phase4 runtime folders, expose `8000`, and define `HEALTHCHECK CMD curl -fsS http://127.0.0.1:8000/health || exit 1`.

- [ ] **Step 2: Add compose file**

Use root build context `../..`, Dockerfile `phase-5-production/02-docker-deploy/Dockerfile`, publish `"8000:8000"`, persist `.memory` through `phase5_agent_memory`, and repeat the `/health` healthcheck.

- [ ] **Step 3: Add README**

Document `docker compose up --build`, `/health`, `/api/v1/agent/answer`, logs, and teardown.

### Task 3: Verification

**Files:**
- Modify: `docs/phase-5/README.md`
- Create: `docs/phase-5/02-docker-deploy.md`

- [ ] **Step 1: Run file contract tests**

Run: `PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s phase-5-production/02-docker-deploy/tests`

Expected: PASS with 4 tests.

- [ ] **Step 2: Run backend regression tests**

Run: `PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s phase-5-production/01-fastapi-backend/tests`

Expected: PASS with 3 tests.

- [ ] **Step 3: Run Docker verification when daemon is available**

Run:

```bash
docker compose -f phase-5-production/02-docker-deploy/docker-compose.yml build
docker compose -f phase-5-production/02-docker-deploy/docker-compose.yml up -d
curl http://127.0.0.1:8000/health
docker compose -f phase-5-production/02-docker-deploy/docker-compose.yml down
```

Expected: build succeeds, `/health` returns JSON with `status: ok`, and teardown succeeds.
