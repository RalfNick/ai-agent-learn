from __future__ import annotations

import unittest
from pathlib import Path


DEPLOY_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = DEPLOY_ROOT.parents[1]


def read_deploy_file(name: str) -> str:
    return (DEPLOY_ROOT / name).read_text(encoding="utf-8")


class DockerDeployFilesTest(unittest.TestCase):
    def test_dockerfile_packages_fastapi_and_phase4_runtime(self) -> None:
        dockerfile = read_deploy_file("Dockerfile")

        self.assertIn("FROM python:3.12-slim", dockerfile)
        self.assertIn("WORKDIR /app/phase-5-production/01-fastapi-backend", dockerfile)
        self.assertIn("phase-5-production/01-fastapi-backend/requirements.txt", dockerfile)
        self.assertIn("phase-5-production/01-fastapi-backend/app", dockerfile)
        self.assertIn("COPY docs /app/docs", dockerfile)
        self.assertIn("phase-2-rag/05-rag-benchmark/outputs", dockerfile)
        self.assertIn("phase-3-frameworks/02-agentic-rag-langgraph/outputs", dockerfile)
        self.assertIn("phase-4-advanced/03-memory-system", dockerfile)
        self.assertIn("phase-4-advanced/04-multi-agent-patterns", dockerfile)
        self.assertIn("phase-4-advanced/05-agent-runtime-integration", dockerfile)
        self.assertIn("HEALTHCHECK", dockerfile)
        self.assertIn("http://127.0.0.1:8000/health", dockerfile)
        self.assertIn('CMD ["uvicorn", "app.main:app"', dockerfile)

    def test_compose_exposes_service_and_healthcheck(self) -> None:
        compose = read_deploy_file("docker-compose.yml")

        self.assertIn("phase5-agent-api:", compose)
        self.assertIn("context: ../..", compose)
        self.assertIn("dockerfile: phase-5-production/02-docker-deploy/Dockerfile", compose)
        self.assertIn('"8000:8000"', compose)
        self.assertIn("healthcheck:", compose)
        self.assertIn("http://127.0.0.1:8000/health", compose)
        self.assertIn("phase5_agent_memory:/app/phase-5-production/01-fastapi-backend/.memory", compose)

    def test_deploy_readme_documents_build_run_and_healthcheck(self) -> None:
        readme = read_deploy_file("README.md")

        self.assertIn("docker compose up --build", readme)
        self.assertIn("curl http://127.0.0.1:8000/health", readme)
        self.assertIn("docker compose down", readme)

    def test_root_dockerignore_excludes_local_and_sensitive_files(self) -> None:
        dockerignore = (REPO_ROOT / ".dockerignore").read_text(encoding="utf-8")

        self.assertIn(".env", dockerignore)
        self.assertIn("node_modules/", dockerignore)
        self.assertIn("__pycache__/", dockerignore)
        self.assertIn(".memory/", dockerignore)


if __name__ == "__main__":
    unittest.main()
