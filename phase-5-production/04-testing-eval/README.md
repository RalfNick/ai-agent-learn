# Phase5 04 Testing Eval

这一节给 FastAPI Agent 服务增加 API 回归测试和最小评估入口。它不引入外部评测平台，先用固定样本 replay 当前 Agent runtime，检查结果是否仍满足工程验收条件。

## 已实现接口

- `GET /api/v1/evaluations/cases`：查看内置评估样本。
- `POST /api/v1/evaluations/run`：运行全部或指定样本。
- `GET /api/v1/observability/traces/eval-{case_id}`：查看某条 eval case 的 trace detail。

## 启动服务

```bash
cd phase-5-production/01-fastapi-backend
python3 -m uvicorn app.main:app --reload --port 8000
```

## 查看样本

```bash
curl http://127.0.0.1:8000/api/v1/evaluations/cases
```

## 运行全部评估

```bash
curl -X POST http://127.0.0.1:8000/api/v1/evaluations/run \
  -H "Content-Type: application/json" \
  -d '{"session_prefix":"manual-eval"}'
```

## 运行指定样本

```bash
curl -X POST http://127.0.0.1:8000/api/v1/evaluations/run \
  -H "Content-Type: application/json" \
  -d '{"case_ids":["phase4-memory-evidence"],"session_prefix":"manual-eval"}'
```

## 验证

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s phase-5-production/01-fastapi-backend/tests
```

当前 smoke 结果：

```text
total_cases=3
passed_cases=3
failed_cases=0
pass_rate=1.0
```

当前评估仍是确定性规则，不替代真实线上评测。它的目标是先把 API 回归和可复盘 replay 的接口立住。
