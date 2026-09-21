# Infrastructure

Docker Compose and helper scripts for the local ContextVault stack.

## Services

| Service     | Port (host) | Purpose                                |
|-------------|-------------|----------------------------------------|
| `postgres`  | 5432        | Primary relational store               |
| `redis`     | 6379        | Cache + Celery broker / result backend |
| `minio`     | 9000 / 9001 | S3-compatible object storage           |
| `minio-init`| -           | One-shot bucket bootstrap              |
| `api`       | 8000        | FastAPI backend                        |
| `web`       | 3000        | Next.js frontend                       |
| `worker`    | -           | Celery worker                          |

## Bring everything up

From the project root (`contextvault/`):

```bash
cp .env.example .env                  # fill in real values
docker compose -f infrastructure/docker-compose.yml --env-file .env up --build
```

Convenience wrapper (from repo root):

```bash
docker compose -f infrastructure/docker-compose.yml up --build
```

## Health probes

```bash
curl -s http://localhost:8000/api/v1/health
open http://localhost:3000
open http://localhost:9001              # MinIO console
```