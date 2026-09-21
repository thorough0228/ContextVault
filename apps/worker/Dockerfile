FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY apps/worker/requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/requirements.txt

# The worker runs the API's heavy tasks (document ingestion), so it
# needs the API package on the import path. We only copy the files
# needed at runtime — the API's routers / FastAPI app are not used
# here, but ``app.tasks`` pulls in models + services.
COPY apps/api/app /app/apps/api/app
COPY apps/api/alembic /app/apps/api/alembic
COPY apps/api/requirements.txt /app/apps/api/requirements.txt
COPY apps/worker /app/apps/worker
COPY packages/shared /app/packages/shared

ENV PYTHONPATH=/app

WORKDIR /app/apps/worker
CMD ["celery", "-A", "app.celery_app", "worker", "--loglevel=INFO"]