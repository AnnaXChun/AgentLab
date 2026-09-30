FROM python:3.12-slim
WORKDIR /app
RUN pip install --no-cache-dir uv==0.12.21
COPY pyproject.toml uv.lock ./
COPY packages packages
COPY adapters adapters
COPY domain_packs domain_packs
COPY src src
RUN uv sync --frozen --no-dev --no-editable --python /usr/local/bin/python
COPY migrations migrations
COPY alembic.ini ./
COPY examples examples
ENV PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1
EXPOSE 8000
CMD ["sh", "-c", "alembic upgrade head && uvicorn packages.sdk.api:app --host 0.0.0.0 --port 8000 --workers 1"]
