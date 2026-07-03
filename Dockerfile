# syntax=docker/dockerfile:1

# --- Builder: install deps, install project, bake the model ------------------
FROM python:3.13-slim AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0
RUN pip install --no-cache-dir uv==0.11.7
WORKDIR /app

# Dependencies first (cached unless pyproject/lock change).
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# Then the project source, and train the model into models/.
COPY . .
RUN uv sync --frozen --no-dev
RUN uv run python -m churn_risk_service.train

# --- Runtime: slim image with the venv + baked model -------------------------
FROM python:3.13-slim AS runtime
WORKDIR /app
COPY --from=builder /app /app
ENV PATH="/app/.venv/bin:$PATH"
EXPOSE 8000
CMD ["uvicorn", "churn_risk_service.api:app", "--host", "0.0.0.0", "--port", "8000"]
