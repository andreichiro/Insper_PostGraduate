FROM python:3.13-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY src/ src/
COPY conf/ conf/
COPY data/01_raw/ data/01_raw/

EXPOSE 8000 8501

COPY entrypoint.sh dashboard-entrypoint.sh ./
RUN chmod +x entrypoint.sh dashboard-entrypoint.sh

ENTRYPOINT ["./entrypoint.sh"]
