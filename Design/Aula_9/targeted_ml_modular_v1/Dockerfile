FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml README.md ./
COPY targeted_ml ./targeted_ml
COPY specs ./specs
COPY raw_to_modelled_build.py modelled_to_ml_build.py ml_to_html_build.py ./

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e .

ENTRYPOINT ["targeted-ml"]
