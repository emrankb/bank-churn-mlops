FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CHURN_MODEL_PATH=/app/artifacts/churn-model/model.joblib

WORKDIR /app
COPY requirements.txt requirements-api.txt ./
RUN python -m pip install --no-cache-dir -r requirements-api.txt
COPY bank_churn/ ./bank_churn/
COPY scripts/__init__.py scripts/serve_api.py ./scripts/
RUN useradd --create-home --uid 10001 appuser
USER appuser

EXPOSE 8000
CMD ["python", "-m", "scripts.serve_api", "--host", "0.0.0.0", "--port", "8000"]
