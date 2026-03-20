FROM python:3.11-slim-bookworm AS builder

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --target /app/deps -r requirements.txt

FROM python:3.11-slim-bookworm

RUN useradd --create-home appuser
WORKDIR /app

COPY --from=builder /app/deps /app/deps
COPY app.py .

ENV PYTHONPATH=/app/deps
ENV PYTHONUNBUFFERED=1
EXPOSE 8000
USER appuser

CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
