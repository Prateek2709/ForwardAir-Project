# FastAPI Dockerfile
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

# Run FastAPI app (Change main:app if your module/object is different)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
