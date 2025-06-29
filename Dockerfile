# syntax=docker/dockerfile:1

FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    WEALTHAPI_PRETRAINED=1 \
    PORT=8000

RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libfontconfig1 \
    libice6 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Don't run as root
RUN useradd --create-home --shell /bin/bash app

WORKDIR /app

# Install Python packages first (better Docker caching)
COPY requirements.txt ./

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && pip cache purge

# Copy the actual code
COPY app/ ./app/
COPY *.py ./

# Make sure the app user owns everything
RUN chown -R app:app /app

USER app

# Basic health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:$PORT/health || exit 1

EXPOSE $PORT

# Start the server
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port $PORT --workers 1"] 