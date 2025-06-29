# syntax=docker/dockerfile:1

FROM python:3.10-alpine as builder

# Install build dependencies for Alpine
RUN apk add --no-cache \
    gcc \
    musl-dev \
    linux-headers \
    g++ \
    make \
    libffi-dev

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Final stage
FROM python:3.10-alpine

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    WEALTHAPI_PRETRAINED=1 \
    PORT=8000 \
    PATH="/opt/venv/bin:$PATH"

# Install runtime dependencies
RUN apk add --no-cache \
    libstdc++ \
    libgomp \
    curl

# Copy virtual environment
COPY --from=builder /opt/venv /opt/venv

# Create user
RUN adduser -D -s /bin/sh app

WORKDIR /app

# Copy app
COPY app/ ./app/
COPY *.py ./

RUN chown -R app:app /app

USER app

HEALTHCHECK --interval=30s --timeout=30s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:$PORT/health || exit 1

EXPOSE $PORT

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port $PORT --workers 1"]
