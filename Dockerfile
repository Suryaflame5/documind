# DocuMind: Enterprise Document Intelligence System
FROM python:3.10-slim

# Set working directory & environment variables
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# Install system utilities & graphical libs required by PyMuPDF and OCR ONNX runtimes
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code, pre-trained model artifacts, and sample data
COPY documind/ ./documind/
COPY artifacts/ ./artifacts/
COPY data/ ./data/
COPY scripts/ ./scripts/
COPY tests/ ./tests/
COPY pytest.ini .

# Expose HTTP port
EXPOSE 8000

# Health check probe
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Start FastAPI application
CMD ["uvicorn", "documind.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
