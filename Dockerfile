# TokenLedger Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy package files
COPY pyproject.toml .
COPY tokenledger/ tokenledger/

# Install Python package with all dependencies
RUN pip install --no-cache-dir -e ".[server]"

# Expose API port
EXPOSE 8765

# Default command
CMD ["python", "-m", "tokenledger.server"]
