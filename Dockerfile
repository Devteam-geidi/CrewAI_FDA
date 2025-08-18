FROM python:3.10-slim AS builder
WORKDIR /app

# Install only the build dependencies needed
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    gcc \
    g++ \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Create a modified requirements file without chromadb and related packages
RUN grep -v -E "chromadb|sentence-transformers|torch|hnswlib" requirements.txt > requirements-no-chroma.txt

# Install dependencies with pip
RUN pip install --no-cache-dir -r requirements-no-chroma.txt

# Second stage - runtime image
FROM python:3.10-slim
WORKDIR /app

# Install only the minimal runtime dependencies
RUN apt-get update && apt-get install -y \
    sqlite3 \
    libsqlite3-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Create directory for persistent storage
RUN mkdir -p /app/data
ENV LAST_AGENTS_FILE="/app/data/last_agents.json"

# Copy application files - one at a time to avoid globbing issues
COPY main.py .

# Use a more Docker-friendly approach to copy optional files
COPY *.py ./
COPY *.json ./

# The above commands may fail if no matching files exist, but Docker will continue
# We don't need the "|| :" syntax which doesn't work in Dockerfile

# Expose port for Cloud Run
EXPOSE 8080

# Run the FastAPI app
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}"]