# ── Stage 1: build React frontend ────────────────────────────────────────────
FROM node:20-alpine AS frontend-builder
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm ci --prefer-offline
COPY frontend/ ./
RUN npm run build

# ── Stage 2: Python runtime ───────────────────────────────────────────────────
FROM python:3.13-slim

# System libs required by opencv-python-headless and scikit-image
RUN apt-get update && apt-get install -y --no-install-recommends \
      libglib2.0-0 \
      libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application source
COPY src/       ./src/
COPY backend/   ./backend/
COPY haarcascade_mcs_nose.xml ./

# Demo images (test split — ~35MB, needed for gallery)
COPY data_split_6535/test/ ./data_split_6535/test/

# Built frontend from stage 1
COPY --from=frontend-builder /app/frontend/dist ./frontend/dist

# Models are NOT baked in. Set HF_MODEL_REPO=yourname/acne-cv-models.
# They download on first startup and cache in /tmp/hf_cache.
ENV PORT=8080
ENV HF_HOME=/tmp/hf_cache
EXPOSE 8080

CMD ["sh", "-c", "uvicorn backend.app:app --host 0.0.0.0 --port ${PORT}"]
