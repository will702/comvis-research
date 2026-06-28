#!/bin/bash
# Start the Acne CV Playground
# Backend (FastAPI) serves the built frontend on http://localhost:8000
cd "$(dirname "$0")"
.venv/bin/python3 -m uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
