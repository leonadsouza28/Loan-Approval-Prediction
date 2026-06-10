# ── Base image ────────────────────────────────────────────────────────
FROM python:3.11-slim

# ── System dependencies ────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ──────────────────────────────────────────────────
WORKDIR /app

# ── Copy requirements first (layer cache optimisation) ────────────────
COPY requirements.txt .

# ── Install Python dependencies ────────────────────────────────────────
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy project files ─────────────────────────────────────────────────
COPY . .

# ── Train model if not already trained ────────────────────────────────
RUN python src/train.py

# ── Expose Streamlit port ──────────────────────────────────────────────
EXPOSE 8501

# ── Health-check ────────────────────────────────────────────────────────
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# ── Run Streamlit ──────────────────────────────────────────────────────
ENTRYPOINT ["streamlit", "run", "app.py", \
            "--server.port=8501", \
            "--server.address=0.0.0.0", \
            "--server.headless=true"]
