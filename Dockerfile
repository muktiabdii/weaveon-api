FROM python:3.12-slim

WORKDIR /app

# Install deps untuk OpenCV + video
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && pip cache purge

# Copy source code
COPY . .

# Expose port (Railway pakai $PORT)
EXPOSE ${PORT:-8080}

# Jalankan uvicorn server dengan shell
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}