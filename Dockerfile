# =========================================================================
# Multi-Stage Dockerfile for OCUNEXA on Hugging Face Spaces (Port 7860)
# =========================================================================

# Stage 1: Build React Kiosk Frontend
FROM node:20-slim AS frontend-builder
WORKDIR /app/frontend

COPY sih26038-clinical-ai-screening-kiosk/package*.json ./
RUN npm install

COPY sih26038-clinical-ai-screening-kiosk/ ./
RUN npm run build

# Stage 2: Python Backend & AI Engine
FROM python:3.11-slim

# Create user with UID 1000 for Hugging Face Spaces
RUN useradd -m -u 1000 user
WORKDIR /home/user/app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch CPU and Python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy backend files
COPY . .

# Copy compiled React frontend from Stage 1
COPY --from=frontend-builder /app/frontend/dist ./sih26038-clinical-ai-screening-kiosk/dist

# Set ownership to non-root user
RUN chown -R user:user /home/user/app
USER user

# Hugging Face Spaces requires port 7860
ENV PORT=7860
EXPOSE 7860

# Launch OCUNEXA AI Server
CMD ["python", "server.py"]
