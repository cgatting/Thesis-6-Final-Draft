# Stage 1: Build React Frontend
FROM node:18-slim AS frontend-builder
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY . .
# Set API URL to empty string so it uses relative paths (same origin)
ENV VITE_DEEPSEARCH_API_URL=""
RUN npm run build

# Stage 2: Python Backend
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
# python3-tk is needed for customtkinter imports (even if headless)
# build-essential for compiling some python packages
RUN apt-get update && apt-get install -y \
    python3-tk \
    tk-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user
RUN useradd -m -u 1000 user

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
# Use CPU-only PyTorch to save space (remove --extra-index-url if GPU is needed)
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
# Remove torch from requirements.txt to prevent reinstall from PyPI
RUN sed -i '/torch/d' requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download NLTK data
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# Pre-download ML models to bake them into the image
# This speeds up the first startup significantly
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2'); from transformers import pipeline; pipeline('summarization', model='sshleifer/distilbart-cnn-12-6')"

# Copy backend code
COPY DEEPSEARCH.py deepsearch_api.py ./

# Copy built frontend from Stage 1
COPY --from=frontend-builder /app/dist ./dist

# Set ownership to non-root user
RUN chown -R user:user /app

# Switch to non-root user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Expose port 7860 (Hugging Face Spaces default)
EXPOSE 7860

# Command to run the application
CMD ["uvicorn", "deepsearch_api:app", "--host", "0.0.0.0", "--port", "7860"]
