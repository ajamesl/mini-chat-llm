FROM python:3.12-slim

WORKDIR /app

# Install basic utilities for debugging (optional)
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Copy code and checkpoints (ensure .dockerignore skips large/unneeded files if needed)
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Set environment variable for unbuffered output (for logs)
ENV PYTHONUNBUFFERED=1

# Expose FastAPI port
EXPOSE 7000

# (Optional) Set TRANSFORMERS_CACHE if you want to use a local cache for HuggingFace models

# Start the app using uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7000"]