# Use official Python 3.12 image
FROM python:3.12-slim-bookworm

# Set environment variables using key=value format
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies including build tools
RUN apt-get update && \
    apt-get install -y git build-essential g++ && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Remove build tools to reduce image size
RUN apt-get purge -y --auto-remove build-essential g++ && \
    rm -rf /var/lib/apt/lists/*

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

# Note:
# - APScheduler 3.10.1 is used for Python 3.12 compatibility
# - ChromaDB >=0.5.0 is used for Python 3.12 compatibility
# - Build tools are installed temporarily for ChromaDB compilation