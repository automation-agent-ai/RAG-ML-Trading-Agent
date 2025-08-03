# Use slim Python base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install dependencies for headless Chromium (for scraping or rendering if needed)
RUN apt-get update && apt-get install -y \
    curl \
    unzip \
    xvfb \
    chromium \
    chromium-driver \
    && rm -rf /var/lib/apt/lists/*

# Environment settings
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Copy requirements first to leverage Docker layer caching
# Upgrade pip first
RUN pip install --upgrade pip

# Then install requirements
COPY requirements.txt .
RUN pip install -r requirements.txt


# Copy the rest of the app into the container
COPY . .

# Expose the backend port (your FastAPI backend runs here)
EXPOSE 8000

# NOTE:
# Do NOT include the DuckDB file inside the image – mount it at runtime via docker -v option

# Start the backend (this will wait until the FastAPI server is up)
CMD ["python", "frontend/start.py"]
