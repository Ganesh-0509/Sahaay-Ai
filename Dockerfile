# Use official Python runtime
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies (needed by some Python libs like python-magic)
RUN apt-get update && apt-get install -y \
    build-essential \
    libmagic1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt gunicorn

# Copy rest of the app
COPY . .

# Set environment variables
ENV PORT=8080
ENV PYTHONUNBUFFERED=True

# Run the app with Gunicorn
CMD ["gunicorn", "-b", ":$PORT", "app:app"]
