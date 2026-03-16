# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies (ffmpeg for audio processing)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Create the data directory (persists via ECS volumes or local bind mount)
RUN mkdir -p data

# Expose the port the app runs on
EXPOSE 8000

# Health check for ECS / load balancer
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the FastAPI application
# Note: 'app:app' = module 'app.py', FastAPI instance named 'app'
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
