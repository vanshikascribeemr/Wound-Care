# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies for audio processing if needed
RUN apt-get update && apt-get install -y \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Create the data directory (ensure it persists via volumes in your deployment)
RUN mkdir -p data

# Expose the port the app runs on
EXPOSE 8000

# Run the application
CMD ["uvicorn", "app.py:app", "--host", "0.0.0.0", "--port", "8000"]
