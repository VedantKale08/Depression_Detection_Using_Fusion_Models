# Use official Python runtime as a parent image
FROM python:3.10-slim

# Install system dependencies needed for Audio/Video processing
# We also install git just in case any pip package needs it
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install Python dependencies, including gunicorn for the production server
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir gunicorn

# Copy the rest of the backend files needed for inference
# We specifically exclude the frontend directory to keep the image lightweight
COPY app.py .
COPY src/ ./src/
COPY Audio/ ./Audio/
COPY Face/ ./Face/
COPY Text/ ./Text/
COPY models/ ./models/
COPY weights/ ./weights/
COPY data/ ./data/

# Expose port 7860 (Hugging Face Spaces default port)
EXPOSE 7860

# Command to run the Flask application using Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--timeout", "120", "app:app"]
