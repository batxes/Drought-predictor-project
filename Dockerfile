# Use the official Python image.
FROM python:3.12-slim

# Install build dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    make \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy the requirements file first to leverage Docker cache
COPY requirements.txt ./

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project directory to the container
COPY . .

# Expose the port on which your service will run
EXPOSE 5000

# Set environment variables
ENV MODEL_PATH=/app/models/model_xgboost.bin
ENV DV_PATH=/app/models/dv.bin

# Command to run the Flask app with Gunicorn
CMD ["gunicorn", "-b", "0.0.0.0:5000", "scripts.predict:app"]