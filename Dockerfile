# Use a lightweight Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install dependencies and Java
RUN apt-get update && apt-get install -y \
    default-jdk \
    procps \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set JAVA_HOME environment variable
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH=$JAVA_HOME/bin:$PATH

ENV PYSPARK_PYTHON=python3
ENV PYSPARK_DRIVER_PYTHON=python3

# Copy requirements and install them
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and models
COPY src/ /app/src/
COPY models/ /app/models/
COPY data/ /app/data/

# Default command
CMD ["python", "/app/src/predict.py"]
