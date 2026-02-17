FROM python:3.11-slim

LABEL maintainer="Kevin Kawchak"
LABEL description="Physical AI Federated Oncology Trial Platform"

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the project
COPY . .

# Install the package in development mode
RUN pip install --no-cache-dir -e .

# Default command: run the federation simulation
CMD ["python", "examples/run_federation.py", "--num-sites", "3", "--rounds", "10"]
