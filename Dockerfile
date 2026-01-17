# Base image for SWE-bench task environments
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set up workspace
WORKDIR /workspace

# Install common Python testing tools
RUN pip install --no-cache-dir \
    pytest \
    pytest-xdist \
    coverage

# Default command
CMD ["/bin/bash"]
