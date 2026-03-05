# GPU Infrastructure Lab - Training Workbench
# Base image with PyTorch, CUDA toolkit stubs, and monitoring utilities
# Used for distributed training labs and GPU metrics collection.

FROM python:3.11-slim AS base

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    wget \
    git \
    vim \
    jq \
    net-tools \
    iputils-ping \
    && rm -rf /var/lib/apt/lists/*

# Install kubectl for Kubernetes lab exercises
RUN curl -LO "https://dl.k8s.io/release/v1.29.2/bin/linux/amd64/kubectl" \
    && chmod +x kubectl \
    && mv kubectl /usr/local/bin/

# Install Helm for GPU Operator deployment labs
RUN curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash

# Set up workspace
WORKDIR /workspace

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY manifests/ ./manifests/

# Expose ports for distributed training
# 29500 = PyTorch distributed master port
# 8888  = Jupyter (optional)
# 9400  = Metrics exporter
EXPOSE 29500 8888 9400

# Default command drops into a shell for interactive lab work
CMD ["bash"]
