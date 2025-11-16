# Base image for ARM64 architecture, Python 3.10
FROM --platform=linux/arm64 arm64v8/python:3.10-slim

# Set a non-interactive frontend for package installations
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies required by OpenCV and other libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    libjpeg-dev \
    zlib1g-dev \
    libgl1 \
    libglib2.0-0 \
    vim \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --- rknn_toolkit_lite2 & InspireFace Installation ---
COPY rknn-toolkit-lite2/packages/rknn_toolkit_lite2-2.3.2-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl /tmp/
COPY inspireface-1.2.3-cp310-cp310-linux_aarch64.whl /tmp/
RUN pip install --no-cache-dir \
    /tmp/rknn_toolkit_lite2-2.3.2-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl \
    /tmp/inspireface-1.2.3-cp310-cp310-linux_aarch64.whl && \
    rm /tmp/rknn_toolkit_lite2-2.3.2-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl \
       /tmp/inspireface-1.2.3-cp310-cp310-linux_aarch64.whl

# --- Application Setup ---
COPY rknn/ .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-load InspireFace models
RUN mkdir -p /root/.inspireface/ms/tunmxy/InspireFace/
COPY InspireFace/ /root/.inspireface/ms/tunmxy/InspireFace/

# Expose the port the application will run on
EXPOSE 8060

# Command to run the application
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8060"]
