FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Set CUDA environment variables
ENV CUDA_HOME="/usr/local/cuda" \
    PATH="/usr/local/cuda/bin:${PATH}" \
    LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

# Install Python 3.11 and required build tools
RUN apt-get update && \
    apt-get install -y \
    software-properties-common \
    build-essential \
    wget \
    git \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y \
    python3.11 \
    python3.11-distutils \
    python3.11-dev \
    ninja-build \
    python3-pip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Make python3.11 the default python3
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Upgrade pip and install basic requirements
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install --upgrade setuptools wheel

# Install NumPy first - this is critical
RUN pip3 install numpy==1.24.3

# Install PyTorch with CUDA support
RUN pip3 install --no-cache-dir torch==2.5.0+cu124 torchvision==0.20.0+cu124 --extra-index-url https://download.pytorch.org/whl/cu124

# Copy wheel file
COPY wheels/flash_attn-2.6.3-cp311-cp311-linux_x86_64.whl /app/wheels/

# Install other dependencies
RUN pip3 install --no-cache-dir \
    transformers==4.45.0 \
    accelerate==0.34.1 \
    sentencepiece==0.2.0 \
    Pillow \
    requests \
    /app/wheels/flash_attn-2.6.3-cp311-cp311-linux_x86_64.whl

# Install grouped_gemm with specific CUDA architecture and build settings
RUN TORCH_CUDA_ARCH_LIST="8.6" \
    CUDA_HOME=/usr/local/cuda \
    pip3 install --no-build-isolation --no-deps --verbose grouped_gemm==0.1.6

COPY scripts/inference_script.py /app/scripts/

CMD ["python3", "scripts/inference_script.py"]