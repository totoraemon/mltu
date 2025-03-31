# Base stage: Build your environment with CUDA and Python dependencies
FROM nvidia/cuda:12.8.0-devel-ubuntu24.04 AS base

# Set environment variables for CUDA and Python
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Install required packages
RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y --no-install-recommends \
        nano \
        git \
        apt-utils \
        sudo \
        wget \
        python3-dev \
        python3-pip \
        python3-venv \
        build-essential \
        libnss3 \
        libatk-bridge2.0-0 \
        libatk1.0-0 \
        libxcomposite1 \
        libxdamage1 \
        libxrandr2 \
        libgtk-3-0 \
        libgbm1 \
        xvfb \
        libpq-dev \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# Ensure "python" points to "python3"
RUN ln -s /usr/bin/python3 /usr/bin/python

# Upgrade pip and install TensorFlow with GPU support
RUN pip install --break-system-packages --upgrade --ignore-installed pip setuptools wheel \
&& pip install --break-system-packages tensorflow[and-cuda]==2.19.0

# Define user parameters
ARG USERNAME=vscode
ARG USER_UID=1001
ARG USER_GID=1001

# Create group and user if they don't exist
RUN if ! getent group $USER_GID; then groupadd --gid $USER_GID $USERNAME; fi && \
if ! id -u $USERNAME >/dev/null 2>&1; then \
useradd -s /bin/bash --uid $USER_UID --gid $USER_GID -m $USERNAME; \
fi && \
echo "$USERNAME ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/$USERNAME && \
chmod 0440 /etc/sudoers.d/$USERNAME

# Set working directory and default user in base stage
WORKDIR /home/$USERNAME
USER $USERNAME

# Final stage: Use the base image with all modifications
FROM base AS dev_containers_target_stage

# Explicitly set the user and working directory (this ensures the user entry is present)
USER vscode
WORKDIR /home/vscode

# Disable the warning about pip being installed in a system directory (this is a workaround for the warning) use only inside dockerfile
RUN python -m pip config set global.break-system-packages true

CMD ["/bin/bash"]