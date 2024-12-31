FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

# Avoid prompts from apt
ENV DEBIAN_FRONTEND=noninteractive

# Install basic development tools
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    curl \
    git \
    pkg-config \
    python3 \
    python3-pip \
    sudo \
    wget \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# Switch to user and set up bash configuration
USER $USERNAME
WORKDIR /home/$USERNAME

# Set up a better bash prompt and aliases
RUN echo 'export PS1="\[\033[01;32m\]\u@\h\[\033[00m\]:\[\033[01;34m\]\w\[\033[00m\]\$ "' >> ~/.bashrc && \
    echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc && \
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc && \
    echo 'alias ll="ls -lah"' >> ~/.bashrc && \
    echo 'alias ..="cd .."' >> ~/.bashrc && \
    echo 'alias ...="cd ../.."' >> ~/.bashrc && \
    echo 'alias gpu="nvidia-smi"' >> ~/.bashrc && \
    echo 'alias nvcc-ver="nvcc --version"' >> ~/.bashrc

# Set bash as default shell
SHELL ["/bin/bash", "-c"]