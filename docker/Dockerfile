# Usage:
# Self-contained build (default: builds from main): docker buildx build -f docker/Dockerfile --tag <registry>/nemo-rl:latest --push .
# Self-contained build (specific git ref): docker buildx build -f docker/Dockerfile --build-arg NRL_GIT_REF=r0.3.0 --tag <registry>/nemo-rl:r0.3.0 --push .
# Self-contained build (remote NeMo RL source; no need for a local clone of NeMo RL): docker buildx build -f docker/Dockerfile --build-arg NRL_GIT_REF=r0.3.0 --tag <registry>/nemo-rl:r0.3.0 --push https://github.com/NVIDIA-NeMo/RL.git
# Local NeMo RL source override: docker buildx build --build-context nemo-rl=. -f docker/Dockerfile --tag <registry>/nemo-rl:latest --push .

ARG BASE_IMAGE=nvcr.io/nvidia/cuda-dl-base:25.05-cuda12.9-devel-ubuntu24.04
FROM scratch AS nemo-rl
ARG NRL_GIT_REF=main
ADD --keep-git-dir=true https://github.com/NVIDIA-NeMo/RL.git#${NRL_GIT_REF} /

FROM ${BASE_IMAGE} AS base

# It is more convenient for users to run as root
USER root

RUN <<"EOF" bash -exu -o pipefail
export DEBIAN_FRONTEND=noninteractive
export TZ=America/Los_Angeles

apt-get update
apt-get install -y --no-install-recommends \
    git \
    curl \
    git \
    rsync \
    wget \
    less \
    vim \
    openssh-server \
    nginx \
    software-properties-common \

# Nsight
apt install -y --no-install-recommends gnupg
echo "deb http://developer.download.nvidia.com/devtools/repos/ubuntu$(source /etc/lsb-release; echo "$DISTRIB_RELEASE" | tr -d .)/$(dpkg --print-architecture) /" | tee /etc/apt/sources.list.d/nvidia-devtools.list
apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
apt update
apt install -y nsight-systems-cli


apt-get clean
rm -rf /var/lib/apt/lists/*

# Configure SSH for RunPod
mkdir -p /var/run/sshd
mkdir -p /root/.ssh

# SSH configuration for RunPod compatibility
echo 'PermitRootLogin yes' >> /etc/ssh/sshd_config
echo 'PasswordAuthentication yes' >> /etc/ssh/sshd_config
echo 'PubkeyAuthentication yes' >> /etc/ssh/sshd_config
echo 'Port 22' >> /etc/ssh/sshd_config
echo 'AddressFamily inet' >> /etc/ssh/sshd_config

# Set root password (RunPod will override this)
echo 'root:runpod' | chpasswd

# Generate SSH host keys
ssh-keygen -A

EOF

# Install uv and python
ARG UV_VERSION=0.7.2
ARG PYTHON_VERSION=3.12
ENV PATH="/root/.local/bin:$PATH"
ENV CUDA_HOME=/usr/local/cuda
ENV MAX_JOBS=2
ENV FLASH_ATTENTION_FORCE_BUILD=FALSE
ENV FLASH_ATTENTION_USE_PREBUILT=TRUE
RUN curl -LsSf https://astral.sh/uv/${UV_VERSION}/install.sh | sh && \
    uv python install ${PYTHON_VERSION}

# Disable usage stats by default for users who are sensitive to sharing usage.
# Users are encouraged to enable if the wish.
ENV RAY_USAGE_STATS_ENABLED=0
ENV NEMO_RL_VENV_DIR=/opt/ray_venvs


FROM base AS hermetic

WORKDIR /opt/nemo-rl

# Variables to control the build of TE. If there are issues with parallelization, consider
# setting these to 1.
ARG MAX_JOBS
ARG NVTE_BUILD_THREADS_PER_JOB

ENV UV_PROJECT_ENVIRONMENT=/opt/nemo_rl_venv
ENV UV_LINK_MODE=copy

# This step is to warm the uv cache with flash-attn without invalidating it due to COPY layers
# This layer has to be manually updated
RUN <<"EOF" bash -exu
uv venv ${UV_PROJECT_ENVIRONMENT}

# NOTE: Use pre-built wheels to avoid compilation - adjust PyTorch version to match wheel
# Install PyTorch 2.5 with CUDA 12.4 to match the flash-attn wheel
VIRTUAL_ENV=$UV_PROJECT_ENVIRONMENT uv pip install --link-mode symlink setuptools torch==2.5.0 psutil ninja --torch-backend=cu124
# Download and install pre-built flash-attn wheel to avoid memory-intensive compilation
wget -q https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.3.14/flash_attn-2.6.3+cu124torch2.5-cp312-cp312-linux_x86_64.whl
VIRTUAL_ENV=$UV_PROJECT_ENVIRONMENT uv pip install --link-mode symlink flash_attn-2.6.3+cu124torch2.5-cp312-cp312-linux_x86_64.whl
EOF

# First copy only the dependency files
COPY --from=nemo-rl pyproject.toml uv.lock ./
COPY --from=nemo-rl --link 3rdparty/ ./3rdparty/

RUN <<"EOF" bash -exu
# uv sync has a more reliable resolver than simple uv pip install which can fail

# Sync each training + inference backend one at a time (since they may conflict)
# to warm the uv cache, then at the end just sync the default dependencies.
# Do everything in one layer to prevent large layers.

# The venv is symlinked to avoid bloating the layer size
uv sync --link-mode symlink --locked --no-install-project
uv sync --link-mode symlink --locked --extra vllm --no-install-project
uv sync --link-mode symlink --locked --extra mcore --no-install-project
uv sync --link-mode symlink --locked --all-groups --no-install-project

# Install Jupyter
VIRTUAL_ENV=$UV_PROJECT_ENVIRONMENT uv pip install --link-mode symlink \
    jupyterlab \
    notebook \
    ipywidgets
EOF

ENV PATH="/opt/nemo_rl_venv/bin:$PATH"
ENV NEMO_RL_VENV_DIR=/opt/ray_venvs

# Create workspace and set as working directory
RUN mkdir -p /workspace
WORKDIR /workspace

FROM hermetic AS release

ARG NEMO_RL_COMMIT
ARG NVIDIA_BUILD_ID
ARG NVIDIA_BUILD_REF
ENV NEMO_RL_COMMIT=${NEMO_RL_COMMIT:-<unknown>}
ENV NVIDIA_BUILD_ID=${NVIDIA_BUILD_ID:-<unknown>}
ENV NVIDIA_BUILD_REF=${NVIDIA_BUILD_REF:-<unknown>}
LABEL com.nvidia.build.id="${NVIDIA_BUILD_ID}"
LABEL com.nvidia.build.ref="${NVIDIA_BUILD_REF}"

ENV NEMO_RL_VENV_DIR=/opt/ray_venvs

# Copy NeMo RL source to workspace
COPY --from=nemo-rl . /workspace/nemo-rl
WORKDIR /workspace/nemo-rl
# Unshallow the repo to get the full history (in the case it was from the scratch layer).
# Potentially not necessary if the repo is passed in as a complete repository (w/ full git history),
# so do a quick check before trying to unshallow.
RUN git rev-parse --is-shallow-repository | grep -q true && git fetch --unshallow || true
RUN UV_LINK_MODE=symlink uv run nemo_rl/utils/prefetch_venvs.py

# Create simple startup script for RunPod
RUN <<"EOF" cat > /start.sh
#!/bin/bash

# Start SSH service
service ssh start

# Start Jupyter Lab
jupyter lab --ip=0.0.0.0 --port=8888 --allow-root --no-browser --notebook-dir=/workspace &

# Keep container running
if [ $# -eq 0 ]; then
    echo "✅ NeMo RL environment ready!"
    echo "📁 Workspace: /workspace/nemo-rl"
    echo "📓 JupyterLab: http://localhost:8888"
    tail -f /dev/null
else
    exec "$@"
fi
EOF

RUN chmod +x /start.sh

# Expose ports for SSH and Jupyter
EXPOSE 22 8888

# Set default command to comprehensive startup script
CMD ["/start.sh"]

