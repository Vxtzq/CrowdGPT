# syntax=docker/dockerfile:1.7

ARG BASE_IMAGE=pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime
FROM ${BASE_IMAGE}

ARG BACKEND=cuda
ARG TORCH_INDEX=""
ARG TORCH_SPEC="torch"
ARG INSTALL_BITSANDBYTES=auto

ENV PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false \
    CROWDGPT_DATA_DIR=/data \
    HF_HOME=/data/hf_home \
    TORCH_HOME=/data/torch_home \
    CUDA_CACHE_PATH=/data/cuda_cache \
    XDG_CACHE_HOME=/data/cache \
    HOME=/home/crowd

# gosu: drop root; ca-certificates/curl: HTTPS & diagnostics.
# On XPU builds, install the Intel Level Zero loader and OpenCL ICD loader.
RUN apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      gosu ca-certificates curl \
 && if ! command -v python >/dev/null 2>&1 && command -v python3 >/dev/null 2>&1; then \
      ln -s "$(command -v python3)" /usr/local/bin/python; \
    fi \
 && if [ "${BACKEND}" = "xpu" ]; then \
      DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        libze1 ocl-icd-libopencl1 || true; \
    fi \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies before copying repo code so code edits keep the pip layer cached.
COPY requirements.txt /tmp/requirements.txt
COPY --chmod=0755 docker/install-deps.sh /usr/local/bin/install-deps.sh
RUN --mount=type=cache,target=/root/.cache/pip \
    /usr/local/bin/install-deps.sh \
      --backend "${BACKEND}" \
      --torch-index "${TORCH_INDEX}" \
      --torch-spec "${TORCH_SPEC}" \
      --bitsandbytes "${INSTALL_BITSANDBYTES}" \
 && rm -f /tmp/requirements.txt /tmp/requirements.docker.txt

# Copy application code into /app (read-only at runtime; persistent state lives in /data).
COPY . /app

RUN mkdir -p /data/checkpoints /data/hf_home /data/torch_home /data/cache \
 && rm -rf /app/checkpoints \
 && ln -sfn /data/checkpoints /app/checkpoints \
 && (id -u crowd >/dev/null 2>&1 || useradd -m -s /bin/bash crowd) \
 && groupadd -f video \
 && groupadd -f render \
 && usermod -aG video,render crowd \
 && chown -R crowd:crowd /app /data /home/crowd

VOLUME ["/data"]

COPY --chmod=0755 docker/docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["--headless"]
