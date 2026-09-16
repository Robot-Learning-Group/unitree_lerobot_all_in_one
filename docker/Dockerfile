# syntax=docker/dockerfile:1
FROM mambaorg/micromamba:2.3.3 AS micromamba
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG MAX_JOBS=2
ENV MAMBA_ROOT_PREFIX=/opt/conda \
    PATH=/opt/conda/envs/lerobot/bin:/usr/local/cuda/bin:${PATH} \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/workspace/unitree_lerobot/.cache/huggingface \
    HF_LEROBOT_HOME=/workspace/unitree_lerobot/data \
    TORCH_HOME=/workspace/unitree_lerobot/.cache/torch \
    MPLCONFIGDIR=/workspace/unitree_lerobot/.cache/matplotlib \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,video

COPY --from=micromamba /bin/micromamba /usr/local/bin/micromamba
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential ca-certificates cmake git gosu ninja-build pkg-config \
      libegl1 libgl1 libglib2.0-0 libgomp1 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*
RUN micromamba create -y -n lerobot -c conda-forge --strict-channel-priority \
      python=3.10 pip 'numpy=2.2.6' 'pinocchio=3.3.1' 'ffmpeg=7.1.1' \
    && micromamba clean --all --yes

WORKDIR /workspace/unitree_lerobot
COPY docker/constraints.txt /opt/constraints.txt
ENV PIP_CONSTRAINT=/opt/constraints.txt
RUN python -m pip install --no-cache-dir 'setuptools<81' wheel packaging ninja \
    && python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu128 \
      torch==2.7.1 torchvision==0.22.1 torchcodec==0.5 \
    && MAX_JOBS=${MAX_JOBS} python -m pip install --no-cache-dir --no-build-isolation flash-attn==2.8.3

COPY unitree_sdk2_python /workspace/unitree_sdk2_python
COPY unitree_lerobot /workspace/unitree_lerobot
# Unitree v3.0 uses basic_config/get_logger, provided by logging_mp 0.1.6.
RUN python -m pip install --no-cache-dir \
      -e './unitree_lerobot/lerobot[groot]' -e /workspace/unitree_sdk2_python -e . \
      'pyzmq>=26.2.1,<28' pyyaml 'logging_mp==0.1.6'

# These checks run without a GPU, model downloads, cameras or DDS initialization.
COPY docker/smoke_test.py /opt/smoke_test.py
RUN python /opt/smoke_test.py
COPY docker/entrypoint.sh /usr/local/bin/lerobot-entrypoint
RUN chmod 755 /usr/local/bin/lerobot-entrypoint
ENTRYPOINT ["/usr/local/bin/lerobot-entrypoint"]
CMD ["bash"]
