# SageMaker training image for openpi (PyTorch trainer for pi05_libero).
#
# Differs from scripts/docker/train.Dockerfile (DGX/local) in three ways:
#   1. Bakes the openpi source into the image (no bind mount on SageMaker).
#   2. Default CMD is a shell wrapper that translates SageMaker's SM_CHANNEL_*
#      env vars into the env vars + paths openpi expects.
#   3. Caches resolve to /opt/ml/... so they live on SageMaker's writable
#      volume, not the read-only image layers.
#
# Build:
#   docker build -f scripts/sagemaker/train.Dockerfile \
#       --build-arg AWS_REGION=us-west-2 -t openpi_sagemaker_train .
ARG AWS_REGION=us-west-2
ARG DLC_TAG=2.8.0-gpu-py312-cu129-ubuntu22.04-sagemaker
FROM 763104351884.dkr.ecr.${AWS_REGION}.amazonaws.com/pytorch-training:${DLC_TAG}

# uv binary for fast deps install (we still use uv to materialize the locked env,
# even though the DLC already has a system Python — it's the simplest way to get
# JAX + Orbax + LeRobot pinned correctly).
COPY --from=ghcr.io/astral-sh/uv:0.5.1 /uv /uvx /bin/

WORKDIR /opt/ml/code

RUN apt-get update && apt-get install -y --no-install-recommends \
    git git-lfs build-essential libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

ENV UV_LINK_MODE=copy
ENV UV_PROJECT_ENVIRONMENT=/.venv
ENV UV_PYTHON_INSTALL_DIR=/opt/uv-python

RUN uv venv --python 3.11.9 $UV_PROJECT_ENVIRONMENT

# Sync locked deps before copying source so this layer caches across code edits.
COPY pyproject.toml uv.lock /tmp/openpi/
COPY packages/openpi-client/pyproject.toml /tmp/openpi/packages/openpi-client/pyproject.toml
COPY packages/openpi-client/src /tmp/openpi/packages/openpi-client/src
COPY README.md LICENSE /tmp/openpi/
RUN --mount=type=cache,target=/root/.cache/uv \
    cd /tmp/openpi && \
    GIT_LFS_SKIP_SMUDGE=1 uv sync --frozen --no-install-project --no-dev

# Bake the openpi source into the image. SageMaker has no bind mount, and we
# want submitted jobs to be reproducible by image tag.
# `assets/` ships the (tiny, ~20 KB) norm_stats.json so we don't burn an S3
# channel on it; sm_entrypoint.sh passes --assets-base-dir=/opt/ml/code/assets.
COPY src /opt/ml/code/src
COPY scripts /opt/ml/code/scripts
COPY packages /opt/ml/code/packages
COPY assets /opt/ml/code/assets
COPY pyproject.toml uv.lock README.md LICENSE /opt/ml/code/

RUN uv pip install --python $UV_PROJECT_ENVIRONMENT --no-deps /opt/ml/code/packages/openpi-client \
 && uv pip install --python $UV_PROJECT_ENVIRONMENT --no-deps /opt/ml/code

# Apply the transformers patch the PyTorch path needs (AdaRMS, KV cache w/o
# update, etc.). Same operation as the DGX image.
RUN /.venv/bin/python -c "import transformers; print(transformers.__file__)" \
    | xargs dirname \
    | xargs -I{} cp -r /opt/ml/code/src/openpi/models_pytorch/transformers_replace/. {}

# SageMaker mounts:
#   /opt/ml/input/data/<channel>/   (read-only on FastFile, RW on File)
#   /opt/ml/checkpoints/            (synced to checkpoint_s3_uri, RW)
#   /opt/ml/output/                 (synced to output_path on exit, RW)
# Caches must live on writable volumes — under /opt/ml/, not in image layers.
ENV HOME=/opt/ml/code
# LeRobot reads from $HF_LEROBOT_HOME/<repo_id>/{data,meta,...} — point it
# at the SageMaker `libero` channel so dataset I/O streams from S3 FastFile.
# HF_HOME stays at /opt/ml/output/hf so any HF Hub code that runs in-container
# (snapshot_download fallbacks, etc.) writes to the syncable output volume.
ENV HF_LEROBOT_HOME=/opt/ml/input/data/libero
ENV HF_HOME=/opt/ml/output/hf
ENV OPENPI_DATA_HOME=/opt/ml/openpi_assets
ENV WANDB_DIR=/opt/ml/output/wandb
ENV TRITON_CACHE_DIR=/opt/ml/output/triton

ENV PATH=/.venv/bin:$PATH
ENV PYTHONPATH=/opt/ml/code/src:/opt/ml/code/packages/openpi-client/src:${PYTHONPATH}

# SageMaker contract.
ENV SAGEMAKER_SUBMIT_DIRECTORY=/opt/ml/code
ENV SAGEMAKER_PROGRAM=sm_entrypoint.sh

COPY scripts/sagemaker/sm_entrypoint.sh /opt/ml/code/sm_entrypoint.sh
RUN chmod +x /opt/ml/code/sm_entrypoint.sh

# SageMaker invokes the entrypoint with hyperparameters appended; we ignore
# them and read everything from env vars instead (set in launch.py).
ENTRYPOINT ["/opt/ml/code/sm_entrypoint.sh"]
