# Dockerfile for fine-tuning openpi models (JAX + PyTorch).
# Mirrors serve_policy.Dockerfile but is built for training: it also applies the
# transformers patch needed by the PyTorch path so the same image can run both
# scripts/train.py and scripts/train_pytorch.py.
#
# Build:
#   docker build . -t openpi_train -f scripts/docker/train.Dockerfile
#
# Run (manual; prefer the compose file):
#   docker run --rm -it --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all \
#       --network=host --shm-size=16g \
#       -v $PWD:/app \
#       -v $HOME/.cache/openpi:/openpi_assets \
#       -v $HOME/.cache/huggingface:/root/.cache/huggingface \
#       openpi_train /bin/bash

FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04@sha256:2d913b09e6be8387e1a10976933642c73c840c0b735f0bf3c28d97fc9bc422e0
COPY --from=ghcr.io/astral-sh/uv:0.5.1 /uv /uvx /bin/

WORKDIR /app

RUN apt-get update && apt-get install -y \
    git git-lfs linux-headers-generic build-essential clang libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

ENV UV_LINK_MODE=copy
ENV UV_PROJECT_ENVIRONMENT=/.venv
# Install Python under a world-readable location so the container can run under
# the host user's UID. Default uv install dir is under root's HOME, which a
# non-root user can't traverse — that breaks /.venv/bin/python -> /root/...
ENV UV_PYTHON_INSTALL_DIR=/opt/uv-python

RUN uv venv --python 3.11.9 $UV_PROJECT_ENVIRONMENT
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=packages/openpi-client/pyproject.toml,target=packages/openpi-client/pyproject.toml \
    --mount=type=bind,source=packages/openpi-client/src,target=packages/openpi-client/src \
    GIT_LFS_SKIP_SMUDGE=1 uv sync --frozen --no-install-project --no-dev

# Install the openpi package into the venv so `python -m` / direct invocation
# works without uv re-syncing at container start. We don't COPY the source —
# instead we let pip install the package from the bind mount, then we drop the
# editable `.pth` so the container can run under a non-root UID without uv
# trying to mutate the venv. The repo will be bind-mounted at runtime, so the
# package directory is on the path via $PYTHONPATH below.
COPY pyproject.toml /tmp/openpi/pyproject.toml
COPY README.md /tmp/openpi/README.md
COPY LICENSE /tmp/openpi/LICENSE
COPY packages/openpi-client /tmp/openpi/packages/openpi-client
COPY src /tmp/openpi/src
RUN uv pip install --python $UV_PROJECT_ENVIRONMENT --no-deps /tmp/openpi/packages/openpi-client \
 && uv pip install --python $UV_PROJECT_ENVIRONMENT --no-deps /tmp/openpi \
 && rm -rf /tmp/openpi

# Make the venv + uv python install readable+executable by non-root users so the
# container can run under the host user's UID for clean file ownership on
# bind-mounted outputs.
RUN chmod -R a+rX $UV_PROJECT_ENVIRONMENT $UV_PYTHON_INSTALL_DIR

# Apply the transformers monkey-patch required by models_pytorch (AdaRMS, activation
# precision controls, KV cache without update). Safe no-op if the JAX trainer is used.
COPY src/openpi/models_pytorch/transformers_replace/ /tmp/transformers_replace/
RUN /.venv/bin/python -c "import transformers; print(transformers.__file__)" \
    | xargs dirname \
    | xargs -I{} cp -r /tmp/transformers_replace/. {} \
 && rm -rf /tmp/transformers_replace

# Caches resolve to bind-mounted host directories at runtime (compose handles this).
ENV OPENPI_DATA_HOME=/openpi_assets
ENV HF_HOME=/cache/huggingface
# Run-as-host-user friendly: uv/HF/wandb/triton all need a writable HOME.
# /cache is created world-writable below; bind-mounting subdirs over it is fine.
ENV HOME=/cache
ENV UV_CACHE_DIR=/cache/uv
ENV TRITON_CACHE_DIR=/cache/triton
ENV WANDB_DIR=/app/wandb
RUN mkdir -p /cache/huggingface /cache/uv /cache/triton && chmod -R 0777 /cache
# Give JAX more device memory by default; override at runtime if needed.
ENV XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

# Use the prebuilt venv directly (no `uv run`/`uv sync` at container start).
# The repo is bind-mounted at /app, so scripts/* and the latest src/* are
# picked up. PYTHONPATH ensures the bind-mounted source is preferred over
# the snapshot we baked into the image during build.
ENV PATH=/.venv/bin:$PATH
ENV PYTHONPATH=/app/src:/app/packages/openpi-client/src:$PYTHONPATH

# Default entrypoint: JAX trainer. Override with TRAIN_CMD for the PyTorch
# path or for ad-hoc commands (norm stats, conversion, shells, etc.).
CMD ["/bin/bash", "-c", "${TRAIN_CMD:-python scripts/train.py ${TRAIN_CONFIG:-pi05_libero_debug} --exp-name=${EXP_NAME:-debug} ${TRAIN_ARGS:-}}"]
