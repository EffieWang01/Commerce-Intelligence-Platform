#!/usr/bin/env bash
# Source this file before working on the server.
export COMMERCE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export XDG_CACHE_HOME="$COMMERCE_ROOT/cache"
export XDG_DATA_HOME="$COMMERCE_ROOT/cache/data"
export XDG_CONFIG_HOME="$COMMERCE_ROOT/cache/config"
export UV_CACHE_DIR="$COMMERCE_ROOT/cache/uv"
export UV_PYTHON_INSTALL_DIR="$COMMERCE_ROOT/python"
export UV_PYTHON_BIN_DIR="$COMMERCE_ROOT/tools"
export UV_NO_MODIFY_PATH=1
export UV_NO_INSTALLER_METADATA=1
export PIP_CACHE_DIR="$COMMERCE_ROOT/cache/pip"
export HF_HOME="$COMMERCE_ROOT/cache/huggingface"
export TORCH_HOME="$COMMERCE_ROOT/cache/torch"
export TMPDIR="$COMMERCE_ROOT/tmp"
export PYTHONPATH="$COMMERCE_ROOT/src:$COMMERCE_ROOT"
export PATH="$COMMERCE_ROOT/.venv/bin:$COMMERCE_ROOT/tools:$PATH"
export COMMERCE_DATA_DIR="$COMMERCE_ROOT/data/demo"
export COMMERCE_ARTIFACTS_DIR="$COMMERCE_ROOT/artifacts"
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
