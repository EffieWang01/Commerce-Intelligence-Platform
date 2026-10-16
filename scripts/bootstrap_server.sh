#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/server_env.sh"
cd "$COMMERCE_ROOT"
mkdir -p tools python cache tmp logs artifacts
if [[ ! -x tools/uv ]]; then
    curl --fail --location --retry 3 --connect-timeout 15 --max-time 120 \
        https://astral.sh/uv/install.sh --output tools/uv-install.sh
    UV_INSTALL_DIR="$COMMERCE_ROOT/tools" sh tools/uv-install.sh
fi
tools/uv python install 3.11
if [[ ! -x .venv/bin/python ]]; then
    tools/uv venv --python 3.11 .venv
fi
tools/uv pip install --python .venv/bin/python \
    --index-url "${COMMERCE_PYPI_INDEX:-https://pypi.org/simple}" -e '.[ranking,test]'
.venv/bin/python --version
tools/uv pip freeze --python .venv/bin/python > artifacts/server-requirements.txt
