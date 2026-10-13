#!/usr/bin/env bash
# Run ON the pod after scripts/cloud/sync_data.sh has copied repo.bundle and data.
# Checks out the bundled commit (the GitHub repo is private, so no clone) and installs it.
#   bash /workspace/setup_pod.sh <commit>
set -euo pipefail
COMMIT=${1:?usage: setup_pod.sh <commit>}
ROOT=/workspace/peru-flood-segmentation

command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
# The venv and uv cache go on the container disk (they overfill the volume quota);
# the Hugging Face cache stays on the volume so a new pod reuses the weights.
export UV_CACHE_DIR=/root/.cache/uv UV_PROJECT_ENVIRONMENT=/root/venv HF_HOME=/workspace/.cache/huggingface

[ -d "$ROOT/.git" ] || git init -q "$ROOT"  # data/ may already be there
cd "$ROOT"
git fetch -q /workspace/repo.bundle main
git checkout -q --detach "$COMMIT"
git status --short | head
uv sync --frozen
uv run python -c "import torch; assert torch.cuda.is_available(), 'no CUDA'; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"
