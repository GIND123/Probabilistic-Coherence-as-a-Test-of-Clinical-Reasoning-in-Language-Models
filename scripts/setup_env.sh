#!/usr/bin/env bash
# Full environment build for the CoDx project. Run once.
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH=/home/george/.local/bin:$PATH
export UV_HTTP_TIMEOUT=600
uv pip install --python .venv/bin/python -e . 
# Blackwell (sm_120) needs a CUDA 12.8+ torch build.
uv pip install --python .venv/bin/python torch torchvision --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv/bin/python vllm openai
uv pip install --python .venv/bin/python peft trl accelerate bitsandbytes
