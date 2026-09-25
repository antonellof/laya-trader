#!/usr/bin/env bash
# Deploy the dashboard to a free Hugging Face Space (Gradio SDK, ZeroGPU hardware; Laya
# runs on the CPU with its PyTorch runtime).
# Needs `uv run hf auth login` with a write token. Usage: ./deploy_space.sh [user/space]
set -euo pipefail
cd "$(dirname "$0")"
SPACE="${1:-$(uv run hf auth whoami | sed -n 's/^user=//p')/laya-trader}"
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

cp config.toml core.py markets.py laya_trader.py backtest.py dashboard.html "$STAGE"/
cp space/app.py space/requirements.txt space/README.md "$STAGE"/
# The latest local backtest, shown until the Space builds its own (it runs daily).
[ -f backtest.html ] && cp backtest.html "$STAGE"/

uv run hf repos create "$SPACE" --type space --sdk gradio --flavor zero-a10g --public --exist-ok
uv run hf upload "$SPACE" "$STAGE" . --repo-type space --commit-message "Deploy laya-trader"
echo "https://huggingface.co/spaces/$SPACE"
