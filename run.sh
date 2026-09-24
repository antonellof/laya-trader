#!/usr/bin/env bash
# Run laya-trader. Extra arguments go to the script, e.g. ./run.sh --once
set -euo pipefail
cd "$(dirname "$0")"
exec uv run python laya_trader.py "$@"
