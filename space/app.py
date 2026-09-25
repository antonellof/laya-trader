"""Hugging Face Space entry point (Gradio SDK on free ZeroGPU hardware).

The Space runs `python app.py` and serves whatever listens on port 7860, so this starts
the laya-trader dashboard there directly. Laya runs on the CPU with the PyTorch runtime.
"""

import os
import sys

import spaces


@spaces.GPU
def gpu_unused():
    """ZeroGPU requires one GPU function at startup. Laya never needs the GPU."""


os.environ.setdefault("LAYA_BACKEND", "torch")
sys.argv = [
    "laya_trader.py",
    "--host", "0.0.0.0",
    "--port", "7860",
    "--public",
    "--no-open",
    "--interval", "10",
    "--backtest-every-hours", "24",
    "--backtest-days", "30",
]  # fmt: skip

from laya_trader import main  # noqa: E402

main()
