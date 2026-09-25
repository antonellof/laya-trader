"""Hugging Face Space entry point (Gradio SDK on free ZeroGPU hardware).

ZeroGPU starts only through Gradio's launch() and needs one @spaces.GPU function, so this
launches a minimal Gradio app on port 7860, runs the laya-trader dashboard on an internal
port, and routes the dashboard's pages through Gradio's server in place of Gradio's own
page. Laya runs on the CPU with the PyTorch runtime; the GPU is never used.
"""

import os
import sys

import gradio as gr
import httpx
import spaces
from starlette.responses import Response
from starlette.routing import Route

INTERNAL_PORT = 8765
PATHS = [
    "/",
    "/index.html",
    "/state.json",
    "/decision",
    "/decisions.jsonl",
    "/backtest",
    "/backtest/",
    "/backtest/{rest:path}",
]


@spaces.GPU
def gpu_unused():
    """ZeroGPU requires one GPU function at startup. Laya never needs the GPU."""


client = httpx.AsyncClient(base_url=f"http://127.0.0.1:{INTERNAL_PORT}", timeout=30)


async def proxy(request):
    try:
        upstream = await client.request(
            request.method,
            request.url.path,
            params=request.query_params,
            content=await request.body(),
        )
    except httpx.ConnectError:  # the dashboard starts after the model loads
        return Response("Starting: loading Laya, try again in a minute.", status_code=503)
    return Response(
        upstream.content,
        status_code=upstream.status_code,
        media_type=upstream.headers.get("content-type"),
    )


with gr.Blocks(title="Laya Trader") as demo:
    gr.Markdown("laya-trader")

# ssr_mode=False: on Spaces, server-side rendering puts a Node server in front that
# would answer every page itself.
demo.launch(server_name="0.0.0.0", server_port=7860, ssr_mode=False, prevent_thread_lock=True)
routes = demo.app.router.routes
routes[:] = [Route(path, proxy, methods=["GET", "POST"]) for path in PATHS] + [
    route for route in routes if getattr(route, "path", None) not in ("/", "/index.html")
]

os.environ.setdefault("LAYA_BACKEND", "torch")
sys.argv = [
    "laya_trader.py",
    "--port", str(INTERNAL_PORT),
    "--public",
    "--no-open",
    "--interval", "10",
    "--backtest-every-hours", "24",
    "--backtest-days", "30",
]  # fmt: skip

from laya_trader import main  # noqa: E402

main()
