"""Run tinyserve on a Modal GPU (the engine needs CUDA; macOS does not have it).

    pip install modal
    modal token new
    modal serve modal_app.py     # dev endpoint with hot reload
    modal deploy modal_app.py    # persistent endpoint

Overrides:
    TINYSERVE_MODEL   HF model id        (default: Qwen/Qwen3-1.7B)
    TINYSERVE_GPU     Modal GPU spec     (default: A10G)
"""

import os
from pathlib import Path

import modal

APP_NAME = "tinyserve"
MODEL = os.environ.get("TINYSERVE_MODEL", "Qwen/Qwen3-1.7B")
GPU = os.environ.get("TINYSERVE_GPU", "A10G")

LOCAL_SOURCES = ["server"] + (
    ["tinykernels"] if (Path(__file__).parent / "tinykernels.py").exists() else []
)

app = modal.App(APP_NAME)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers",
        "safetensors",
        "huggingface_hub",
        "numpy",
        "fastapi[standard]",
        "uvicorn",
        "openai",
    )
    .add_local_python_source(*LOCAL_SOURCES)
)

hf_cache = modal.Volume.from_name("tinyserve-hf-cache", create_if_missing=True)


@app.cls(
    gpu=GPU,
    image=image,
    volumes={"/root/.cache/huggingface": hf_cache},
    timeout=60 * 60,
    startup_timeout=60 * 15,
)
class Tinyserve:
    @modal.enter()
    def load(self):
        import server

        cfg = server.Config(
            model=MODEL,
            attention_backend="triton",
            use_triton_kernels=True,
            use_cuda_graphs=True,
            enable_prefix_caching=True,
        )
        self.engine = server.Engine(cfg)
        self.web_app = server.build_app(server.AsyncEngine(self.engine, 10.0), MODEL)

    @modal.asgi_app()
    def web(self):
        return self.web_app
