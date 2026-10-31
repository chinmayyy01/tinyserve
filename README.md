# tinyserve

## Running on Modal (no local GPU)

The engine is CUDA only, so on a Mac run it on a Modal GPU:

```bash
pip install modal
modal token new
modal serve modal_app.py     # dev endpoint with hot reload
modal deploy modal_app.py    # persistent endpoint
```

- `TINYSERVE_MODEL` overrides the model (default `Qwen/Qwen3-1.7B`)
- `TINYSERVE_GPU` overrides the GPU (default `A10G`)
- Weights are cached in the `tinyserve-hf-cache` volume, so only the first
  cold start downloads them
- `flash-attn` is intentionally omitted (source build is slow/flaky); the
  Triton backend is used instead
- Modal web endpoints have a 150s request timeout (it redirects with a 303),
  so keep `max_tokens` bounded for long generations
