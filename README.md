# qwen21-gpu-infra

[![CI](https://github.com/eagleblastmusic-lgtm/qwen21-gpu-infra/actions/workflows/ci.yml/badge.svg)](https://github.com/eagleblastmusic-lgtm/qwen21-gpu-infra/actions/workflows/ci.yml)

Scripts, PyTorch checks, and ComfyUI setup helpers for testing **`Comfy-Org/Qwen-Image-2.1`** (`QwenImage21Transformer2DModel`: 32 layers, `32` attention heads × `128` head_dim, `patch_size=1`, `causal_condition=True`; `AutoencoderKLQwenImage21`: `16×` spatial compression, `64`-channel latents) on **AMD Instinct MI300X (`ROCm / gfx942`)** in unquantized `full-bf16` mode.

---

## Why I put this together

Most `16–24 GB VRAM` GPUs only run `Qwen-Image-2.1` after quantizing weights to `INT8 ConvRot` or `W4A8`, because the unquantized `full-bf16` weights alone are roughly **`32.4 GB`** (`14.2 GB` DiT + `17.53 GB` `Qwen3-VL 8B` + `0.68 GB` VAE) before high-resolution activations.

Since **MI300X** provides **192 GB HBM3**, this suite prepares reproducible checks to evaluate `full-bf16` natively without CPU offload:

1. **`EXP-01` - SDPA & `causal_condition` Attention Mask Behavior:** Benchmarks PyTorch SDPA (`FLASH_ATTENTION`, `EFFICIENT_ATTENTION`, `MATH`) across multimodal sequence lengths with `32` heads × `128` dim under both unmasked attention and `Qwen-Image-2.1`-style prefix `causal_condition` block masks, reporting median/p95 ms.
2. **`EXP-02` - Quantization Fidelity vs. Native `torch._int_mm` Runtime:** Strictly separates **emulated numerical error** (`BF16` vs. naive `INT8` vs. Walsh-Hadamard rotated `ConvRot INT8` SNR/cosine drift) from **hardware kernel timing** (`BF16` matmul vs. native `torch._int_mm` `int8×int8→int32` + dequant scale).
3. **`EXP-03` - `16×` VAE / `patch_size=1` Memory Sweep (`512px` to `2048px`):** Computes exact latent shapes (`[1, 64, H/16, W/16]`) + pre-calibration heuristic activation estimates alongside a live GPU `max_memory_allocated` probe hook.
4. **`EXP-04` - Kernel Traces & ComfyUI Timing:** Captures `torch.profiler` ops for a `32`-head DiT attention + SwiGLU block, checks `triton` / `comfy-kitchen` / `rocm-smi`, and runs multi-repeat fixed-seed ComfyUI `/prompt` benchmarks (`median_ms`, `p95_ms`).

*(Protocol notes: [docs/RESEARCH_PLAN.md](docs/RESEARCH_PLAN.md).)*

---

## Weight Profiles & Pre-Calibration Memory Estimates

| Profile | Format | Raw Weights | Pre-Calib. `2048×2048` (`B=1`) | Hardware Note |
| :--- | :--- | :--- | :--- | :--- |
| `full-bf16` | Unquantized `BF16` | `~32.4 GB` | `~48.35 GB` *(heuristic)* | Exceeds 24/48 GB cards; fits `MI300X` without tiling |
| `full-int8` | `INT8 ConvRot` + PE | `~34.0 GB` | `~49.95 GB` *(heuristic)* | Base + turbo DiT + 9B enhancer |
| `lite-int8` | `INT8 ConvRot` | `~17.3 GB` | `~33.25 GB` *(heuristic)* | Base DiT + `Qwen3-VL` + VAE |
| `w4a8` | `W4A8` + `INT8` | - | - | Encoder quantization comparison |
| `turbo-lite-int8` | `INT8` (Turbo) | - | - | Distilled DiT variant |

---

## Files in this Repo

| File | What it does |
| :--- | :--- |
| [`src/qwen21_infra/experiments.py`](src/qwen21_infra/experiments.py) | `EXP-01` (32-head SDPA + `causal_condition` mask), `EXP-02` (SNR + native `torch._int_mm`), `EXP-03` (`16×` VRAM sweep), `EXP-04` (`torch.profiler`) |
| [`src/qwen21_infra/rocm_bench.py`](src/qwen21_infra/rocm_bench.py) | `torch.version.hip`, `gcnArchName`, SDPA/Triton/`comfy-kitchen`, `rocm-smi` parser, median/p95 stats, multi-repeat ComfyUI `/prompt` telemetry |
| [`src/qwen21_infra/manifest.py`](src/qwen21_infra/manifest.py) | HuggingFace manifest (`models.json`), `16×`/`64`-ch/`patch_size=1` VRAM calculator, `.tmp -> replace` downloader |
| [`src/qwen21_infra/workflows.py`](src/qwen21_infra/workflows.py) | Profile workflow rewriting, fixed `KSampler` seed setup, graph link integrity validator |
| [`scripts/rocm/run_mi300x_suite.sh`](scripts/rocm/run_mi300x_suite.sh) | Runs env check + `EXP-01..EXP-04` and emits JSON/Markdown reports |
| [`scripts/rocm/setup_mi300x_bf16.sh`](scripts/rocm/setup_mi300x_bf16.sh) | Reproducible ComfyUI setup pinned to `COMFYUI_REF` (`v0.45.0` default) preserving ROCm PyTorch |
| [`tests/`](tests) | `unittest` suite covering manifests, `16×` shapes, graph invariants, `causal_condition` masks, `_int_mm`, and mock telemetry |

---

## Current Status

* **CI (`GitHub Actions`):** Static checks, offline tests, and CPU PyTorch smoke tests (`EXP-01..EXP-04`) pass on every push.
* **MI300X (`gfx942`) Hardware Runs:** **Not run yet** - waiting for AMD Developer Cloud access to record actual `gfx942` latency medians, HBM peaks, `torch._int_mm` support, and end-to-end ComfyUI samples.

---

## Quickstart on MI300X (`ROCm`)

```bash
git clone https://github.com/eagleblastmusic-lgtm/qwen21-gpu-infra.git
cd qwen21-gpu-infra

# 1. Check ROCm stack & execute PyTorch checks (EXP-01..EXP-04)
bash scripts/rocm/run_mi300x_suite.sh

# 2. Optional: install pinned ComfyUI (COMFYUI_REF=v0.45.0) + full-bf16 weights
bash scripts/rocm/setup_mi300x_bf16.sh
```

## License

MIT - see [LICENSE](LICENSE).
