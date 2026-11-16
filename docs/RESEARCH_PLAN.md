# Benchmark Plan - Qwen-Image-2.1 on AMD Instinct MI300X (`gfx942`)

**Status:** Harness, `unittest`, and CPU PyTorch checks pass in CI; awaiting an **AMD Developer Cloud MI300X** VM to capture real hardware measurements.

---

## Architectural Alignment (`Qwen/Qwen-Image-2.1`)

* **Transformer (`QwenImage21Transformer2DModel`):** 32 layers, `32` attention heads × `128` head_dim (`4096` hidden), `patch_size=1`, `causal_condition=True`.
* **VAE (`AutoencoderKLQwenImage21`):** `16×` spatial compression, `64`-channel latent space (`in_channels=4` RGBA).

---

## Checks Included

| ID | What it runs | Command | Output |
| :--- | :--- | :--- | :--- |
| **EXP-01** | 32-head SDPA sweep (`unmasked` + `causal_condition` block mask) across `FLASH_ATTENTION`, `EFFICIENT_ATTENTION`, `MATH` | `PYTHONPATH=src python3 -m qwen21_infra.experiments --exp exp01` | Median/p95 `ms`, TFLOPS, peak allocated VRAM, fallback behavior |
| **EXP-02** | Quantization SNR (`BF16` vs. naive `INT8` vs. Walsh-Hadamard `ConvRot`) **separately from** `torch._int_mm` vs. `BF16` matmul latency | `PYTHONPATH=src python3 -m qwen21_infra.experiments --exp exp02` | `snr_db`, `cosine_similarity`, `_int_mm` kernel + dequant median ms vs. `BF16` |
| **EXP-03** | `16×` / `patch_size=1` VRAM scaling (`512px`–`2048px`) + live GPU latent/SDPA peak memory probe | `PYTHONPATH=src python3 -m qwen21_infra.experiments --exp exp03` | `[B, 64, H/16, W/16]` shapes, pre-calib. estimates, `max_memory_allocated` |
| **EXP-04** | `32`-head DiT block `torch.profiler` + multi-repeat ComfyUI `/prompt` timing | `PYTHONPATH=src python3 -m qwen21_infra.experiments --exp exp04` | Top PyTorch/HIP/Triton ops (`device_time_ms`), `rocm-smi`, median/p95 s |
