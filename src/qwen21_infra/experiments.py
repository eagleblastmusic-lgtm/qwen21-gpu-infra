"""ROCm MI300X experiment harness (EXP-01..EXP-04) aligned with Qwen-Image-2.1 config.

Architecture constants verified against QwenImage21Transformer2DModel and AutoencoderKLQwenImage21:
  - DiT transformer: 32 layers, 32 attention heads x 128 head_dim (4096 hidden), patch_size=1, causal_condition=True
  - VAE: 16x spatial compression, 64 latent channels

EXP-02 strictly separates:
  1. Emulated quantization numerical fidelity (`cosine_similarity`, `rel_l2_error`, `snr_db`), and
  2. Real hardware INT8 matrix multiplication (`torch._int_mm`) vs. native BF16 matmul latency.
"""

from __future__ import annotations

import argparse
import contextlib
import math
import time
import json
from pathlib import Path
from typing import Any

from .manifest import VAE_LATENT_CHANNELS, VAE_SPATIAL_COMPRESSION, estimate_memory_budget
from .rocm_bench import probe_environment, summarize_timings_ms

EXPERIMENT_CATALOG: list[dict[str, str]] = [
    {
        "id": "EXP-01",
        "title": "SDPA Scaling Sweep (32 heads x 128 dim, unmasked & causal_condition block mask)",
        "hypothesis": "Check whether ROCm SDPA retains Flash/Efficient/AOTriton or falls back to Math under block-causal masks.",
    },
    {
        "id": "EXP-02",
        "title": "BF16 vs. Real INT8 (`torch._int_mm`) Runtime + Hadamard ConvRot Numerical Drift",
        "hypothesis": "Separate quantization SNR benefits (Walsh-Hadamard) from true integer core speed (`_int_mm` vs BF16).",
    },
    {
        "id": "EXP-03",
        "title": "VAE 16x / patch_size=1 Resolution Memory Cliff Model + Empirical Peak VRAM Probe",
        "hypothesis": "Map activation cliffs across 512px-2048px and calibrate heuristic estimates against `max_memory_allocated`.",
    },
    {
        "id": "EXP-04",
        "title": "DiT Block `torch.profiler` Kernel Breakdown (32 heads x 128 dim)",
        "hypothesis": "Inspect dominant HIP / AOTriton / Triton vs. PyTorch fallback operators on gfx942.",
    },
]


def _sync_if_gpu(torch_mod: Any, device: Any) -> None:
    if device.type == "cuda":
        torch_mod.cuda.synchronize(device)


def _select_device(torch_mod: Any, force_cpu: bool = False) -> Any:
    if not force_cpu and torch_mod.cuda.is_available():
        return torch_mod.device("cuda:0")
    return torch_mod.device("cpu")


def build_causal_condition_mask(seq_len: int, condition_tokens: int, device: Any) -> Any:
    """Build block-causal prefix mask matching Qwen-Image-2.1 `causal_condition=True`.

    Condition/prefix tokens `[0:cond]` cannot attend to noisy latent tokens `[cond:seq_len]`,
    keeping prefix activations timestep-invariant for caching while noisy tokens attend everywhere.
    Returns boolean mask of shape `(1, 1, seq_len, seq_len)` where True permits attention.
    """
    import torch  # type: ignore[import-not-found]

    cond = min(max(0, condition_tokens), seq_len)
    if cond == 0 or cond == seq_len:
        return torch.ones((1, 1, seq_len, seq_len), dtype=torch.bool, device=device)

    row_idx = torch.arange(seq_len, device=device).unsqueeze(1)
    col_idx = torch.arange(seq_len, device=device).unsqueeze(0)
    allow = (row_idx >= cond) | (col_idx < cond)
    return allow.view(1, 1, seq_len, seq_len)


def walsh_hadamard_rotate(x: Any, block_size: int = 64) -> Any:
    """Apply an orthonormal normalized Walsh-Hadamard block rotation along the last dimension."""
    import torch  # type: ignore[import-not-found]

    if block_size <= 0 or (block_size & (block_size - 1)) != 0:
        raise ValueError("block_size must be a positive power of two.")
    last_dim = x.shape[-1]
    if last_dim % block_size != 0:
        raise ValueError(f"Last dimension {last_dim} must be divisible by block_size={block_size}.")

    orig_shape = x.shape
    h = x.reshape(-1, last_dim // block_size, block_size)
    step = 1
    while step < block_size:
        h = h.reshape(-1, last_dim // block_size, block_size // (2 * step), 2, step)
        a = h[..., 0, :]
        b = h[..., 1, :]
        h = torch.stack((a + b, a - b), dim=-2)
        step *= 2
    scale = 1.0 / math.sqrt(block_size)
    return h.reshape(orig_shape) * scale


def quantize_symmetric_int8(tensor: Any, axis: int = -1) -> tuple[Any, Any]:
    """Quantize tensor to real `torch.int8` and return `(q_int8, scale)`."""
    import torch  # type: ignore[import-not-found]

    f = tensor.float()
    max_abs = f.abs().amax(dim=axis, keepdim=True).clamp(min=1e-8)
    scale = max_abs / 127.0
    q_int8 = torch.clamp(torch.round(f / scale), -127, 127).to(torch.int8)
    return q_int8, scale


def dequantize_symmetric_int8(q_int8: Any, scale: Any, dtype: Any) -> Any:
    """Restore quantized `torch.int8` + scale into floating-point tensor for fidelity analysis."""
    return (q_int8.float() * scale).to(dtype)


def quantize_symmetric_int8_fake(tensor: Any, axis: int = -1) -> Any:
    """Emulated INT8 quantize + dequantize preserving input dtype (for SNR error studies only)."""
    q_int8, scale = quantize_symmetric_int8(tensor, axis=axis)
    return dequantize_symmetric_int8(q_int8, scale, tensor.dtype)


def compute_error_metrics(ref: Any, approx: Any) -> dict[str, float]:
    """Compute cosine similarity, relative L2 error, SNR (dB), and max abs error between two tensors."""
    import torch  # type: ignore[import-not-found]

    r = ref.float().reshape(-1)
    a = approx.float().reshape(-1)
    diff = r - a
    r_norm = torch.linalg.vector_norm(r).item()
    d_norm = torch.linalg.vector_norm(diff).item()
    cos = torch.nn.functional.cosine_similarity(r.unsqueeze(0), a.unsqueeze(0)).item()
    rel_l2 = d_norm / max(r_norm, 1e-12)
    snr_db = 20.0 * math.log10(max(r_norm, 1e-12) / max(d_norm, 1e-12))
    return {
        "cosine_similarity": round(cos, 6),
        "rel_l2_error": round(rel_l2, 6),
        "snr_db": round(snr_db, 2),
        "max_abs_error": round(diff.abs().max().item(), 6),
    }


def _sdpa_context(torch_mod: Any, backend_name: str) -> Any:
    attn_mod = getattr(torch_mod.nn, "attention", None)
    sdp_backend_enum = getattr(attn_mod, "SDPBackend", None) if attn_mod else None
    sdpa_kernel_fn = getattr(attn_mod, "sdpa_kernel", None) if attn_mod else None
    if sdp_backend_enum is not None and sdpa_kernel_fn is not None:
        enum_map = {
            "FLASH_ATTENTION": sdp_backend_enum.FLASH_ATTENTION,
            "EFFICIENT_ATTENTION": sdp_backend_enum.EFFICIENT_ATTENTION,
            "MATH": sdp_backend_enum.MATH,
        }
        return sdpa_kernel_fn(enum_map[backend_name])
    return contextlib.nullcontext()


def run_sdpa_scaling_sweep(
    seq_lengths: tuple[int, ...] = (1024, 2048, 4096),
    num_heads: int = 32,
    head_dim: int = 128,
    condition_tokens: int = 512,
    include_causal_condition_mask: bool = True,
    backends: tuple[str, ...] = ("FLASH_ATTENTION", "EFFICIENT_ATTENTION", "MATH"),
    iters: int = 5,
    warmup: int = 2,
    force_cpu: bool = False,
) -> dict[str, Any]:
    """Benchmark Qwen-Image-2.1-shaped SDPA (32 heads x 128 dim) across unmasked and causal_condition modes."""
    try:
        import torch  # type: ignore[import-not-found]
    except ImportError:
        return {"experiment": "EXP-01", "status": "SKIPPED_NO_TORCH", "rows": []}

    device = _select_device(torch, force_cpu=force_cpu)
    dtype = torch.bfloat16
    mask_modes = ("unmasked", "causal_condition") if include_causal_condition_mask else ("unmasked",)
    rows: list[dict[str, Any]] = []

    for seq_len in seq_lengths:
        q = torch.randn((1, num_heads, seq_len, head_dim), device=device, dtype=dtype)
        k = torch.randn((1, num_heads, seq_len, head_dim), device=device, dtype=dtype)
        v = torch.randn((1, num_heads, seq_len, head_dim), device=device, dtype=dtype)
        cond_len = min(condition_tokens, seq_len // 2)

        for mask_mode in mask_modes:
            attn_mask = build_causal_condition_mask(seq_len, cond_len, device) if mask_mode == "causal_condition" else None
            for backend_name in backends:
                if device.type == "cpu" and backend_name != "MATH":
                    rows.append(
                        {
                            "seq_len": seq_len,
                            "num_heads": num_heads,
                            "head_dim": head_dim,
                            "mask_mode": mask_mode,
                            "backend": backend_name,
                            "device": device.type,
                            "status": "SKIPPED_CPU_NON_MATH",
                        }
                    )
                    continue
                try:
                    if device.type == "cuda":
                        torch.cuda.reset_peak_memory_stats(device)
                    samples_ms: list[float] = []
                    with torch.no_grad(), _sdpa_context(torch, backend_name):
                        for _ in range(warmup):
                            _ = torch.nn.functional.scaled_dot_product_attention(
                                q, k, v, attn_mask=attn_mask, is_causal=False
                            )
                        _sync_if_gpu(torch, device)
                        for _ in range(iters):
                            t0 = time.perf_counter()
                            _ = torch.nn.functional.scaled_dot_product_attention(
                                q, k, v, attn_mask=attn_mask, is_causal=False
                            )
                            _sync_if_gpu(torch, device)
                            samples_ms.append((time.perf_counter() - t0) * 1000.0)
                    peak_mb = (
                        round(torch.cuda.max_memory_allocated(device) / (1024**2), 2)
                        if device.type == "cuda"
                        else None
                    )
                    stats = summarize_timings_ms(samples_ms)
                    med_s = (stats["median_ms"] or 1.0) / 1000.0
                    flops = 4.0 * num_heads * (seq_len**2) * head_dim
                    rows.append(
                        {
                            "seq_len": seq_len,
                            "num_heads": num_heads,
                            "head_dim": head_dim,
                            "mask_mode": mask_mode,
                            "backend": backend_name,
                            "device": device.type,
                            "status": "OK",
                            **stats,
                            "approx_tflops": round((flops / med_s) / 1e12, 2),
                            "peak_allocated_mb": peak_mb,
                        }
                    )
                except RuntimeError as exc:
                    rows.append(
                        {
                            "seq_len": seq_len,
                            "num_heads": num_heads,
                            "head_dim": head_dim,
                            "mask_mode": mask_mode,
                            "backend": backend_name,
                            "device": device.type,
                            "status": "UNSUPPORTED",
                            "reason": str(exc).splitlines()[0][:180],
                        }
                    )

    return {"experiment": "EXP-01", "status": "OK", "device": str(device), "rows": rows}


def run_quantization_drift_study(
    batch_tokens: int = 256,
    hidden_dim: int = 4096,
    block_size: int = 64,
    outlier_scale: float = 25.0,
    seed: int = 42,
    repeats: int = 5,
    warmup: int = 2,
    force_cpu: bool = False,
) -> dict[str, Any]:
    """Measure quantization SNR drift AND native `torch._int_mm` vs. BF16 GEMM runtime separately."""
    try:
        import torch  # type: ignore[import-not-found]
    except ImportError:
        return {"experiment": "EXP-02", "status": "SKIPPED_NO_TORCH"}

    device = _select_device(torch, force_cpu=force_cpu)
    gen = torch.Generator(device="cpu").manual_seed(seed)

    x_cpu = torch.randn((batch_tokens, hidden_dim), generator=gen, dtype=torch.float32)
    x_cpu[:, : max(1, hidden_dim // 64)] *= outlier_scale
    w_cpu = torch.randn((hidden_dim, hidden_dim), generator=gen, dtype=torch.float32) / math.sqrt(hidden_dim)

    x_fp32 = x_cpu.to(device)
    w_fp32 = w_cpu.to(device)
    ref_out = x_fp32 @ w_fp32

    x_bf16 = x_fp32.bfloat16()
    w_bf16 = w_fp32.bfloat16()
    out_bf16 = x_bf16 @ w_bf16

    # Emulated quantization SNR analysis (clearly marked as fidelity-only, NOT kernel timing).
    out_naive_fake = quantize_symmetric_int8_fake(x_bf16, axis=-1) @ quantize_symmetric_int8_fake(w_bf16, axis=0)
    x_rot_q, scale_x_rot = quantize_symmetric_int8(walsh_hadamard_rotate(x_bf16, block_size=block_size), axis=-1)
    w_rot_t_q, scale_w_rot_t = quantize_symmetric_int8(
        walsh_hadamard_rotate(w_bf16.transpose(0, 1), block_size=block_size),
        axis=-1,
    )
    out_convrot_fake = (
        dequantize_symmetric_int8(x_rot_q, scale_x_rot, torch.bfloat16)
        @ dequantize_symmetric_int8(w_rot_t_q, scale_w_rot_t, torch.bfloat16).transpose(0, 1)
    )

    # Runtime benchmark comparing native BF16 matmul against real torch._int_mm (int8 x int8 -> int32).
    bf16_samples: list[float] = []
    for _ in range(warmup):
        _ = x_bf16 @ w_bf16
    _sync_if_gpu(torch, device)
    for _ in range(repeats):
        t0 = time.perf_counter()
        _ = x_bf16 @ w_bf16
        _sync_if_gpu(torch, device)
        bf16_samples.append((time.perf_counter() - t0) * 1000.0)

    # Row-major RHS stored as (N, K) and transposed gives column-major (K, N) expected by cuBLASLt/hipBLASLt.
    w_rot_col = w_rot_t_q.contiguous().t()
    scale_row = scale_x_rot.float()
    scale_col = scale_w_rot_t.float().t()

    native_int8: dict[str, Any]
    try:
        for _ in range(warmup):
            acc = torch._int_mm(x_rot_q, w_rot_col)
            _ = (acc.float() * (scale_row * scale_col)).bfloat16()
        _sync_if_gpu(torch, device)

        kernel_samples: list[float] = []
        e2e_samples: list[float] = []
        last_int_out = None
        for _ in range(repeats):
            t_k = time.perf_counter()
            acc = torch._int_mm(x_rot_q, w_rot_col)
            _sync_if_gpu(torch, device)
            kernel_samples.append((time.perf_counter() - t_k) * 1000.0)

            t_e = time.perf_counter()
            acc = torch._int_mm(x_rot_q, w_rot_col)
            last_int_out = (acc.float() * (scale_row * scale_col)).bfloat16()
            _sync_if_gpu(torch, device)
            e2e_samples.append((time.perf_counter() - t_e) * 1000.0)

        native_int8 = {
            "status": "OK",
            "kernel_only": summarize_timings_ms(kernel_samples),
            "with_scale_dequant": summarize_timings_ms(e2e_samples),
            "fidelity": compute_error_metrics(ref_out, last_int_out),
        }
    except (RuntimeError, NotImplementedError, AttributeError) as exc:
        native_int8 = {"status": "UNSUPPORTED", "reason": str(exc).splitlines()[0][:180]}

    return {
        "experiment": "EXP-02",
        "status": "OK",
        "device": str(device),
        "shape": [batch_tokens, hidden_dim],
        "outlier_scale": outlier_scale,
        "fidelity_comparison": {
            "bf16": compute_error_metrics(ref_out, out_bf16),
            "naive_int8_emulated": compute_error_metrics(ref_out, out_naive_fake),
            "hadamard_convrot_int8_emulated": compute_error_metrics(ref_out, out_convrot_fake),
        },
        "runtime_comparison": {
            "bf16_matmul": summarize_timings_ms(bf16_samples),
            "native_int_mm": native_int8,
        },
    }


def run_resolution_memory_model(force_cpu: bool = False) -> dict[str, Any]:
    """Generate EXP-03 VAE 16x resolution sweep and optionally probe live latent+attention VRAM."""
    rows = [
        estimate_memory_budget(profile, width=dim, height=dim, batch_size=1)
        for profile in ("full-bf16", "full-int8", "lite-int8")
        for dim in (512, 1024, 1536, 2048)
    ]
    empirical_probe: list[dict[str, Any]] = []
    try:
        import torch  # type: ignore[import-not-found]

        if not force_cpu and torch.cuda.is_available():
            device = torch.device("cuda:0")
            for dim in (512, 1024, 1536, 2048):
                lat_h = dim // VAE_SPATIAL_COMPRESSION
                tokens = lat_h * lat_h
                torch.cuda.reset_peak_memory_stats(device)
                with torch.no_grad():
                    latent = torch.randn((1, VAE_LATENT_CHANNELS, lat_h, lat_h), device=device, dtype=torch.bfloat16)
                    qkv = torch.randn((3, 1, 32, tokens, 128), device=device, dtype=torch.bfloat16)
                    out = torch.nn.functional.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2])
                    torch.cuda.synchronize(device)
                empirical_probe.append(
                    {
                        "resolution": f"{dim}x{dim}",
                        "latent_shape": list(latent.shape),
                        "dit_tokens": tokens,
                        "sdpa_peak_vram_mb": round(torch.cuda.max_memory_allocated(device) / (1024**2), 2),
                        "out_shape": list(out.shape),
                    }
                )
    except ImportError:
        pass

    return {
        "experiment": "EXP-03",
        "status": "OK",
        "model_sweep": rows,
        "empirical_sdpa_peak_probe": empirical_probe,
    }


def profile_dit_block(
    seq_len: int = 1024,
    hidden_dim: int = 4096,
    num_heads: int = 32,
    top_k: int = 8,
    force_cpu: bool = False,
) -> dict[str, Any]:
    """Profile a single Qwen-Image-2.1-sized DiT Attention + SwiGLU block with `torch.profiler`."""
    try:
        import torch  # type: ignore[import-not-found]
    except ImportError:
        return {"experiment": "EXP-04", "status": "SKIPPED_NO_TORCH", "top_ops": []}

    device = _select_device(torch, force_cpu=force_cpu)
    head_dim = hidden_dim // num_heads
    dtype = torch.bfloat16

    x = torch.randn((1, seq_len, hidden_dim), device=device, dtype=dtype)
    qkv_proj = torch.nn.Linear(hidden_dim, hidden_dim * 3, bias=False, device=device, dtype=dtype)
    w_gate = torch.nn.Linear(hidden_dim, hidden_dim * 2, bias=False, device=device, dtype=dtype)
    w_out = torch.nn.Linear(hidden_dim, hidden_dim, bias=False, device=device, dtype=dtype)

    def _step() -> Any:
        qkv = qkv_proj(x).view(1, seq_len, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
        attn = torch.nn.functional.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2])
        attn = attn.transpose(1, 2).reshape(1, seq_len, hidden_dim)
        g, u = w_gate(attn).chunk(2, dim=-1)
        return w_out(torch.nn.functional.silu(g) * u)

    with torch.no_grad():
        _ = _step()
        _sync_if_gpu(torch, device)
        activities = [torch.profiler.ProfilerActivity.CPU]
        sort_key = "cpu_time_total"
        if device.type == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)
            sort_key = "cuda_time_total"

        with torch.profiler.profile(activities=activities) as prof:
            _ = _step()
            _sync_if_gpu(torch, device)

    events = sorted(prof.key_averages(), key=lambda evt: getattr(evt, sort_key, 0.0), reverse=True)[:top_k]
    top_ops = [
        {
            "name": evt.key,
            "count": evt.count,
            "cpu_time_ms": round(evt.cpu_time_total / 1000.0, 3),
            "device_time_ms": round(getattr(evt, "device_time_total", getattr(evt, "cuda_time_total", 0.0)) / 1000.0, 3),
        }
        for evt in events
    ]
    return {
        "experiment": "EXP-04",
        "status": "OK",
        "device": str(device),
        "num_heads": num_heads,
        "hidden_dim": hidden_dim,
        "top_ops": top_ops,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ROCm/MI300X experimental benchmark suite (EXP-01..04).")
    parser.add_argument(
        "--exp",
        choices=("all", "exp01", "exp02", "exp03", "exp04", "list"),
        default="list",
        help="Experiment suite to execute (default: list).",
    )
    parser.add_argument("--cpu", action="store_true", help="Force CPU execution for smoke checks.")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output file path.")
    args = parser.parse_args()

    if args.exp == "list":
        print(json.dumps(EXPERIMENT_CATALOG, indent=2))
        return

    payload: dict[str, Any] = {"environment": probe_environment()}
    use_cpu = args.cpu or not payload["environment"]["gpu_available"]
    if args.exp in ("all", "exp01"):
        seqs = (128, 256) if use_cpu else (1024, 2048, 4096, 8192)
        heads = 4 if use_cpu else 32
        payload["exp01"] = run_sdpa_scaling_sweep(seq_lengths=seqs, num_heads=heads, force_cpu=args.cpu)
    if args.exp in ("all", "exp02"):
        dim = 256 if use_cpu else 4096
        payload["exp02"] = run_quantization_drift_study(hidden_dim=dim, force_cpu=args.cpu)
    if args.exp in ("all", "exp03"):
        payload["exp03"] = run_resolution_memory_model(force_cpu=args.cpu)
    if args.exp in ("all", "exp04"):
        seq = 128 if use_cpu else 2048
        hdim = 256 if use_cpu else 4096
        heads = 4 if use_cpu else 32
        payload["exp04"] = profile_dit_block(seq_len=seq, hidden_dim=hdim, num_heads=heads, force_cpu=args.cpu)

    text = json.dumps(payload, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
