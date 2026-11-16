"""ROCm/HIP environment probe, rocm-smi parser, median timing helpers, and ComfyUI telemetry."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

ROCM_RELEVANT_ENV_VARS = (
    "HIP_VISIBLE_DEVICES",
    "ROCR_VISIBLE_DEVICES",
    "HSA_OVERRIDE_GFX_VERSION",
    "TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL",
    "MIOPEN_FIND_MODE",
    "PYTORCH_HIP_ALLOC_CONF",
)


def summarize_timings_ms(samples_ms: list[float]) -> dict[str, Any]:
    """Compute robust latency distribution metrics (median, p95, min, max) in milliseconds."""
    if not samples_ms:
        return {"repeats": 0, "median_ms": None, "p95_ms": None, "min_ms": None, "max_ms": None}
    ordered = sorted(samples_ms)
    idx_95 = min(len(ordered) - 1, max(0, math.ceil(0.95 * len(ordered)) - 1))
    return {
        "repeats": len(ordered),
        "median_ms": round(statistics.median(ordered), 3),
        "p95_ms": round(ordered[idx_95], 3),
        "min_ms": round(ordered[0], 3),
        "max_ms": round(ordered[-1], 3),
    }


def parse_rocm_smi_json(raw_json: str) -> list[dict[str, Any]]:
    """Parse normalized GPU metrics from `rocm-smi --showtemp --showuse --showmemuse --showpower --json`."""
    data = json.loads(raw_json)
    devices: list[dict[str, Any]] = []
    if not isinstance(data, dict):
        return devices
    for card_key in sorted(data):
        if not str(card_key).startswith("card"):
            continue
        card = data[card_key]
        if not isinstance(card, dict):
            continue
        devices.append(
            {
                "card": str(card_key),
                "gpu_busy_percent": _to_float_or_none(card.get("GPU use (%)")),
                "vram_used_percent": _to_float_or_none(card.get("GPU memory use (%)")),
                "temperature_edge_c": _to_float_or_none(
                    card.get("Temperature (Sensor edge) (C)") or card.get("Temperature (Sensor junction) (C)")
                ),
                "power_w": _to_float_or_none(card.get("Average Graphics Package Power (W)")),
            }
        )
    return devices


def _to_float_or_none(val: Any) -> float | None:
    if val is None:
        return None
    try:
        return float(str(val).strip())
    except (TypeError, ValueError):
        return None


def collect_rocm_smi_telemetry() -> dict[str, Any]:
    """Collect optional `rocm-smi` telemetry when running on an AMD ROCm host."""
    rocm_smi = shutil.which("rocm-smi")
    if rocm_smi is None:
        return {"available": False, "devices": []}
    try:
        proc = subprocess.run(
            [rocm_smi, "--showtemp", "--showuse", "--showmemuse", "--showpower", "--json"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if proc.returncode != 0 or not proc.stdout.strip():
            return {"available": True, "error": proc.stderr.strip() or f"exit {proc.returncode}", "devices": []}
        return {"available": True, "devices": parse_rocm_smi_json(proc.stdout)}
    except Exception as exc:
        return {"available": True, "error": str(exc), "devices": []}


def probe_environment() -> dict[str, Any]:
    """Inspect PyTorch ROCm/CUDA backend, GPU topology, BF16/SDPA, Triton, and comfy-kitchen."""
    report: dict[str, Any] = {
        "python": sys.version.split()[0],
        "torch_installed": False,
        "backend": "none",
        "hip_version": None,
        "cuda_version": None,
        "gpu_available": False,
        "bf16_supported": None,
        "devices": [],
        "sdpa_backends": {},
        "triton_version": None,
        "comfy_kitchen_installed": importlib.util.find_spec("comfy_kitchen") is not None,
        "rocm_env": {k: v for k in ROCM_RELEVANT_ENV_VARS if (v := os.environ.get(k)) is not None},
        "rocm_smi": collect_rocm_smi_telemetry(),
    }
    try:
        import torch  # type: ignore[import-not-found]

        report["torch_installed"] = True
        report["torch_version"] = torch.__version__
        report["hip_version"] = getattr(torch.version, "hip", None)
        report["cuda_version"] = getattr(torch.version, "cuda", None)
        if report["hip_version"]:
            report["backend"] = "rocm"
        elif report["cuda_version"]:
            report["backend"] = "cuda"
        else:
            report["backend"] = "cpu"

        report["gpu_available"] = bool(torch.cuda.is_available())
        if hasattr(torch.cuda, "is_bf16_supported") and report["gpu_available"]:
            report["bf16_supported"] = bool(torch.cuda.is_bf16_supported())

        sdp = getattr(torch.backends, "cuda", None)
        if sdp is not None:
            for attr in ("flash_sdp_enabled", "mem_efficient_sdp_enabled", "math_sdp_enabled"):
                fn = getattr(sdp, attr, None)
                if callable(fn):
                    try:
                        report["sdpa_backends"][attr] = bool(fn())
                    except Exception:
                        pass

        if report["gpu_available"]:
            for idx in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(idx)
                report["devices"].append(
                    {
                        "index": idx,
                        "name": props.name,
                        "gcn_arch_name": getattr(props, "gcnArchName", None),
                        "total_vram_gib": round(props.total_memory / (1024**3), 2),
                    }
                )
    except ImportError:
        pass

    try:
        import triton  # type: ignore[import-not-found]

        report["triton_version"] = getattr(triton, "__version__", "unknown")
    except ImportError:
        pass

    return report


def benchmark_bf16_ops(matrix_size: int = 4096, warmup: int = 3, iters: int = 10) -> dict[str, Any]:
    """Run a BF16 GEMM micro-benchmark on GPU reporting median/p95 latency and TFLOPS."""
    try:
        import torch  # type: ignore[import-not-found]
    except ImportError:
        return {"status": "SKIPPED_NO_TORCH"}

    if not torch.cuda.is_available():
        return {"status": "SKIPPED_NO_GPU"}

    device = torch.device("cuda:0")
    a = torch.randn((matrix_size, matrix_size), device=device, dtype=torch.bfloat16)
    b = torch.randn((matrix_size, matrix_size), device=device, dtype=torch.bfloat16)
    for _ in range(warmup):
        _ = a @ b
    torch.cuda.synchronize(device)

    samples_ms: list[float] = []
    for _ in range(iters):
        t0 = time.perf_counter()
        _ = a @ b
        torch.cuda.synchronize(device)
        samples_ms.append((time.perf_counter() - t0) * 1000.0)

    stats = summarize_timings_ms(samples_ms)
    flops = 2.0 * (matrix_size**3)
    median_s = (stats["median_ms"] or 1.0) / 1000.0
    return {
        "status": "OK",
        "op": f"bf16_matmul_{matrix_size}",
        **stats,
        "median_tflops": round((flops / median_s) / 1e12, 2),
    }


def fetch_comfy_system_stats(comfy_url: str, timeout_s: float = 5.0) -> dict[str, Any]:
    """Query ComfyUI /system_stats to verify active device VRAM and runtime flags."""
    base = comfy_url.rstrip("/")
    with urllib.request.urlopen(f"{base}/system_stats", timeout=timeout_s) as resp:
        return json.loads(resp.read().decode("utf-8"))


def run_comfy_prompt_benchmark(
    comfy_url: str,
    prompt_json_path: Path,
    timeout_s: float = 300.0,
    poll_interval_s: float = 0.5,
    repeats: int = 1,
) -> dict[str, Any]:
    """Submit a ComfyUI API prompt JSON N times and record median/p95 latency + VRAM snapshots."""
    base = comfy_url.rstrip("/")
    prompt_payload = json.loads(prompt_json_path.read_text(encoding="utf-8"))
    if "prompt" not in prompt_payload:
        prompt_payload = {"prompt": prompt_payload}

    body = json.dumps(prompt_payload).encode("utf-8")
    runs_info: list[dict[str, Any]] = []
    latencies_ms: list[float] = []

    for run_idx in range(max(1, repeats)):
        stats_before = fetch_comfy_system_stats(base)
        req = urllib.request.Request(
            f"{base}/prompt",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        started = time.perf_counter()
        with urllib.request.urlopen(req, timeout=30) as resp:
            queued = json.loads(resp.read().decode("utf-8"))
        prompt_id = queued["prompt_id"]

        deadline = time.perf_counter() + timeout_s
        finished_entry: dict[str, Any] | None = None
        while time.perf_counter() < deadline:
            with urllib.request.urlopen(f"{base}/history/{prompt_id}", timeout=10) as resp:
                history = json.loads(resp.read().decode("utf-8"))
            if prompt_id in history:
                finished_entry = history[prompt_id]
                break
            time.sleep(poll_interval_s)

        if finished_entry is None:
            raise TimeoutError(f"Prompt {prompt_id} did not finish within {timeout_s}s.")

        elapsed_s = time.perf_counter() - started
        stats_after = fetch_comfy_system_stats(base)
        latencies_ms.append(elapsed_s * 1000.0)
        runs_info.append(
            {
                "run_index": run_idx,
                "prompt_id": prompt_id,
                "elapsed_seconds": round(elapsed_s, 3),
                "status": finished_entry.get("status", {}),
                "devices_before": stats_before.get("devices", []),
                "devices_after": stats_after.get("devices", []),
            }
        )

    last = runs_info[-1]
    return {
        "prompt_id": last["prompt_id"],
        "elapsed_seconds": last["elapsed_seconds"],
        "timing_summary": summarize_timings_ms(latencies_ms),
        "status": last["status"],
        "devices_before": last["devices_before"],
        "devices_after": last["devices_after"],
        "runs": runs_info,
    }


def format_markdown_report(report: dict[str, Any]) -> str:
    """Format environment probe and benchmark dictionaries as a concise Markdown summary."""
    lines = ["### ROCm / ComfyUI Evaluation Report", ""]
    env = report.get("environment")
    if isinstance(env, dict):
        lines.extend(
            [
                "| Metric | Value |",
                "| :--- | :--- |",
                f"| Python | `{env.get('python')}` |",
                f"| PyTorch / Backend | `{env.get('torch_version', 'not installed')}` (`{env.get('backend')}`) |",
                f"| HIP / CUDA Version | `{env.get('hip_version')}` / `{env.get('cuda_version')}` |",
                f"| BF16 Supported | `{env.get('bf16_supported')}` |",
                f"| Triton / `comfy-kitchen` | `{env.get('triton_version')}` / `{env.get('comfy_kitchen_installed')}` |",
                f"| Devices | `{len(env.get('devices', []))}` detected |",
            ]
        )
        for dev in env.get("devices", []):
            lines.append(
                f"| GPU `{dev.get('index')}` | `{dev.get('name')}` (`{dev.get('gcn_arch_name')}`, `{dev.get('total_vram_gib')} GiB`) |"
            )
    bench = report.get("benchmark")
    if isinstance(bench, dict):
        summary = bench.get("timing_summary") or {}
        lines.extend(
            [
                "",
                f"* **ComfyUI Prompt:** `{bench.get('prompt_id')}` (last `{bench.get('elapsed_seconds')} s`, median `{summary.get('median_ms')} ms` over `{summary.get('repeats')}` run(s))",
            ]
        )
    op_bench = report.get("op_benchmark")
    if isinstance(op_bench, dict):
        lines.extend(
            [
                "",
                f"* **BF16 GEMM Sanity:** `{op_bench.get('status')}` (median `{op_bench.get('median_tflops', '-')}` TFLOPS)",
            ]
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="AMD ROCm & ComfyUI Qwen-Image-2.1 benchmark helper.")
    parser.add_argument("--check-env", action="store_true", help="Print local PyTorch/ROCm/Triton diagnostics.")
    parser.add_argument("--run-op-bench", action="store_true", help="Run BF16 GEMM sanity micro-benchmark on GPU.")
    parser.add_argument("--matrix-size", type=int, default=4096, help="Square matrix dimension for GEMM sanity check.")
    parser.add_argument("--comfy-url", default=None, help="ComfyUI base URL, e.g. http://127.0.0.1:8188")
    parser.add_argument("--prompt-json", type=Path, default=None, help="Optional ComfyUI API prompt JSON to time.")
    parser.add_argument("--repeats", type=int, default=1, help="Number of timed ComfyUI prompt repeats.")
    parser.add_argument("--timeout", type=float, default=300.0, help="Benchmark timeout in seconds.")
    parser.add_argument("--format", choices=("json", "markdown"), default="json", help="Output format.")
    args = parser.parse_args()

    output: dict[str, Any] = {}
    if args.check_env or (not args.run_op_bench and args.comfy_url is None and args.prompt_json is None):
        output["environment"] = probe_environment()
    if args.run_op_bench:
        output["op_benchmark"] = benchmark_bf16_ops(matrix_size=args.matrix_size)
    if args.comfy_url:
        try:
            output["comfy_system_stats"] = fetch_comfy_system_stats(args.comfy_url)
            if args.prompt_json is not None:
                output["benchmark"] = run_comfy_prompt_benchmark(
                    args.comfy_url,
                    args.prompt_json,
                    timeout_s=args.timeout,
                    repeats=args.repeats,
                )
        except urllib.error.URLError as exc:
            output["comfy_error"] = str(exc)

    if args.format == "markdown":
        print(format_markdown_report(output))
    else:
        print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
