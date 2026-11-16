"""Canonical Qwen-Image-2.1 weight manifest and VAE 16x / patch_size=1 VRAM budget model."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

MANIFEST_PATH = Path(__file__).resolve().parent / "models.json"

# Verified against AutoencoderKLQwenImage21 (vae/config.json) & QwenImage21Transformer2DModel:
# 16x spatial compression, 64 latent channels, DiT patch_size=1, 32 layers, 32 heads x 128 head_dim.
VAE_SPATIAL_COMPRESSION = 16
VAE_LATENT_CHANNELS = 64
DIT_PATCH_SIZE = 1

PROFILE_METADATA: dict[str, dict[str, Any]] = {
    "full-bf16": {
        "precision": "bf16",
        "quantized": False,
        "documented_weights_gib": 32.4,
        "recommended_min_vram_gib": 48,
        "target_hardware": "AMD Instinct MI300X (192 GB HBM3, ROCm gfx942)",
        "component_breakdown_gib": {
            "dit_backbone_7b_bf16": 14.2,
            "qwen3_vl_8b_bf16": 17.53,
            "vae_bf16": 0.68,
        },
        "notes": "Unquantized BF16 reference weights; exceeds 16-24 GB GPUs prior to 2K activations.",
    },
    "full-int8": {
        "precision": "int8_convrot",
        "quantized": True,
        "documented_weights_gib": 34.0,
        "recommended_min_vram_gib": 24,
        "target_hardware": "24 GB VRAM baseline + Prompt Enhancer",
        "notes": "Includes base + turbo DiT and Qwen3.5-9B prompt enhancer in INT8 ConvRot.",
    },
    "lite-int8": {
        "precision": "int8_convrot",
        "quantized": True,
        "documented_weights_gib": 17.3,
        "recommended_min_vram_gib": 16,
        "target_hardware": "16 GB VRAM constrained baseline",
        "notes": "Base DiT + Qwen3-VL 8B + VAE in INT8 ConvRot without turbo DiT or enhancer.",
    },
    "turbo-lite-int8": {
        "precision": "int8_convrot",
        "quantized": True,
        "documented_weights_gib": None,
        "recommended_min_vram_gib": 16,
        "target_hardware": "16 GB VRAM fast-sampling baseline",
        "notes": "Turbo DiT variant + Qwen3-VL 8B + VAE.",
    },
    "w4a8": {
        "precision": "w4a8",
        "quantized": True,
        "documented_weights_gib": None,
        "recommended_min_vram_gib": 16,
        "target_hardware": "16 GB VRAM W4A8 encoder comparison",
        "notes": "W4A8 quantized text/prompt encoders with INT8 ConvRot DiT.",
    },
}


def load_manifest(manifest_path: Path = MANIFEST_PATH) -> dict[str, Any]:
    """Load the canonical Qwen-Image-2.1 model manifest JSON."""
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def get_profile_files(profile: str = "full-bf16", manifest: dict[str, Any] | None = None) -> list[str]:
    """Return the relative weight paths for a named model profile."""
    data = manifest or load_manifest()
    profiles = data["profiles"]
    if profile not in profiles:
        available = ", ".join(sorted(profiles))
        raise ValueError(f"Unknown model profile '{profile}'. Available: {available}")
    return list(profiles[profile])


def get_profile_summary(profile: str, manifest: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return resolved file paths alongside documented VRAM and precision metadata."""
    files = get_profile_files(profile=profile, manifest=manifest)
    meta = PROFILE_METADATA.get(profile, {})
    return {
        "profile": profile,
        "file_count": len(files),
        "files": files,
        "precision": meta.get("precision"),
        "quantized": meta.get("quantized"),
        "documented_weights_gib": meta.get("documented_weights_gib"),
        "recommended_min_vram_gib": meta.get("recommended_min_vram_gib"),
        "target_hardware": meta.get("target_hardware"),
        "component_breakdown_gib": meta.get("component_breakdown_gib"),
        "notes": meta.get("notes"),
    }


def estimate_memory_budget(
    profile: str,
    width: int = 1024,
    height: int = 1024,
    batch_size: int = 1,
    include_activation_headroom: bool = True,
) -> dict[str, Any]:
    """Estimate static weights plus pre-calibration heuristic activation headroom.

    Latent tensor dimensions follow AutoencoderKLQwenImage21 (16x spatial compression,
    64 channels) and QwenImage21Transformer2DModel (patch_size=1). Activation figures
    are pre-calibration analytical projections awaiting EXP-03 GPU telemetry on MI300X.
    """
    summary = get_profile_summary(profile)
    weights_gib = summary["documented_weights_gib"]
    if width <= 0 or height <= 0 or batch_size <= 0:
        raise ValueError("width, height, and batch_size must be positive integers.")

    latent_h = max(1, math.ceil(height / VAE_SPATIAL_COMPRESSION))
    latent_w = max(1, math.ceil(width / VAE_SPATIAL_COMPRESSION))
    dit_spatial_tokens = batch_size * math.ceil(latent_h / DIT_PATCH_SIZE) * math.ceil(latent_w / DIT_PATCH_SIZE)

    est_activation_gib = 0.0
    if include_activation_headroom:
        megapixels = (width * height) / (1024.0 * 1024.0)
        est_activation_gib = round(batch_size * (2.2 * megapixels + 1.1 * (megapixels ** 1.35)), 2)

    total_est_gib = round(weights_gib + est_activation_gib, 2) if weights_gib is not None else None
    fits_tiers = {}
    if total_est_gib is not None:
        for tier_gib in (16, 24, 48, 80, 192):
            fits_tiers[f"{tier_gib}GB"] = total_est_gib <= (tier_gib * 0.92)

    return {
        "profile": profile,
        "resolution": f"{width}x{height}",
        "batch_size": batch_size,
        "vae_compression": VAE_SPATIAL_COMPRESSION,
        "patch_size": DIT_PATCH_SIZE,
        "latent_shape": [batch_size, VAE_LATENT_CHANNELS, latent_h, latent_w],
        "dit_spatial_tokens": dit_spatial_tokens,
        "documented_weights_gib": weights_gib,
        "heuristic_activation_headroom_gib": est_activation_gib,
        "estimated_total_vram_gib": total_est_gib,
        "activation_model_status": "HEURISTIC_PRE_CALIBRATION",
        "fits_vram_tiers_without_offload": fits_tiers,
    }


def download_file_if_missing(url: str, target_path: Path) -> bool:
    """Download a file atomically if missing or empty; returns True when downloaded."""
    if target_path.exists() and target_path.stat().st_size > 0:
        return False

    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target_path.with_name(f"{target_path.name}.tmp")
    try:
        with urllib.request.urlopen(url, timeout=60) as resp, tmp_path.open("wb") as out:
            shutil.copyfileobj(resp, out)
        if tmp_path.stat().st_size <= 0:
            raise RuntimeError(f"Downloaded empty file from {url}")
        tmp_path.replace(target_path)
        return True
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def ensure_models(
    model_dir: str | Path,
    profile: str = "full-bf16",
    include_taesd: bool = True,
    commit_cb: Callable[[], None] | None = None,
) -> bool:
    """Ensure ComfyUI model directories exist and download missing weights for the chosen profile."""
    from huggingface_hub import hf_hub_download

    manifest = load_manifest()
    hf_repo_id = manifest["hf_repo_id"]
    model_root = Path(model_dir)
    for subdir in manifest["standard_model_subdirs"]:
        (model_root / subdir).mkdir(parents=True, exist_ok=True)

    downloaded_any = False
    for rel_path in get_profile_files(profile=profile, manifest=manifest):
        target_path = model_root / rel_path
        if target_path.exists() and target_path.stat().st_size > 0:
            print(f"[OK] Already cached: {rel_path}")
            continue
        print(f"[DOWNLOAD] Fetching {hf_repo_id}/{rel_path} ...")
        hf_hub_download(
            repo_id=hf_repo_id,
            filename=rel_path,
            local_dir=str(model_root),
        )
        downloaded_any = True

    if include_taesd:
        taesd = manifest["taesd_decoder"]
        taesd_path = model_root / taesd["rel_path"]
        if taesd_path.exists() and taesd_path.stat().st_size > 0:
            print(f"[OK] Already cached: {taesd['rel_path']}")
        else:
            print(f"[DOWNLOAD] Fetching {taesd['url']} -> {taesd['rel_path']} ...")
            if download_file_if_missing(taesd["url"], taesd_path):
                downloaded_any = True

    if downloaded_any and commit_cb is not None:
        commit_cb()
    return downloaded_any


def main() -> None:
    """CLI entrypoint for inspecting Qwen-Image-2.1 profiles, budgets, and staging weights."""
    from .workflows import ensure_user_workflows

    manifest = load_manifest()
    parser = argparse.ArgumentParser(description="Inspect and stage Qwen-Image-2.1 ComfyUI profiles.")
    parser.add_argument("--list-profiles", action="store_true", help="Print JSON summary of all weight profiles.")
    parser.add_argument("--budget", action="store_true", help="Print heuristic VRAM sweep across 1K-2K resolutions.")
    parser.add_argument("--model-dir", type=Path, default=Path.cwd() / "models", help="Target models directory.")
    parser.add_argument("--profile", choices=sorted(manifest["profiles"]), default="full-bf16", help="Weight profile.")
    parser.add_argument("--workflow-dir", type=Path, default=None, help="Optional workflows output directory.")
    parser.add_argument("--no-taesd", action="store_true", help="Skip TAESD decoder download.")
    args = parser.parse_args()

    if args.list_profiles:
        summaries = [get_profile_summary(name, manifest=manifest) for name in sorted(manifest["profiles"])]
        print(json.dumps(summaries, indent=2))
        return

    if args.budget:
        sweep = [
            estimate_memory_budget(args.profile, width=dim, height=dim, batch_size=batch)
            for dim in (1024, 1536, 2048)
            for batch in (1, 2)
        ]
        print(json.dumps(sweep, indent=2))
        return

    ensure_models(args.model_dir, profile=args.profile, include_taesd=not args.no_taesd)
    if args.workflow_dir is not None:
        ensure_user_workflows(args.workflow_dir, profile=args.profile)


if __name__ == "__main__":
    main()
