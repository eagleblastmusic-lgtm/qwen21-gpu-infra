"""Unit tests for canonical Qwen-Image-2.1 profile definitions and 16x VAE / patch=1 VRAM budgets."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from qwen21_infra.manifest import (
    DIT_PATCH_SIZE,
    VAE_LATENT_CHANNELS,
    VAE_SPATIAL_COMPRESSION,
    estimate_memory_budget,
    get_profile_files,
    get_profile_summary,
    load_manifest,
)


class ManifestTests(unittest.TestCase):
    def test_all_expected_profiles_present(self) -> None:
        manifest = load_manifest()
        self.assertEqual(
            sorted(manifest["profiles"]),
            ["full-bf16", "full-int8", "lite-int8", "turbo-lite-int8", "w4a8"],
        )

    def test_full_bf16_contains_unquantized_weights_and_exceeds_24gb(self) -> None:
        summary = get_profile_summary("full-bf16")
        self.assertFalse(summary["quantized"])
        self.assertEqual(summary["precision"], "bf16")
        self.assertGreater(summary["documented_weights_gib"], 30.0)
        self.assertGreaterEqual(summary["recommended_min_vram_gib"], 48)
        self.assertIn("diffusion_models/qwen_image_2.1_bf16.safetensors", summary["files"])

    def test_memory_budget_uses_vae_16x_64ch_patch_1_geometry(self) -> None:
        self.assertEqual((VAE_SPATIAL_COMPRESSION, VAE_LATENT_CHANNELS, DIT_PATCH_SIZE), (16, 64, 1))
        budget_2k = estimate_memory_budget("full-bf16", width=2048, height=2048, batch_size=1)
        self.assertEqual(budget_2k["latent_shape"], [1, 64, 128, 128])
        self.assertEqual(budget_2k["dit_spatial_tokens"], 16384)
        self.assertEqual(budget_2k["activation_model_status"], "HEURISTIC_PRE_CALIBRATION")
        self.assertFalse(budget_2k["fits_vram_tiers_without_offload"]["24GB"])
        self.assertTrue(budget_2k["fits_vram_tiers_without_offload"]["192GB"])

    def test_unknown_profile_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            get_profile_files("nonexistent-fp4")


if __name__ == "__main__":
    unittest.main()
