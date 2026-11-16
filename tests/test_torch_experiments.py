"""Real PyTorch execution tests for EXP-01 (causal_condition), EXP-02 (_int_mm & SNR), and EXP-04."""

import importlib.util
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

HAS_TORCH = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(HAS_TORCH, "PyTorch not installed in lightweight stdlib environment")
class TorchExperimentsTests(unittest.TestCase):
    def test_causal_condition_mask_blocks_prefix_to_noise_attention(self) -> None:
        import torch

        from qwen21_infra.experiments import build_causal_condition_mask

        mask = build_causal_condition_mask(seq_len=6, condition_tokens=2, device=torch.device("cpu"))
        self.assertEqual(tuple(mask.shape), (1, 1, 6, 6))
        # Prefix rows [0:2] can attend to prefix cols [0:2] but NOT noise cols [2:6].
        self.assertTrue(mask[0, 0, 0, 1].item())
        self.assertFalse(mask[0, 0, 0, 2].item())
        # Noise rows [2:6] attend to both prefix and noise.
        self.assertTrue(mask[0, 0, 2, 0].item())
        self.assertTrue(mask[0, 0, 2, 5].item())

    def test_walsh_hadamard_rotate_is_orthonormal(self) -> None:
        import torch

        from qwen21_infra.experiments import walsh_hadamard_rotate

        x = torch.randn((4, 128), dtype=torch.float32)
        y = walsh_hadamard_rotate(x, block_size=64)
        recovered = walsh_hadamard_rotate(y, block_size=64)
        self.assertTrue(torch.allclose(x, recovered, atol=1e-5))

    def test_convrot_fidelity_and_runtime_separation(self) -> None:
        from qwen21_infra.experiments import run_quantization_drift_study

        res = run_quantization_drift_study(
            batch_tokens=32,
            hidden_dim=256,
            block_size=64,
            outlier_scale=30.0,
            repeats=2,
            warmup=1,
            force_cpu=True,
        )
        self.assertEqual(res["status"], "OK")
        fid = res["fidelity_comparison"]
        self.assertGreater(fid["hadamard_convrot_int8_emulated"]["snr_db"], fid["naive_int8_emulated"]["snr_db"])
        self.assertGreater(fid["bf16"]["cosine_similarity"], 0.999)
        rt = res["runtime_comparison"]
        self.assertEqual(rt["bf16_matmul"]["repeats"], 2)
        self.assertIn(rt["native_int_mm"]["status"], ("OK", "UNSUPPORTED"))

    def test_sdpa_sweep_with_causal_condition_and_profiler_on_cpu(self) -> None:
        from qwen21_infra.experiments import profile_dit_block, run_sdpa_scaling_sweep

        sdpa = run_sdpa_scaling_sweep(
            seq_lengths=(128,),
            num_heads=4,
            head_dim=64,
            condition_tokens=32,
            include_causal_condition_mask=True,
            iters=2,
            warmup=1,
            force_cpu=True,
        )
        self.assertEqual(sdpa["status"], "OK")
        math_modes = {r["mask_mode"]: r for r in sdpa["rows"] if r["backend"] == "MATH"}
        self.assertEqual(set(math_modes), {"unmasked", "causal_condition"})
        self.assertEqual(math_modes["causal_condition"]["status"], "OK")
        self.assertEqual(math_modes["causal_condition"]["repeats"], 2)

        prof = profile_dit_block(seq_len=128, hidden_dim=256, num_heads=4, top_k=5, force_cpu=True)
        self.assertEqual(prof["status"], "OK")
        self.assertGreater(len(prof["top_ops"]), 0)


if __name__ == "__main__":
    unittest.main()
