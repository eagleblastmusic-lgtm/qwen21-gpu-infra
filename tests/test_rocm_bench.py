"""Unit tests for ROCm environment probe, median stats, rocm-smi parser, and ComfyUI telemetry."""

import json
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from qwen21_infra.experiments import EXPERIMENT_CATALOG, run_resolution_memory_model
from qwen21_infra.rocm_bench import (
    benchmark_bf16_ops,
    format_markdown_report,
    parse_rocm_smi_json,
    probe_environment,
    run_comfy_prompt_benchmark,
    summarize_timings_ms,
)


class _MockComfyHandler(BaseHTTPRequestHandler):
    poll_count = 0

    def log_message(self, format: str, *args) -> None:  # noqa: A002
        return

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/system_stats":
            payload = {
                "devices": [
                    {
                        "name": "Mock AMD Instinct MI300X",
                        "vram_total": 192 * 1024**3,
                        "vram_free": 150 * 1024**3,
                    }
                ]
            }
            body = json.dumps(payload).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path.startswith("/history/job-"):
            type(self).poll_count += 1
            prompt_id = self.path.split("/")[-1]
            payload = {prompt_id: {"status": {"completed": True}}}
            body = json.dumps(payload).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_response(404)
        self.end_headers()

    def do_POST(self) -> None:  # noqa: N802
        if self.path == "/prompt":
            length = int(self.headers.get("Content-Length", "0"))
            _ = self.rfile.read(length)
            type(self).poll_count += 1
            body = json.dumps({"prompt_id": f"job-{type(self).poll_count}"}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_response(404)
        self.end_headers()


class RocmBenchTests(unittest.TestCase):
    def test_summarize_timings_ms_computes_median_and_p95(self) -> None:
        stats = summarize_timings_ms([50.0, 10.0, 20.0, 30.0, 40.0])
        self.assertEqual(stats["repeats"], 5)
        self.assertEqual(stats["median_ms"], 30.0)
        self.assertEqual(stats["p95_ms"], 50.0)
        self.assertEqual(stats["min_ms"], 10.0)

    def test_probe_environment_schema_is_well_formed(self) -> None:
        env = probe_environment()
        self.assertIn(env["backend"], ("rocm", "cuda", "cpu", "none"))
        self.assertIsInstance(env["devices"], list)
        self.assertIn("rocm_smi", env)

    def test_parse_rocm_smi_json_extracts_card_metrics(self) -> None:
        sample = json.dumps(
            {
                "card0": {
                    "GPU use (%)": "98",
                    "GPU memory use (%)": "44.5",
                    "Temperature (Sensor edge) (C)": "62.0",
                    "Average Graphics Package Power (W)": "540.2",
                }
            }
        )
        cards = parse_rocm_smi_json(sample)
        self.assertEqual(len(cards), 1)
        self.assertAlmostEqual(cards[0]["gpu_busy_percent"], 98.0)
        self.assertAlmostEqual(cards[0]["power_w"], 540.2)

    def test_experiment_catalog_and_resolution_model(self) -> None:
        self.assertEqual([e["id"] for e in EXPERIMENT_CATALOG], ["EXP-01", "EXP-02", "EXP-03", "EXP-04"])
        sweep = run_resolution_memory_model(force_cpu=True)
        self.assertEqual(sweep["status"], "OK")
        self.assertEqual(len(sweep["model_sweep"]), 12)

    def test_op_benchmark_returns_explicit_skip_without_gpu(self) -> None:
        env = probe_environment()
        if not env["gpu_available"]:
            res = benchmark_bf16_ops(matrix_size=64, warmup=1, iters=1)
            self.assertIn(res["status"], ("SKIPPED_NO_TORCH", "SKIPPED_NO_GPU"))

    def test_run_comfy_prompt_benchmark_against_local_mock_repeats(self) -> None:
        _MockComfyHandler.poll_count = 0
        server = HTTPServer(("127.0.0.1", 0), _MockComfyHandler)
        port = server.server_address[1]
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                prompt_file = Path(tmpdir) / "prompt.json"
                prompt_file.write_text(json.dumps({"1": {"class_type": "CLIPTextEncode"}}), encoding="utf-8")
                res = run_comfy_prompt_benchmark(
                    f"http://127.0.0.1:{port}",
                    prompt_file,
                    timeout_s=5.0,
                    poll_interval_s=0.01,
                    repeats=3,
                )
            self.assertEqual(res["timing_summary"]["repeats"], 3)
            self.assertEqual(len(res["runs"]), 3)
            self.assertEqual(res["status"], {"completed": True})
            md = format_markdown_report({"environment": probe_environment(), "benchmark": res})
            self.assertIn("over `3` run(s)", md)
        finally:
            server.shutdown()
            server.server_close()


if __name__ == "__main__":
    unittest.main()
