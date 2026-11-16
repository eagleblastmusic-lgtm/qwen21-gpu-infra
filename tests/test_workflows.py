"""Unit tests for ComfyUI workflow integrity, deterministic A/B seeding, and offline loading."""

import copy
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from qwen21_infra.workflows import (
    BUNDLED_WORKFLOWS_DIR,
    adapt_workflow_for_profile,
    build_live_preview_workflow,
    ensure_user_workflows,
    extract_referenced_models,
    set_deterministic_eval_params,
    validate_workflow_integrity,
)


def _load_bundled_workflow() -> dict:
    wf_path = BUNDLED_WORKFLOWS_DIR / "qwen_image_2_1_edit_live_preview.json"
    return json.loads(wf_path.read_text(encoding="utf-8"))


class WorkflowTests(unittest.TestCase):
    def test_bundled_live_preview_workflow_passes_integrity_check(self) -> None:
        wf = _load_bundled_workflow()
        self.assertEqual(validate_workflow_integrity(wf), [])

    def test_integrity_validator_detects_dangling_link(self) -> None:
        wf = _load_bundled_workflow()
        wf["links"] = [link for link in wf["links"] if link[0] != 920]
        errors = validate_workflow_integrity(wf)
        self.assertTrue(any("920" in err for err in errors), msg=f"Unexpected errors: {errors}")

    def test_adapt_workflow_for_full_bf16_updates_active_loaders(self) -> None:
        base = _load_bundled_workflow()
        adapted = adapt_workflow_for_profile(base, profile="full-bf16")
        self.assertEqual(validate_workflow_integrity(adapted), [])
        models = extract_referenced_models(adapted)
        self.assertIn("qwen_image_2.1_bf16.safetensors", models)
        self.assertIn("qwen3vl_8b_bf16.safetensors", models)
        self.assertNotIn("qwen_image_2.1_int8_convrot.safetensors", models)

    def test_deterministic_eval_params_fixes_ksampler_seed(self) -> None:
        base = _load_bundled_workflow()
        fixed = set_deterministic_eval_params(base, seed=1337, steps=30, cfg=1.5)
        ksamplers = [n for n in fixed["nodes"] if n.get("type") == "KSampler"]
        self.assertEqual(len(ksamplers), 1)
        self.assertEqual(ksamplers[0]["widgets_values"][:4], [1337, "fixed", 30, 1.5])

    def test_build_live_preview_workflow_refactors_subgraph_cleanly(self) -> None:
        base_wf = {
            "id": "orig",
            "nodes": [
                {
                    "id": 459,
                    "type": "sg-orig",
                    "inputs": [
                        {"name": "cfg"},
                        {"name": "prompt"},
                        {"name": "steps"},
                        {"name": "scheduler"},
                        {"name": "scheduler_1"},
                    ],
                    "widgets_values": [1.0, "hi", 25, "simple", "normal"],
                    "outputs": [],
                },
                {"id": 501, "type": "MarkdownNote", "pos": [0, 0], "size": [100, 100]},
                {"id": 461, "type": "SaveImage", "pos": [0, 0], "order": 1},
                {"id": 472, "type": "PrimitiveString", "pos": [0, 0], "order": 2},
            ],
            "links": [
                [10, 472, 0, 459, 1, "STRING"],
                [672, 459, 0, 461, 0, "IMAGE"],
            ],
            "definitions": {
                "subgraphs": [
                    {
                        "id": "sg-orig",
                        "inputs": [
                            {"name": "cfg", "linkIds": [1]},
                            {"name": "prompt", "linkIds": [2], "pos": [0, 0]},
                            {"name": "steps", "linkIds": [3]},
                            {"name": "scheduler", "linkIds": [4]},
                            {"name": "scheduler_1", "linkIds": [5]},
                        ],
                        "outputs": [],
                        "nodes": [
                            {"id": 457, "outputs": [{"links": [672]}]},
                            {"id": 458, "outputs": [{"links": []}]},
                            {"id": 469, "outputs": [{"links": []}]},
                            {"id": 474, "outputs": [{"links": []}, {"links": []}]},
                            {"id": 468, "outputs": [{"links": []}]},
                            {"id": 454, "outputs": [{"links": []}]},
                        ],
                        "links": [
                            {"id": 1, "origin_id": -10, "origin_slot": 0, "target_id": 457, "target_slot": 0, "type": "FLOAT"},
                            {"id": 2, "origin_id": -10, "origin_slot": 1, "target_id": 474, "target_slot": 0, "type": "STRING"},
                            {"id": 672, "origin_id": 458, "origin_slot": 0, "target_id": -20, "target_slot": 0, "type": "IMAGE"},
                        ],
                    }
                ]
            },
        }
        rebuilt = build_live_preview_workflow(copy.deepcopy(base_wf))
        self.assertEqual(validate_workflow_integrity(rebuilt), [])
        root_types = {n["type"] for n in rebuilt["nodes"]}
        self.assertIn("KSampler", root_types)
        self.assertIn("VAEDecode", root_types)

    def test_ensure_user_workflows_offline_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, mock.patch.dict(
            os.environ, {"QWEN21_ALLOW_OFFLINE_TEMPLATES": "1"}
        ), mock.patch("qwen21_infra.workflows.download_file_if_missing", side_effect=OSError("offline")):
            changed = ensure_user_workflows(tmpdir, profile="full-bf16")
            self.assertTrue(changed)
            seeded = json.loads((Path(tmpdir) / "qwen_image_2_1_edit_live_preview.json").read_text(encoding="utf-8"))
            self.assertIn("qwen_image_2.1_bf16.safetensors", extract_referenced_models(seeded))


if __name__ == "__main__":
    unittest.main()
