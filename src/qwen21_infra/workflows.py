"""Deterministic ComfyUI evaluation workflow adaptation and structural graph validation."""

from __future__ import annotations

import copy
import json
import os
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .manifest import download_file_if_missing, load_manifest

BUNDLED_WORKFLOWS_DIR = Path(__file__).resolve().parent / "workflows"

PROFILE_WORKFLOW_REPLACEMENTS: dict[str, dict[str, str]] = {
    "full-bf16": {
        "qwen_image_2.1_int8_convrot.safetensors": "qwen_image_2.1_bf16.safetensors",
        "qwen3vl_8b_int8_convrot.safetensors": "qwen3vl_8b_bf16.safetensors",
    },
    "w4a8": {
        "qwen3vl_8b_int8_convrot.safetensors": "qwen3vl_8b_w4a8.safetensors",
        "qwen3.5_9b_qwen_image_2.1_pe_i2i.int8_convrot.safetensors": "qwen3.5_9b_qwen_image_2.1_pe_i2i_w4a8.safetensors",
    },
    "turbo-lite-int8": {
        "qwen_image_2.1_int8_convrot.safetensors": "qwen_image_2.1_turbo_int8_convrot.safetensors",
    },
}


def _replace_workflow_strings(obj: Any, replacements: dict[str, str]) -> Any:
    """Recursively update loader filenames and HF download URLs in a ComfyUI workflow."""
    if isinstance(obj, str):
        if obj in replacements:
            return replacements[obj]
        for src, dst in replacements.items():
            if obj.endswith(f"/{src}"):
                return f"{obj[:-len(src)]}{dst}"
        return obj
    if isinstance(obj, list):
        return [_replace_workflow_strings(item, replacements) for item in obj]
    if isinstance(obj, dict):
        return {key: _replace_workflow_strings(val, replacements) for key, val in obj.items()}
    return obj


def adapt_workflow_for_profile(workflow: dict[str, Any], profile: str = "full-int8") -> dict[str, Any]:
    """Return a copy of a ComfyUI workflow adapted to the selected weight profile."""
    replacements = PROFILE_WORKFLOW_REPLACEMENTS.get(profile)
    if not replacements:
        return copy.deepcopy(workflow)
    return _replace_workflow_strings(copy.deepcopy(workflow), replacements)


def set_deterministic_eval_params(
    workflow: dict[str, Any],
    seed: int = 42,
    steps: int = 25,
    cfg: float = 1.0,
) -> dict[str, Any]:
    """Fix KSampler seed, control_after_generate, steps, and CFG for controlled A/B runs."""
    wf = copy.deepcopy(workflow)

    def _update_nodes(nodes: list[dict[str, Any]]) -> None:
        for node in nodes:
            if node.get("type") == "KSampler":
                vals = list(node.get("widgets_values") or [])
                if len(vals) >= 4:
                    vals[0] = seed
                    vals[1] = "fixed"
                    vals[2] = steps
                    vals[3] = cfg
                    node["widgets_values"] = vals

    _update_nodes(wf.get("nodes", []))
    for sg in wf.get("definitions", {}).get("subgraphs", []):
        _update_nodes(sg.get("nodes", []))
    return wf


def extract_referenced_models(workflow: dict[str, Any]) -> list[str]:
    """Extract unique checkpoint/encoder/VAE filenames from loader nodes."""
    found: set[str] = set()

    def _walk(node_list: list[dict[str, Any]]) -> None:
        for node in node_list:
            props = node.get("properties")
            if isinstance(props, dict):
                for entry in props.get("models") or []:
                    if isinstance(entry, dict) and isinstance(entry.get("name"), str):
                        found.add(entry["name"])
            named = node.get("widgets_values_named")
            if isinstance(named, dict):
                for key, val in named.items():
                    if key.endswith("_name") and isinstance(val, str) and val.endswith(".safetensors"):
                        found.add(val)
            for val in node.get("widgets_values") or []:
                if isinstance(val, str) and val.endswith(".safetensors"):
                    found.add(val)

    _walk(workflow.get("nodes", []))
    for sg in workflow.get("definitions", {}).get("subgraphs", []):
        _walk(sg.get("nodes", []))
    return sorted(found)


def validate_workflow_integrity(workflow: dict[str, Any]) -> list[str]:
    """Validate structural invariants of a ComfyUI root graph and its subgraphs."""
    errors: list[str] = []
    root_nodes = workflow.get("nodes", [])
    root_links = workflow.get("links", [])
    node_ids = {n["id"] for n in root_nodes}

    root_links_by_id: dict[int, list[Any]] = {}
    for link in root_links:
        if not isinstance(link, list) or len(link) < 6:
            errors.append(f"Invalid root link shape: {link!r}")
            continue
        lid, oid, _, tid, _, _ = link
        root_links_by_id[lid] = link
        if oid not in node_ids:
            errors.append(f"Root link {lid} references missing origin node {oid}")
        if tid not in node_ids:
            errors.append(f"Root link {lid} references missing target node {tid}")

    for node in root_nodes:
        nid = node["id"]
        for inp in node.get("inputs", []):
            lid = inp.get("link")
            if lid is not None and lid not in root_links_by_id:
                errors.append(f"Root node {nid} input '{inp.get('name')}' references missing link {lid}")

    for sg in workflow.get("definitions", {}).get("subgraphs", []):
        sg_id = sg.get("id")
        sg_node_ids = {n["id"] for n in sg.get("nodes", [])} | {-10, -20}
        sg_links_by_id = {lnk["id"]: lnk for lnk in sg.get("links", [])}
        in_count = len(sg.get("inputs", []))
        out_count = len(sg.get("outputs", []))
        for lnk in sg.get("links", []):
            lid = lnk["id"]
            if lnk["origin_id"] not in sg_node_ids:
                errors.append(f"Subgraph {sg_id} link {lid} missing origin {lnk['origin_id']}")
            if lnk["target_id"] not in sg_node_ids:
                errors.append(f"Subgraph {sg_id} link {lid} missing target {lnk['target_id']}")
            if lnk["origin_id"] == -10 and not (0 <= lnk["origin_slot"] < in_count):
                errors.append(f"Subgraph {sg_id} link {lid} invalid input slot {lnk['origin_slot']}")
            if lnk["target_id"] == -20 and not (0 <= lnk["target_slot"] < out_count):
                errors.append(f"Subgraph {sg_id} link {lid} invalid output slot {lnk['target_slot']}")
        for sg_node in sg.get("nodes", []):
            for out in sg_node.get("outputs", []):
                for lid in out.get("links") or []:
                    if lid not in sg_links_by_id:
                        errors.append(
                            f"Subgraph {sg_id} node {sg_node['id']} output '{out.get('name')}' dangling link {lid}"
                        )
    return errors


def build_live_preview_workflow(base_wf: dict[str, Any]) -> dict[str, Any]:
    """Expose KSampler + VAEDecode on the root canvas while keeping the Image Edit subgraph compact."""
    wf = copy.deepcopy(base_wf)
    wf["id"] = str(uuid.uuid5(uuid.NAMESPACE_DNS, "qwen21-edit-live-preview-wf-v1"))

    sg = wf["definitions"]["subgraphs"][0]
    new_sg_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, "qwen21-edit-live-preview-sg-v1"))
    sg["id"] = new_sg_id

    nodes_by_id = {n["id"]: n for n in wf["nodes"]}
    sg_node = nodes_by_id[459]
    sg_node["type"] = new_sg_id

    removed_sg_nodes = {457, 458}
    sg["nodes"] = [n for n in sg["nodes"] if n["id"] not in removed_sg_nodes]

    removed_sg_links = {
        link["id"]
        for link in sg["links"]
        if link["origin_id"] in removed_sg_nodes or link["target_id"] in removed_sg_nodes
    }
    sg["links"] = [link for link in sg["links"] if link["id"] not in removed_sg_links]

    remove_in_names = {"cfg", "steps", "scheduler", "scheduler_1"}
    old_to_new_in_slot = {}
    new_sg_inputs = []
    removed_widget_indices = set()
    for old_idx, inp in enumerate(sg["inputs"]):
        if inp["name"] in remove_in_names:
            removed_widget_indices.add(old_idx)
            continue
        new_idx = len(new_sg_inputs)
        old_to_new_in_slot[old_idx] = new_idx
        inp["linkIds"] = [lid for lid in inp.get("linkIds", []) if lid not in removed_sg_links]
        if isinstance(inp.get("pos"), list) and len(inp["pos"]) >= 2:
            inp["pos"][1] = 654 + 20 * new_idx
        new_sg_inputs.append(inp)
    sg["inputs"] = new_sg_inputs

    for link in sg["links"]:
        if link["origin_id"] == -10:
            link["origin_slot"] = old_to_new_in_slot[link["origin_slot"]]

    sg_nodes_by_id = {n["id"]: n for n in sg["nodes"]}
    for node in sg["nodes"]:
        for out in node.get("outputs", []):
            if out.get("links"):
                out["links"] = [lid for lid in out["links"] if lid not in removed_sg_links]

    sg_node["inputs"] = [x for x in sg_node.get("inputs", []) if x["name"] not in remove_in_names]
    sg_node["widgets_values"] = [
        val
        for idx, val in enumerate(sg_node.get("widgets_values", []))
        if idx not in removed_widget_indices
    ]
    if isinstance(sg_node.get("widgets_values_named"), dict):
        for name in remove_in_names:
            sg_node["widgets_values_named"].pop(name, None)

    output_specs = [
        ("MODEL", "MODEL", 469, 0, 910, 915),
        ("positive", "CONDITIONING", 474, 0, 911, 916),
        ("negative", "CONDITIONING", 474, 1, 912, 917),
        ("latent_image", "LATENT", 468, 0, 913, 918),
        ("vae", "VAE", 454, 0, 914, 919),
    ]
    sg["outputs"] = []
    sg_node["outputs"] = []
    for slot_idx, (out_name, out_type, src_id, src_slot, sg_lid, root_lid) in enumerate(output_specs):
        sg["outputs"].append(
            {
                "id": str(uuid.uuid5(uuid.NAMESPACE_DNS, f"qwen21-sg-out-{out_name}")),
                "name": out_name,
                "type": out_type,
                "linkIds": [sg_lid],
                "localized_name": out_name,
                "pos": [1984, 664 + 20 * slot_idx],
            }
        )
        src_outputs = sg_nodes_by_id[src_id]["outputs"][src_slot]
        if src_outputs.get("links") is None:
            src_outputs["links"] = []
        src_outputs["links"].append(sg_lid)
        sg["links"].append(
            {
                "id": sg_lid,
                "origin_id": src_id,
                "origin_slot": src_slot,
                "target_id": -20,
                "target_slot": slot_idx,
                "type": out_type,
            }
        )
        sg_node["outputs"].append({"name": out_name, "type": out_type, "links": [root_lid]})

    sg_state = sg.setdefault("state", {})
    sg_state["lastLinkId"] = max(sg_state.get("lastLinkId", 0), 920)

    ksampler_node = {
        "id": 502,
        "type": "KSampler",
        "pos": [940, 2420],
        "size": [450, 760],
        "flags": {},
        "order": 6,
        "mode": 0,
        "inputs": [
            {"localized_name": "model", "name": "model", "type": "MODEL", "link": 915},
            {"localized_name": "positive", "name": "positive", "type": "CONDITIONING", "link": 916},
            {"localized_name": "negative", "name": "negative", "type": "CONDITIONING", "link": 917},
            {"localized_name": "latent_image", "name": "latent_image", "type": "LATENT", "link": 918},
            {"localized_name": "seed", "name": "seed", "type": "INT", "widget": {"name": "seed"}, "link": None},
            {"localized_name": "steps", "name": "steps", "type": "INT", "widget": {"name": "steps"}, "link": None},
            {"localized_name": "cfg", "name": "cfg", "type": "FLOAT", "widget": {"name": "cfg"}, "link": None},
            {"localized_name": "sampler_name", "name": "sampler_name", "type": "COMBO", "widget": {"name": "sampler_name"}, "link": None},
            {"localized_name": "scheduler", "name": "scheduler", "type": "COMBO", "widget": {"name": "scheduler"}, "link": None},
        ],
        "outputs": [{"localized_name": "LATENT", "name": "LATENT", "type": "LATENT", "links": [920]}],
        "properties": {"Node name for S&R": "KSampler"},
        "widgets_values": [891836227295375, "randomize", 25, 1, "euler", "simple", 1],
    }
    vae_decode_node = {
        "id": 503,
        "type": "VAEDecode",
        "pos": [1430, 2420],
        "size": [260, 60],
        "flags": {},
        "order": 7,
        "mode": 0,
        "inputs": [
            {"localized_name": "samples", "name": "samples", "type": "LATENT", "link": 920},
            {"localized_name": "vae", "name": "vae", "type": "VAE", "link": 919},
        ],
        "outputs": [{"localized_name": "IMAGE", "name": "IMAGE", "type": "IMAGE", "links": [672]}],
        "properties": {"Node name for S&R": "VAEDecode"},
    }

    nodes_by_id[501]["pos"] = [1430, 2520]
    nodes_by_id[501]["size"] = [260, 660]
    nodes_by_id[461]["pos"] = [1730, 2420]
    nodes_by_id[461]["order"] = 8
    nodes_by_id[472]["pos"] = [2480, 2420]
    nodes_by_id[472]["order"] = 9
    wf["nodes"].extend([ksampler_node, vae_decode_node])

    orig_sg_inputs = [n for n in base_wf["nodes"] if n["id"] == 459][0]["inputs"]
    sg_in_idx_by_name = {x["name"]: idx for idx, x in enumerate(sg_node["inputs"])}
    new_root_links = []
    for link in wf["links"]:
        lid, oid, oslot, tid, tslot, ltype = link
        if lid == 672:
            continue
        if tid == 459:
            tslot = sg_in_idx_by_name[orig_sg_inputs[tslot]["name"]]
        new_root_links.append([lid, oid, oslot, tid, tslot, ltype])
    new_root_links.extend(
        [
            [915, 459, 0, 502, 0, "MODEL"],
            [916, 459, 1, 502, 1, "CONDITIONING"],
            [917, 459, 2, 502, 2, "CONDITIONING"],
            [918, 459, 3, 502, 3, "LATENT"],
            [919, 459, 4, 503, 1, "VAE"],
            [920, 502, 0, 503, 0, "LATENT"],
            [672, 503, 0, 461, 0, "IMAGE"],
        ]
    )
    wf["links"] = new_root_links
    wf["last_node_id"] = 503
    wf["last_link_id"] = 920
    if isinstance(wf.get("extra"), dict) and isinstance(wf["extra"].get("ds"), dict):
        wf["extra"]["ds"] = {"scale": 0.55, "offset": [920, -2050]}
    return wf


def ensure_user_workflows(
    workflow_dir: str | Path,
    commit_cb: Callable[[], None] | None = None,
    profile: str = "full-bf16",
    bundled_dir: Path = BUNDLED_WORKFLOWS_DIR,
) -> bool:
    """Seed bundled and upstream ComfyUI workflows adapted to the requested weight profile."""
    manifest = load_manifest()
    workflow_root = Path(workflow_dir)
    workflow_root.mkdir(parents=True, exist_ok=True)
    allow_offline = os.environ.get("QWEN21_ALLOW_OFFLINE_TEMPLATES", "0") == "1"

    changed = False
    if bundled_dir.is_dir():
        for src_wf in sorted(bundled_dir.glob("*.json")):
            dst_wf = workflow_root / src_wf.name
            if not dst_wf.exists() or dst_wf.stat().st_size == 0:
                wf_data = json.loads(src_wf.read_text(encoding="utf-8"))
                adapted = adapt_workflow_for_profile(wf_data, profile=profile)
                dst_wf.write_text(json.dumps(adapted, ensure_ascii=False, indent=2), encoding="utf-8")
                print(f"[OK] Seeded bundled workflow ({profile}): {dst_wf.name}", flush=True)
                changed = True

    for filename, url in manifest["workflow_templates"].items():
        target = workflow_root / filename
        try:
            if download_file_if_missing(url, target):
                if profile in PROFILE_WORKFLOW_REPLACEMENTS:
                    wf_data = json.loads(target.read_text(encoding="utf-8"))
                    adapted = adapt_workflow_for_profile(wf_data, profile=profile)
                    target.write_text(json.dumps(adapted, ensure_ascii=False, indent=2), encoding="utf-8")
                print(f"[OK] Seeded workflow ({profile}): {target.name}", flush=True)
                changed = True
        except Exception as exc:
            if allow_offline and (workflow_root / manifest["live_preview_workflow_filename"]).exists():
                print(f"[WARNING] Skipping remote template {filename} ({exc})", flush=True)
            else:
                raise

    edit_path = workflow_root / "qwen_image_2_1_edit.json"
    live_preview_path = workflow_root / manifest["live_preview_workflow_filename"]
    if edit_path.exists() and (not live_preview_path.exists() or live_preview_path.stat().st_size == 0):
        base_wf = json.loads(edit_path.read_text(encoding="utf-8"))
        live_wf = adapt_workflow_for_profile(build_live_preview_workflow(base_wf), profile=profile)
        tmp_path = live_preview_path.with_name(f"{live_preview_path.name}.tmp")
        try:
            tmp_path.write_text(json.dumps(live_wf, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp_path.replace(live_preview_path)
            print(f"[OK] Seeded workflow ({profile}): {live_preview_path.name}", flush=True)
            changed = True
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise

    if changed and commit_cb is not None:
        commit_cb()
    return changed
