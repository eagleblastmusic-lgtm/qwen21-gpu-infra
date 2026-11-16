#!/usr/bin/env bash
set -euo pipefail

# Execute the automated MI300X ROCm evaluation suite (env check + EXP-01..EXP-04) and emit JSON/Markdown artifacts.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
OUT_DIR="${OUT_DIR:-$REPO_DIR/results}"
mkdir -p "$OUT_DIR"

export PYTHONPATH="$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

echo "[1/3] Collecting ROCm environment & rocm-smi baseline..."
python3 -m qwen21_infra.rocm_bench --check-env --run-op-bench --format markdown | tee "$OUT_DIR/env_summary.md"

echo "[2/3] Running EXP-01..EXP-04 empirical GPU suite..."
python3 -m qwen21_infra.experiments --exp all --output "$OUT_DIR/mi300x_experiments.json"

if [ -n "${COMFY_URL:-}" ] && [ -n "${PROMPT_JSON:-}" ]; then
  echo "[3/3] Benchmarking end-to-end ComfyUI workflow at $COMFY_URL ..."
  python3 -m qwen21_infra.rocm_bench     --comfy-url "$COMFY_URL"     --prompt-json "$PROMPT_JSON"     --format markdown | tee "$OUT_DIR/comfy_e2e_report.md"
else
  echo "[3/3] Skipping live ComfyUI prompt run (set COMFY_URL and PROMPT_JSON to include)."
fi
