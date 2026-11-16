#!/usr/bin/env bash
set -euo pipefail

# Provision ComfyUI + Qwen-Image-2.1 (full-bf16) on AMD Instinct MI300X (ROCm) with pinned refs.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}"

if [ ! -d "$REPO_DIR/src/qwen21_infra" ]; then
  echo "[ERROR] Cannot locate $REPO_DIR/src/qwen21_infra." >&2
  exit 1
fi

COMFY_DIR="${COMFY_DIR:-$HOME/ComfyUI}"
COMFYUI_REF="${COMFYUI_REF:-v0.45.0}"
VENV_DIR="${VENV_DIR:-$HOME/comfy_rocm_venv}"
PROFILE="${QWEN21_MODEL_PROFILE:-full-bf16}"
ROCM_INDEX_URL="${ROCM_INDEX_URL:-https://download.pytorch.org/whl/rocm6.3}"
TORCH_INSTALL_SPEC="${TORCH_INSTALL_SPEC:-torch torchvision torchaudio}"
PORT="${COMFY_PORT:-8188}"

echo "[1/5] Preparing Python environment ($VENV_DIR)..."
if [ ! -d "$VENV_DIR" ]; then
  python3 -m venv --system-site-packages "$VENV_DIR"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

echo "[2/5] Checking ROCm PyTorch..."
if python3 -c "import torch; assert getattr(torch.version, 'hip', None) and torch.cuda.is_available(); print('[OK] ROCm PyTorch:', torch.__version__, 'HIP:', torch.version.hip, 'Device:', torch.cuda.get_device_name(0))" 2>/dev/null; then
  :
else
  echo "[INFO] Installing pinned stable ROCm PyTorch ($TORCH_INSTALL_SPEC) from $ROCM_INDEX_URL ..."
  pip install --upgrade pip
  # shellcheck disable=SC2086
  pip install $TORCH_INSTALL_SPEC --index-url "$ROCM_INDEX_URL"
fi

echo "[3/5] Installing ComfyUI (pinned ref=$COMFYUI_REF) in $COMFY_DIR ..."
if [ ! -d "$COMFY_DIR/.git" ]; then
  git clone --depth 1 --branch "$COMFYUI_REF" https://github.com/comfyanonymous/ComfyUI.git "$COMFY_DIR" 2>/dev/null || {
    git clone https://github.com/comfyanonymous/ComfyUI.git "$COMFY_DIR"
    git -C "$COMFY_DIR" checkout --detach "$COMFYUI_REF"
  }
fi

# Strip torch/torchvision/torchaudio so pip preserves preinstalled ROCm/HIP PyTorch wheels.
FILTERED_REQS="$(mktemp)"
trap 'rm -f "$FILTERED_REQS"' EXIT
grep -viE '^(torch|torchvision|torchaudio)([<>=!~ ]|$)' "$COMFY_DIR/requirements.txt" > "$FILTERED_REQS"
pip install -r "$FILTERED_REQS"
pip install -U "huggingface-hub[hf_xet]"

echo "[4/5] Staging Qwen-Image-2.1 weights (profile=$PROFILE) and workflows..."
export HF_XET_HIGH_PERFORMANCE=1
export HF_HUB_DISABLE_TELEMETRY=1
PYTHONPATH="$REPO_DIR/src" python3 -m qwen21_infra.manifest   --model-dir "$COMFY_DIR/models"   --workflow-dir "$COMFY_DIR/user/default/workflows"   --profile "$PROFILE"

echo "[5/5] Launching ComfyUI bound to 127.0.0.1:$PORT (tunnel via SSH)..."
echo "      ssh -N -L ${PORT}:127.0.0.1:${PORT} <user>@<vm-ip>"
cd "$COMFY_DIR"
exec python3 main.py   --listen 127.0.0.1   --port "$PORT"   --preview-method taesd   --disable-auto-launch
