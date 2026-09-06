#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${QWEN_BOOTSTRAP_PYTHON:-python3}"
BOOTSTRAP_VENV="${QWEN_BOOTSTRAP_VENV:-/tmp/qwen-cloud-bootstrap-venv}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python 3 is required but was not found." >&2
  exit 2
fi

missing_packages=()
command -v git >/dev/null 2>&1 || missing_packages+=(git)
command -v cmake >/dev/null 2>&1 || missing_packages+=(cmake)
command -v ninja >/dev/null 2>&1 || missing_packages+=(ninja-build)
command -v g++ >/dev/null 2>&1 || missing_packages+=(build-essential)

if (( ${#missing_packages[@]} > 0 )); then
  if [[ "$(id -u)" -ne 0 ]] || ! command -v apt-get >/dev/null 2>&1; then
    echo "Missing required packages: ${missing_packages[*]}" >&2
    echo "Run this script as root in a supported Vast.ai or RunPod base image." >&2
    exit 2
  fi
  export DEBIAN_FRONTEND=noninteractive
  apt-get update
  apt-get install -y --no-install-recommends "${missing_packages[@]}" python3-venv ca-certificates
fi

if ! command -v nvcc >/dev/null 2>&1; then
  echo "nvcc was not found. Use a CUDA 12.8+ development image, not a runtime image." >&2
  exit 2
fi

if [[ ! -x "$BOOTSTRAP_VENV/bin/python" ]]; then
  if ! "$PYTHON_BIN" -m venv "$BOOTSTRAP_VENV"; then
    if [[ "$(id -u)" -eq 0 ]] && command -v apt-get >/dev/null 2>&1; then
      export DEBIAN_FRONTEND=noninteractive
      apt-get update
      apt-get install -y --no-install-recommends python3-venv
      "$PYTHON_BIN" -m venv "$BOOTSTRAP_VENV"
    else
      echo "Unable to create the bootstrap virtual environment." >&2
      exit 2
    fi
  fi
fi

if ! "$BOOTSTRAP_VENV/bin/python" -c "import huggingface_hub, hf_xet" >/dev/null 2>&1; then
  "$BOOTSTRAP_VENV/bin/python" -m pip install \
    --disable-pip-version-check \
    --timeout "${PIP_DEFAULT_TIMEOUT:-300}" \
    --retries "${PIP_RETRIES:-10}" \
    --prefer-binary \
    "huggingface_hub>=0.32,<2" \
    "hf_xet>=1,<2"
fi

exec "$BOOTSTRAP_VENV/bin/python" "$PROJECT_DIR/bootstrap.py" "$@"

