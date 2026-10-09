#!/usr/bin/env bash
#
# Versioned Podman invocation for the ROCm/PyTorch training image — S05/T01.
#
# This is the one documented way to enter the neural slice's compute
# environment. It:
#   1. (optionally) builds the image from containers/Containerfile,
#   2. runs it with the standard AMD device pass-through (/dev/kfd, /dev/dri),
#      seccomp unconfined, keep-groups, and host IPC — the canonical flags for
#      ROCm in a rootless container,
#   3. mounts the repository at /workspace so the generated report comes back
#      to the host.
#
# Default action is the device check (the image ENTRYPOINT is
# `python -m heart.gpu.device_check`), which writes:
#   reports/gpu_check.md    (human-readable verdict)
#   reports/gpu_check.json  (machine-readable sidecar)
#
# Usage:
#   containers/run-rocm.sh                 # build if missing, run device check
#   containers/run-rocm.sh --build         # force a rebuild, then run
#   containers/run-rocm.sh build           # build only
#   containers/run-rocm.sh shell           # interactive shell in the image
#   containers/run-rocm.sh -- --gfx-target gfx1201
#
# Environment overrides:
#   HEART_ROCM_IMAGE   image tag to build/run (default below)
#   HEART_ROCM_FORCE_BUILD=1   rebuild even if the image exists
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

IMAGE="${HEART_ROCM_IMAGE:-localhost/heart-rocm:pytorch2.10.0-rocm7.2.4}"
CONTAINERFILE="${SCRIPT_DIR}/Containerfile"
REPORT_PATH="${REPO_ROOT}/reports/gpu_check.md"
JSON_PATH="${REPO_ROOT}/reports/gpu_check.json"

ACTION="run"
FORCE_BUILD=0
PASSTHROUGH=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    build) ACTION="build"; shift ;;
    shell) ACTION="shell"; shift ;;
    --build) FORCE_BUILD=1; shift ;;
    --) shift; PASSTHROUGH=("$@"); break ;;
    *) PASSTHROUGH+=("$1"); shift ;;
  esac
done

if [[ "${HEART_ROCM_FORCE_BUILD:-0}" == "1" ]]; then
  FORCE_BUILD=1
fi

image_exists() {
  podman image exists "${IMAGE}" >/dev/null 2>&1
}

build_image() {
  echo "==> building ${IMAGE} from ${CONTAINERFILE}"
  podman build -f "${CONTAINERFILE}" -t "${IMAGE}" "${REPO_ROOT}"
}

ensure_image() {
  if [[ "${FORCE_BUILD}" == "1" ]] || ! image_exists; then
    build_image
  else
    echo "==> image ${IMAGE} already present"
  fi
}

# The canonical ROCm device pass-through. `--security-opt seccomp=unconfined`
# and `--group-add keep-groups` let the rootless container keep the host's
# render/video group access to /dev/kfd and /dev/dri.
#
# The repo is mounted BOTH at /workspace (the documented workdir) and at its
# host absolute path: an MLflow experiment created on the host bakes an
# absolute artifact_location (e.g. file:///…/experiments/mlruns/artifacts), so
# container-written runs only stay host-readable when that path resolves inside
# the container too.
run_container() {
  podman run --rm \
    --name heart-rocm \
    --device=/dev/kfd \
    --device=/dev/dri \
    --security-opt seccomp=unconfined \
    --group-add keep-groups \
    --ipc=host \
    -e PYTHONPATH=/workspace/src \
    -v "${REPO_ROOT}:/workspace:Z" \
    -v "${REPO_ROOT}:${REPO_ROOT}:Z" \
    -w /workspace \
    "${IMAGE}" "$@"
}

case "${ACTION}" in
  build)
    build_image
    ;;
  shell)
    ensure_image
    run_container bash
    ;;
  run)
    ensure_image
    echo "==> running GPU device check in ${IMAGE}"
    mkdir -p "${REPO_ROOT}/reports"
    run_container --report-path /workspace/reports/gpu_check.md \
                  --json-path /workspace/reports/gpu_check.json \
                  "${PASSTHROUGH[@]}"
    echo "==> report: ${REPORT_PATH}"
    echo "==> json:   ${JSON_PATH}"
    ;;
esac
