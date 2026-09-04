#!/usr/bin/env bash
set -euo pipefail

if (( $# )); then
  echo "This script is preconfigured; run it without parameters." >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${ROOT}/.venv/bin/python"
SETTING="${ROOT}/reproduction/diffpir/settings/ffhq256_inpainting_diffpir_quad20_v1.json"
FIXTURE_ID="ffhq256_inpainting_diffpir_quad20_v1"
ARTIFACT_ROOT="${ROOT}/reproduction/artifacts"
REFERENCE_REPO="/mnt/afs/L202500464/reference-worktrees/diffpir-2a9898"
IMAGES="/mnt/afs/L202500464/DiffPIR/testsets/demo_test"
CHECKPOINT="/mnt/afs/L202500464/DiffPIR/model_zoo/diffusion_ffhq_10m.pt"
RUN_ID="reproduce-$(date -u +%Y%m%dT%H%M%SZ)"
REFERENCE_DEVICE="cuda:0"
DEEPINV_DEVICE="cuda:1"
METRIC_DEVICE="cuda:2"

test -x "${PYTHON}"
test -d "${REFERENCE_REPO}"
test -d "${IMAGES}"
test -f "${CHECKPOINT}"
cd "${ROOT}"
export PYTHONUNBUFFERED=1

if [[ ! -f "${ARTIFACT_ROOT}/fixtures/${FIXTURE_ID}/manifest.json" ]]; then
  "${PYTHON}" reproduction/diffpir/prepare_inputs.py \
    --setting "${SETTING}" --fixture-id "${FIXTURE_ID}" --images "${IMAGES}" \
    --limit 3 --artifact-root "${ARTIFACT_ROOT}"
fi

"${PYTHON}" reproduction/diffpir/run_reference.py \
  --setting "${SETTING}" --fixture-id "${FIXTURE_ID}" --run-id "${RUN_ID}" \
  --reference-repo "${REFERENCE_REPO}" --checkpoint "${CHECKPOINT}" \
  --device "${REFERENCE_DEVICE}" --artifact-root "${ARTIFACT_ROOT}" &
reference_pid=$!
"${PYTHON}" reproduction/diffpir/run_deepinv.py \
  --setting "${SETTING}" --fixture-id "${FIXTURE_ID}" --run-id "${RUN_ID}" \
  --checkpoint "${CHECKPOINT}" --device "${DEEPINV_DEVICE}" \
  --artifact-root "${ARTIFACT_ROOT}" &
deepinv_pid=$!

reference_status=0
deepinv_status=0
wait "${reference_pid}" || reference_status=$?
wait "${deepinv_pid}" || deepinv_status=$?
if (( reference_status || deepinv_status )); then
  exit 1
fi

"${PYTHON}" reproduction/dps/compare.py \
  --setting "${SETTING}" --fixture-id "${FIXTURE_ID}" --run-id "${RUN_ID}" \
  --metric-device "${METRIC_DEVICE}" --artifact-root "${ARTIFACT_ROOT}"
