#!/usr/bin/env bash
set -euo pipefail

if (( $# )); then
  echo "This script is preconfigured; run it without parameters." >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON=("/mnt/afs/L202500464/uv-env-tool.sh" --proxy off uv run --no-sync python)
ARTIFACT_ROOT="${ROOT}/reproduction/artifacts"
REFERENCE_REPO="/mnt/afs/L202500464/reference-worktrees/diffpir-2a9898"
IMAGES="/mnt/afs/L202500464/DiffPIR/testsets/demo_test"
CHECKPOINT="/mnt/afs/L202500464/DiffPIR/model_zoo/diffusion_ffhq_10m.pt"
RUN_ID="sr4-$(date -u +%Y%m%dT%H%M%SZ)"
SETTING_IDS=(
  ffhq256_bicubic_sr4_sigma005_diffpir_quad20_v2
  ffhq256_bicubic_sr4_sigma005_diffpir_quad100_v2
  ffhq256_bicubic_sr4_sigma000_diffpir_quad20_v2
  ffhq256_bicubic_sr4_sigma000_diffpir_quad100_v2
)
DEVICES=(cuda:0 cuda:1 cuda:2 cuda:3)

wait_jobs() {
  local failed=0 pid
  for pid in "$@"; do
    wait "${pid}" || failed=1
  done
  (( failed == 0 ))
}

test -x "${ROOT}/.venv/bin/python"
test -d "${REFERENCE_REPO}"
test -d "${IMAGES}"
test -f "${CHECKPOINT}"
cd "${ROOT}"
export PYTHONUNBUFFERED=1

echo "[1/3] Preparing four immutable five-image SR fixtures"
for setting_id in "${SETTING_IDS[@]}"; do
  setting="${ROOT}/reproduction/diffpir/settings/${setting_id}.json"
  if [[ ! -f "${ARTIFACT_ROOT}/fixtures/${setting_id}/manifest.json" ]]; then
    "${PYTHON[@]}" reproduction/diffpir/prepare_inputs.py \
      --setting "${setting}" --fixture-id "${setting_id}" --images "${IMAGES}" \
      --reference-repo "${REFERENCE_REPO}" --limit 5 \
      --artifact-root "${ARTIFACT_ROOT}"
  fi
done

echo "[2/3] Running the pinned original repository first"
pids=()
for index in "${!SETTING_IDS[@]}"; do
  setting_id="${SETTING_IDS[index]}"
  "${PYTHON[@]}" reproduction/diffpir/run_reference.py \
    --setting "${ROOT}/reproduction/diffpir/settings/${setting_id}.json" \
    --fixture-id "${setting_id}" --run-id "${RUN_ID}" \
    --reference-repo "${REFERENCE_REPO}" --checkpoint "${CHECKPOINT}" \
    --device "${DEVICES[index]}" --artifact-root "${ARTIFACT_ROOT}" &
  pids+=("$!")
done
wait_jobs "${pids[@]}"

echo "[3/3] Recording original-repository five-image metrics (not paper aggregates)"
pids=()
for index in "${!SETTING_IDS[@]}"; do
  setting_id="${SETTING_IDS[index]}"
  "${PYTHON[@]}" reproduction/diffpir/evaluate_reference.py \
    --setting "${ROOT}/reproduction/diffpir/settings/${setting_id}.json" \
    --fixture-id "${setting_id}" --run-id "${RUN_ID}" \
    --metric-device "${DEVICES[index]}" --artifact-root "${ARTIFACT_ROOT}" &
  pids+=("$!")
done
wait_jobs "${pids[@]}"

echo "Stopped at the original-repository paper gate; DeepInv was not started."
echo "Review: ${ROOT}/reproduction/diffpir/reports/sr4_reference_gate_20260905.zh-CN.md"
echo "Run ID: ${RUN_ID}"
echo "Results: ${ARTIFACT_ROOT}/runs/diffpir/*/${RUN_ID}"
