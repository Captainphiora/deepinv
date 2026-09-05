#!/usr/bin/env bash
set -euo pipefail

if (( $# )); then
  echo "This script is preconfigured; run it without parameters." >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
UV="/mnt/afs/L202500464/uv-env-tool.sh"
ARTIFACT_ROOT="${ROOT}/reproduction/artifacts"
REFERENCE_PROJECT="/mnt/afs/L202500464/DiffPIR"
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

run_reference_python() {
  (cd "${REFERENCE_PROJECT}" && "${UV}" --proxy off uv run --no-sync python "$@")
}

run_deepinv_python() {
  (cd "${ROOT}" && "${UV}" --proxy off uv run --no-sync python "$@")
}

test -x "${ROOT}/.venv/bin/python"
test -x "${REFERENCE_PROJECT}/.venv/bin/python"
test "$(readlink -f "${ROOT}/.venv")" != "$(readlink -f "${REFERENCE_PROJECT}/.venv")"
test -d "${REFERENCE_REPO}"
test -d "${IMAGES}"
test -f "${CHECKPOINT}"
cd "${ROOT}"
export PYTHONUNBUFFERED=1

echo "[1/5] Preparing four immutable five-image SR fixtures in the DiffPIR uv environment"
for setting_id in "${SETTING_IDS[@]}"; do
  setting="${ROOT}/reproduction/diffpir/settings/${setting_id}.json"
  fixture_id="${setting_id}_separate_uv_v1"
  if [[ ! -f "${ARTIFACT_ROOT}/fixtures/${fixture_id}/manifest.json" ]]; then
    run_reference_python "${ROOT}/reproduction/diffpir/prepare_inputs.py" \
      --setting "${setting}" --fixture-id "${fixture_id}" --images "${IMAGES}" \
      --reference-repo "${REFERENCE_REPO}" --limit 5 \
      --artifact-root "${ARTIFACT_ROOT}"
  fi
done

echo "[2/5] Running the pinned original repository in the DiffPIR uv environment"
pids=()
for index in "${!SETTING_IDS[@]}"; do
  setting_id="${SETTING_IDS[index]}"
  fixture_id="${setting_id}_separate_uv_v1"
  run_reference_python "${ROOT}/reproduction/diffpir/run_reference.py" \
    --setting "${ROOT}/reproduction/diffpir/settings/${setting_id}.json" \
    --fixture-id "${fixture_id}" --run-id "${RUN_ID}" \
    --reference-repo "${REFERENCE_REPO}" --checkpoint "${CHECKPOINT}" \
    --device "${DEVICES[index]}" --artifact-root "${ARTIFACT_ROOT}" &
  pids+=("$!")
done
wait_jobs "${pids[@]}"

echo "[3/5] Recording original-repository five-image metrics in the DeepInv metric environment"
pids=()
for index in "${!SETTING_IDS[@]}"; do
  setting_id="${SETTING_IDS[index]}"
  fixture_id="${setting_id}_separate_uv_v1"
  run_deepinv_python "${ROOT}/reproduction/diffpir/evaluate_reference.py" \
    --setting "${ROOT}/reproduction/diffpir/settings/${setting_id}.json" \
    --fixture-id "${fixture_id}" --run-id "${RUN_ID}" \
    --metric-device "${DEVICES[index]}" --artifact-root "${ARTIFACT_ROOT}" &
  pids+=("$!")
done
wait_jobs "${pids[@]}"

echo "[4/5] Running DeepInv in its independent uv environment"
pids=()
for index in "${!SETTING_IDS[@]}"; do
  setting_id="${SETTING_IDS[index]}"
  fixture_id="${setting_id}_separate_uv_v1"
  run_deepinv_python "${ROOT}/reproduction/diffpir/run_deepinv.py" \
    --setting "${ROOT}/reproduction/diffpir/settings/${setting_id}.json" \
    --fixture-id "${fixture_id}" --run-id "${RUN_ID}" \
    --checkpoint "${CHECKPOINT}" --device "${DEVICES[index]}" \
    --artifact-root "${ARTIFACT_ROOT}" &
  pids+=("$!")
done
wait_jobs "${pids[@]}"

echo "[5/5] Comparing tensors/metrics and rendering figures"
for index in "${!SETTING_IDS[@]}"; do
  setting_id="${SETTING_IDS[index]}"
  fixture_id="${setting_id}_separate_uv_v1"
  run_dir="${ARTIFACT_ROOT}/runs/diffpir/${setting_id}/${RUN_ID}"
  run_deepinv_python "${ROOT}/reproduction/dps/compare.py" \
    --setting "${ROOT}/reproduction/diffpir/settings/${setting_id}.json" \
    --fixture-id "${fixture_id}" --run-id "${RUN_ID}" \
    --metric-device "${DEVICES[index]}" --artifact-root "${ARTIFACT_ROOT}" \
    --separate-uv-projects
  run_deepinv_python "${ROOT}/reproduction/diffpir/visualize_deblur.py" \
    --fixture-dir "${ARTIFACT_ROOT}/fixtures/${fixture_id}" --run-dir "${run_dir}"
done

echo "Completed one task only: bicubic SR x4"
echo "Run ID: ${RUN_ID}"
echo "Results: ${ARTIFACT_ROOT}/runs/diffpir/*/${RUN_ID}"
