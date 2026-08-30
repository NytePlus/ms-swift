#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/hpc_stor03/sjtu_home/xiaoyu.gu/workspace/wcc_workspace/ms-swift}"
COMPAT_ROOT="${COMPAT_ROOT:-${REPO_DIR}/experiments/attention_intervention/score_compat}"
SCORER_ROOT="${SCORER_ROOT:-/hpc_stor03/sjtu_home/chencheng.wang/workspace/ContextAudioTestPipeline}"
IMAGE="${IMAGE:-wcc/ms-swift:qwen3-asr-v4.5.2-cuda12.8}"
SCORE_DEPS="${SCORE_DEPS:-${COMPAT_ROOT}/score_deps}"
PYTHON_BIN_DIR="${PYTHON_BIN_DIR:-/hpc_stor03/sjtu_home/xiaoyu.gu/miniconda3/bin}"

if [[ "$#" -eq 0 ]]; then
  groups=(baseline correct-a025 correct-a050 correct-a100 correct-a200 shuffled-a025 shuffled-a050 shuffled-a100 shuffled-a200 zero-context matched-random)
else
  groups=("$@")
fi

for group in "${groups[@]}"; do
  pipeline_root="${COMPAT_ROOT}/${group}/pipeline"
  docker run --rm \
    --volume "${COMPAT_ROOT}:${COMPAT_ROOT}" \
    --volume "${COMPAT_ROOT}/data:/data" \
    --volume "${SCORER_ROOT}:${SCORER_ROOT}:ro" \
    --volume "${SCORE_DEPS}:/opt/score_deps:ro" \
    --env "HOME=${COMPAT_ROOT}/home" \
    --env "PYTHONPATH=/opt/score_deps" \
    --env "PATH=${PYTHON_BIN_DIR}:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
    --workdir "${pipeline_root}" \
    --entrypoint bash \
    "${IMAGE}" \
    "${SCORER_ROOT}/score/score_old.sh"
  printf '%s\t' "${group}"
  grep -E '^(WER|U-WER|B-WER):' "${pipeline_root}/score/results/qwen_asr_pred/dev/dev_qwen_asr_pred_norm_wer" | tr '\n' '\t'
  printf '\n'
done
