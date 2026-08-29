#!/usr/bin/env bash
set -euo pipefail

# Formal test-time attention intervention matrix.  The existing cluster image is
# reused; the Qwen3-ASR conda environment and all inputs live on shared storage.
REPO_DIR="${REPO_DIR:-/hpc_stor03/sjtu_home/xiaoyu.gu/workspace/wcc_workspace/ms-swift}"
DATA_DIR="${DATA_DIR:-/hpc_stor03/sjtu_home/xiaoyu.gu/workspace/wcc_workspace/slidespeech}"
MODEL_DIR="${MODEL_DIR:-/hpc_stor03/sjtu_home/xiaoyu.gu/workspace/data-prepare/model/Qwen3-ASR-1.7B}"
PYTHON_BIN="${PYTHON_BIN:-/hpc_stor03/sjtu_home/xiaoyu.gu/miniconda3/envs/qwen3-asr/bin/python}"
IMAGE="${IMAGE:-docker.v2.aispeech.com/sjtu/sjtu_yukai-grace-vcg:v0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_DIR}/experiments/attention_intervention/formal}"
NUM_SHARDS="${NUM_SHARDS:-8}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_BASELINE_SHARD0="${SKIP_BASELINE_SHARD0:-0}"
SHARD_START="${SHARD_START:-0}"
SHARD_END="${SHARD_END:-$((NUM_SHARDS - 1))}"
SKIP_LOCAL_MATCHED="${SKIP_LOCAL_MATCHED:-0}"

manifest="${DATA_DIR}/test_oracle_v1/history/multitask.jsonl"
path_prefix_to="${DATA_DIR}"

# Single-GPU jobs on the other queues are automatically redirected to minijob
# partitions, whose low-utilization watchdog interrupts this autoregressive
# workload.  The regular 4090 queue is verified to complete full shards.
partitions=(
  pdgpu-4090 pdgpu-4090 pdgpu-4090 pdgpu-4090
  pdgpu-4090 pdgpu-4090 pdgpu-4090 pdgpu-4090
)

# name|strategy|alpha
groups=(
  "baseline|baseline|0"
  "correct-a025|correct_teacher|0.25"
  "correct-a050|correct_teacher|0.5"
  "correct-a100|correct_teacher|1.0"
  "correct-a200|correct_teacher|2.0"
  "shuffled-a025|shuffled_teacher|0.25"
  "shuffled-a050|shuffled_teacher|0.5"
  "shuffled-a100|shuffled_teacher|1.0"
  "shuffled-a200|shuffled_teacher|2.0"
  "zero-context|zero_context|1.0"
  "matched-random|matched_mass_random|1.0"
)

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY

for group in "${groups[@]}"; do
  IFS='|' read -r group_name strategy alpha <<<"${group}"
  for ((shard = SHARD_START; shard <= SHARD_END; shard++)); do
    if [[ "${SKIP_BASELINE_SHARD0}" == "1" && "${group_name}" == "baseline" && "${shard}" == "0" ]]; then
      continue
    fi
    if [[ "${SKIP_LOCAL_MATCHED}" == "1" && "${group_name}" == "matched-random" && "${shard}" -ge 5 ]]; then
      continue
    fi
    partition="${partitions[$((shard % ${#partitions[@]}))]}"
    output="${OUTPUT_ROOT}/${group_name}/part-$(printf '%02d' "${shard}").pred"
    job_name="q3attn-${group_name//./}-${shard}"
    job_cmd="mkdir -p ${OUTPUT_ROOT}/${group_name} && cd ${REPO_DIR} && TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONPATH=. ${PYTHON_BIN} -m examples.custom.qwen3_asr_attention_intervention --manifest ${manifest} --model-path ${MODEL_DIR} --output ${output} --strategy ${strategy} --alpha ${alpha} --layers 14 --batch-size 4 --max-new-tokens 256 --shard-index ${shard} --num-shards ${NUM_SHARDS} --seed 20260830 --path-prefix-to ${path_prefix_to}"

    printf '%s\t%s\t%s\t%s\n' "${job_name}" "${partition}" "${strategy}" "${alpha}"
    if [[ "${DRY_RUN}" == "1" ]]; then
      continue
    fi
    vc submit \
      --partition "${partition}" \
      --image "${IMAGE}" \
      --num-task 1 \
      --cpu-per-task 8 \
      --mem-per-task 32G \
      --gpu-per-task 1 \
      --dir "${REPO_DIR}" \
      --job "${job_name}" \
      --cmd "${job_cmd}"
  done
done
