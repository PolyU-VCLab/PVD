#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
TASK=${TASK:?Set TASK to sd35, flux or qwenimage}
PART=${PART:-1}
GPUS=${GPUS:-1}
case "$PART" in
  1) INTERVAL=1.0_0.6 ;;
  2) INTERVAL=0.6_0.0 ;;
  *) echo 'PART must be 1 or 2' >&2; exit 2 ;;
esac
COMMON=(--data-file "${DATA_FILE:?Set DATA_FILE to the BLIP3o JSONL file}"
  --image-root "${IMAGE_ROOT:?Set IMAGE_ROOT to the BLIP3o image directory}"
  --results-dir "outputs/${TASK}/part${PART}" --distill --image-size 1024
  --intervals "$INTERVAL" --global-batch-size "$GPUS" --accumulation 4
  --eval-every -1 --ckpt-every 1000 --lr 0.00001 --disc-lr 0.00001)
case "$TASK" in
  sd35)
    export INITIALIZATION_CHECKPOINT="${INITIALIZATION_CHECKPOINT:-weights/pvd_sd35m/part${PART}.safetensors}"
    ENTRY=train_t2i_sd3.py
    EXTRA=(--cfgw 7.0)
    ;;
  flux)
    ENTRY=train_t2i_flux.py
    EXTRA=(--cfgw 3.5
      --peft-adapter-name "part${PART}" --peft-adapter-path "weights/pvd_flux/part${PART}" --init-ckpt '')
    ;;
  qwenimage)
    ENTRY=train_t2i_qwenimage.py
    EXTRA=(--use-lora --lora-r 32 --lora-alpha 64 --lora-bias none
      --pretrained-transformer-path weights/pvd_qwenimage/backbone.safetensors)
    ;;
  *) echo 'TASK must be sd35, flux or qwenimage' >&2; exit 2 ;;
esac
accelerate launch --num_processes "$GPUS" --mixed_precision bf16 "$ENTRY" "${COMMON[@]}" "${EXTRA[@]}" "$@"
