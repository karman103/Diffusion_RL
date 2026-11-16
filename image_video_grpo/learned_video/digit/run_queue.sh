#!/bin/zsh
set -e
cd "$(dirname "$0")/../../.."
PY=image_video_grpo/.venv/bin/python
M=image_video_grpo.learned_video.digit
R=image_video_grpo/learned_video/digit/runs
until [ -f $R/pretrain_done.json ]; do sleep 30; done
$PY -m $M.gate
if ! grep -q '"pass": true' $R/gate.json; then echo "GATE FAILED - stopping"; exit 0; fi
for n in pretrained cfg2 best4; do $PY -m $M.grpo eval --name $n; done
for s in 3101 3102 3103; do $PY -m $M.grpo train --seed $s; $PY -m $M.grpo eval --name grpo_$s; done
$PY -m $M.analyze
echo "QUEUE DONE"
