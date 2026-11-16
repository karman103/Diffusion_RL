#!/bin/zsh
# Sequential GPU queue: gate -> (only if pass) controls + 3 GRPO seeds with paired eval.
set -e
cd "$(dirname "$0")/../.."
PY=image_video_grpo/.venv/bin/python
M=image_video_grpo.learned_video
until [ -f image_video_grpo/learned_video/runs/pretrain_done.json ]; do sleep 30; done
$PY -m $M.gate
if ! grep -q '"pass": true' image_video_grpo/learned_video/runs/gate.json; then echo "GATE FAILED - stopping"; exit 0; fi
for n in pretrained cfg2 best4; do $PY -m $M.grpo eval --name $n; done
for s in 2101 2102 2103; do
  $PY -m $M.grpo train --seed $s
  $PY -m $M.grpo eval --name grpo_$s
done
echo "QUEUE DONE"
