#!/bin/zsh
# Runs after the digit-conditioned video queue; then re-releases the deferred neural-shading queue.
set -e
cd "$(dirname "$0")/../.."
PY=image_video_grpo/.venv/bin/python
M=image_video_grpo.cifar_full
while pgrep -f "digit/run_queue" >/dev/null; do sleep 60; done
for n in pretrained best4; do $PY -m $M.grpo eval --name $n; done
for s in 4101 4102 4103; do $PY -m $M.grpo train --seed $s; $PY -m $M.grpo eval --name grpo_$s; done
$PY -m $M.analyze
echo "released after GRPO video+CIFAR fix queues finished" > text_video_grpo/NEURAL_SHADING_RELEASE
echo "QUEUE DONE"
