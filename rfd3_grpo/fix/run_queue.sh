#!/bin/zsh
set -e
cd "$(dirname "$0")/../.."
PY=.venv-rfd3/bin/python
export RFD3_ACTION_ATOMS=all
for s in 21 22; do
  R=rfd3_grpo/fix/runs/seed_$s
  if [ ! -f $R/DONE.json ]; then
    rm -rf $R
    $PY -u -m rfd3_grpo.train --out $R --length 32 --steps 64 --epochs 60 --group-size 4 --eval-n 4 \
      --lr 0.001 --rank 3 --active-min 0.0 --active-max 4.0 --seed $s --deadline 2026-12-31T00:00:00Z > rfd3_grpo/fix/train_$s.log 2>&1
  fi
  $PY -u -m rfd3_grpo.fix.evaluate --run seed_$s > rfd3_grpo/fix/eval_$s.log 2>&1
done
echo "QUEUE DONE"
