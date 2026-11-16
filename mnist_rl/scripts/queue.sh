#!/bin/bash
# Sequential job queue: runs each line of runs/queue.txt once (tracked in runs/queue_done.txt).
cd "$(dirname "$0")/.." && source .venv/bin/activate
touch runs/queue_done.txt
while true; do
  next=$(grep -v -x -F -f runs/queue_done.txt runs/queue.txt | grep -v '^#' | grep -v '^$' | head -1)
  [ -z "$next" ] && { echo "queue empty $(date)"; break; }
  echo "=== START $(date +%H:%M:%S) :: $next"
  bash -c "$next"; rc=$?
  echo "=== END   $(date +%H:%M:%S) rc=$rc :: $next"
  echo "$next" >> runs/queue_done.txt
done
