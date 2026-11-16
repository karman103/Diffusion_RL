#!/bin/bash
# User-directed deferral of the neural-shading GPU queue. This waits for an
# explicit release marker from this study; it never runs concurrently with
# text-to-video work or another neural-shading queue.
set -u
here="$(cd "$(dirname "$0")" && pwd)"
release="$here/NEURAL_SHADING_RELEASE"
project="/Users/karman/Desktop/projects/research/neural-shading-transformer"
log="$here/deferred_neural_shading.log"
printf '%s waiting for video-study release marker\n' "$(date -u +%FT%TZ)" >> "$log"
while [ ! -f "$release" ]; do sleep 60; done
while pgrep -f 'text_video_grpo/(preflight_zeroscope|adapter_backward_probe|train_video_grpo)' >/dev/null; do sleep 60; done
if pgrep -f '/bin/bash ./run_r7.sh|/bin/bash ./run_sweep.sh|/bin/bash ./resume_queue.sh' >/dev/null; then
  printf '%s another neural-shading queue is already active; no duplicate started\n' "$(date -u +%FT%TZ)" >> "$log"
  exit 0
fi
cd "$project" || exit 1
printf '%s restarting resumable round 7\n' "$(date -u +%FT%TZ)" >> "$log"
caffeinate -dims -w $$ &
bash ./run_r7.sh >> "$project/run_r7.log" 2>&1
printf '%s round 7 finished; restarting sweep\n' "$(date -u +%FT%TZ)" >> "$log"
bash ./run_sweep.sh >> "$project/run_sweep.log" 2>&1
printf '%s deferred queue finished\n' "$(date -u +%FT%TZ)" >> "$log"
