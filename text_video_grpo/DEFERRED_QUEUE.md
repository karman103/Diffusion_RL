# Deferred neural-shading queue

On 2026-11-15 the user explicitly asked to stop the other GPU run, prioritize the text-to-video study, and put the other work back in queue. The other project is `/Users/karman/Desktop/projects/research/neural-shading-transformer`.

Before stopping, `resume_queue.sh` had launched `run_r7.sh`, `run_sweep.sh`, and `train_ml.py --variant g64 --radius 4 --seed 1 --out runs_ml/g64_r4_s1`. `train_ml.py` writes an atomic `last.pt` every 1000 steps and resumes its model, optimizer, scheduler, RNG, and elapsed time from that file. Existing `done` markers are skipped by `run_r7.sh` and `run_sweep.sh`. The current interrupted run's `last.pt` existed and must be preserved. No artifacts in that project should be deleted.

`resume_neural_shading_after_video.sh` waits for `NEURAL_SHADING_RELEASE`, then checks for active video or neural-shading jobs and restarts round 7 followed by the sweep. The text-to-video heartbeat should create the release marker **only once the video study has finished or has a terminal blocker** and should verify this queue resumes. The wrapper and log are in this directory. Do not release the queue immediately after only a stock sample if more video GPU work is planned.
