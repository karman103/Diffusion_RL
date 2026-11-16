"""Sequential laptop runner; waits for unrelated compute and refuses duplicate runs."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
PY = ROOT / ".venv/bin/python"
PROTOCOL = json.loads((ROOT / "PROTOCOL.json").read_text())
DEADLINE = time.time() + 24 * 3600


def stamp():
    return datetime.now(timezone.utc).isoformat()


def log(message: str):
    line = f"{stamp()} {message}"
    print(line, flush=True)
    with (ROOT / "queue.log").open("a") as f:
        f.write(line + "\n")


def unrelated_heavy_jobs():
    p = subprocess.run(["ps", "-axo", "pid=,command="], capture_output=True, text=True, check=True)
    names = ("train_ml.py", "render_radii.py", "oracle.py", "analyse_occ.py")
    out = []
    for line in p.stdout.splitlines():
        fields = line.strip().split(maxsplit=2)
        if len(fields) < 3:
            continue
        executable, cmd = fields[1], fields[2]
        if "python" in Path(executable).name and any(name in cmd for name in names):
            out.append(line.strip())
    return out


def wait_idle():
    quiet = 0
    last = None
    while time.time() < DEADLINE:
        jobs = unrelated_heavy_jobs()
        if jobs:
            quiet = 0
            key = tuple(jobs)
            if key != last:
                log(f"Waiting for unrelated jobs: {jobs}")
            last = key
        else:
            quiet += 1
            if quiet >= 2:
                return
        time.sleep(60)
    raise TimeoutError("24-hour wait for unrelated compute expired; no new run started")


def command(args, log_path):
    with log_path.open("w") as f:
        p = subprocess.run([str(PY), *args], cwd=WORKSPACE, stdout=f, stderr=subprocess.STDOUT)
    if p.returncode:
        raise RuntimeError(f"Command failed ({p.returncode}); inspect {log_path}")


def main():
    lock = ROOT / "QUEUE_RUNNING.json"
    if lock.exists() or (ROOT / "QUEUE_DONE.json").exists():
        raise FileExistsError("Queue already running or complete; refusing duplicate")
    lock.write_text(json.dumps({"pid": os.getpid(), "started": stamp()}, indent=2))
    try:
        while not (ROOT / "checkpoints/classifier_results.json").exists():
            if time.time() >= DEADLINE:
                raise TimeoutError("Classifiers did not finish within 24 hours")
            time.sleep(60)
        checks = json.loads((ROOT / "checkpoints/classifier_results.json").read_text())
        if checks["proxy"]["test_accuracy"] < 0.40 or checks["gold"]["test_accuracy"] < 0.60:
            raise RuntimeError("Image classifier quality gate failed")
        log(f"Classifier gate passed: {checks}")
        if not (ROOT / "VIDEO_GATE_FAILED.json").exists():
            raise RuntimeError("Video feasibility decision missing; inspect before running queue")
        if (ROOT / "PREFLIGHT_FAILED.json").exists():
            raise RuntimeError("Preflight previously failed; inspect before any training")
        if not (ROOT / "PREFLIGHT.json").exists():
            wait_idle()
            log("Running real-checkpoint MPS identity and finite-sample preflight")
            command(["-m", "image_video_grpo.preflight"], ROOT / "preflight.log")
        for seed in PROTOCOL["image"]["training_seeds"]:
            run = ROOT / "runs/image" / f"seed_{seed}"
            if not (run / "DONE.json").exists():
                if run.exists():
                    raise RuntimeError(f"Incomplete run exists: {run}; do not duplicate")
                wait_idle()
                log(f"Starting image training seed {seed}")
                command(["-m", "image_video_grpo.experiment", "train", "--kind", "image", "--seed", str(seed)], ROOT / f"train_image_{seed}.log")
                if not (run / "DONE.json").exists():
                    raise RuntimeError(f"Training did not write completion marker: {run}")
            if not (run / "EVAL_DONE.json").exists():
                if (run / "eval.jsonl").exists():
                    raise RuntimeError(f"Incomplete evaluation exists: {run}")
                wait_idle()
                log(f"Starting image evaluation seed {seed}")
                command(["-m", "image_video_grpo.experiment", "eval", "--kind", "image", "--seed", str(seed)], ROOT / f"eval_image_{seed}.log")
                if not (run / "EVAL_DONE.json").exists():
                    raise RuntimeError(f"Evaluation did not write completion marker: {run}")
        log("Image runs complete; generating measured report")
        command(["-m", "image_video_grpo.analyze"], ROOT / "analyze.log")
        (ROOT / "QUEUE_DONE.json").write_text(json.dumps({"completed": stamp()}, indent=2))
    except BaseException as exc:
        (ROOT / "QUEUE_FAILED.json").write_text(json.dumps({"failed": stamp(), "error": repr(exc)}, indent=2))
        log(f"FAILED: {exc!r}")
        raise
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
