"""Bounded, resumable, sequential experiment queue. Never overwrites a run."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .core import ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--deadline", default="2026-11-13T10:26:00Z")
    parser.add_argument("--wait-pid", type=int)
    parser.add_argument("--manifest", type=Path, default=ROOT / "protein_poc/queue.json")
    a = parser.parse_args()
    root = ROOT / "protein_poc"
    lock = (root / "queue.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    lock.seek(0); lock.truncate()
    lock.write(str(os.getpid())); lock.flush()
    deadline = datetime.fromisoformat(a.deadline.replace("Z", "+00:00"))
    if a.wait_pid:
        while datetime.now(timezone.utc) < deadline:
            try:
                os.kill(a.wait_pid, 0)
            except ProcessLookupError:
                break
            time.sleep(10)
    manifest = json.loads(a.manifest.read_text())
    history = (root / "queue_history.jsonl").open("a", buffering=1)
    for entry in manifest:
        if datetime.now(timezone.utc) >= deadline:
            break
        name, args = entry["name"], entry["args"]
        if Path(name).name != name:
            raise ValueError("Run names must be plain directory names")
        out = root / "runs" / name
        if out.exists():
            print(f"SKIP existing {name}: inspect DONE.json or failure log", flush=True)
            continue
        # Evaluation must not silently use an unfinished training checkpoint.
        if "--head" in args:
            head = ROOT / args[args.index("--head")+1]
            done_file = head.parent / "DONE.json"
            if not done_file.exists() or not json.loads(done_file.read_text()).get("complete"):
                history.write(json.dumps(dict(name=name, returncode="skipped: source training incomplete"))+"\n")
                continue
        cmd = [sys.executable, "-u", "-m", "protein_poc.experiment", "--out", str(out),
               "--device", "mps", "--deadline", a.deadline] + args
        started = datetime.now(timezone.utc).isoformat()
        print(f"START {name} {started}", flush=True)
        with (root / f"{name}.log").open("w") as log:
            # A deadline bounds even a stalled subprocess, with time for cleanup.
            remaining = max(1, (deadline-datetime.now(timezone.utc)).total_seconds())
            try:
                status = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                        timeout=remaining).returncode
            except subprocess.TimeoutExpired:
                status = "deadline"
        history.write(json.dumps(dict(name=name, started_utc=started, returncode=status,
                                      finished_utc=datetime.now(timezone.utc).isoformat()))+"\n")
        print(f"END {name}: {status}", flush=True)
        # Refresh only from artifacts on disk. Failures remain visible in history.
        subprocess.run([sys.executable, "-m", "protein_poc.report"], cwd=ROOT, check=False)
        if status == "deadline":
            break
    subprocess.run([sys.executable, "-m", "protein_poc.report"], cwd=ROOT, check=False)
    (root / "QUEUE_FINISHED.json").write_text(json.dumps({
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "deadline": a.deadline,
        "note": "Queue ended; inspect individual DONE.json and history for actual completion."
    }, indent=2)+"\n")


if __name__ == "__main__":
    main()
