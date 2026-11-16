"""Sequential RFD3 GRPO experiments, requiring a successful real-model smoke."""
import argparse
from datetime import datetime,timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
TASK=ROOT/"rfd3_grpo"


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--smoke",type=Path,default=TASK/"runs/smoke_v3")
    p.add_argument("--calibration",type=Path,default=TASK/"runs/calibration64")
    p.add_argument("--deadline",default="2026-11-13T10:26:00Z")
    a=p.parse_args()
    deadline=datetime.fromisoformat(a.deadline.replace("Z","+00:00"))
    lock=(TASK/"queue.lock").open("a+")
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    lock.seek(0);lock.truncate();lock.write(str(os.getpid()));lock.flush()
    done=json.loads((a.smoke/"DONE.json").read_text())
    if not done.get("complete"):
        raise RuntimeError("Real RFD3 smoke must finish successfully before the queue starts")
    rows=[json.loads(line) for line in (a.smoke/"training.jsonl").read_text().splitlines()]
    checks=json.loads((a.smoke/"checks.json").read_text())
    if not rows or rows[-1]["adapter_delta_norm"]<=0 or rows[-1]["frozen_base_gradient_count"]!=0:
        raise RuntimeError("Smoke did not demonstrate the intended adapter update")
    assert "stock_sampler_parity_max_error" in checks
    assert json.loads((a.calibration/"DONE.json").read_text()).get("complete")
    calibration=[json.loads(line) for line in (a.calibration/"metrics.jsonl").read_text().splitlines()]
    if len(calibration)<2 or any(m["bond_mae"]>.25 or m["bond_ok"]<.9 or m["clash_fraction"]>.01 for m in calibration):
        raise RuntimeError("Calibration geometry failed the declared operational gate")
    env=os.environ.copy()
    env.update(PYTORCH_MPS_HIGH_WATERMARK_RATIO="0.8",PYTORCH_MPS_LOW_WATERMARK_RATIO="0.6",
               WANDB_MODE="disabled",XDG_CACHE_HOME=str(TASK/".cache"))
    history=(TASK/"queue_history.jsonl").open("a",buffering=1)
    for entry in json.loads((TASK/"queue.json").read_text()):
        if datetime.now(timezone.utc)>=deadline:break
        name=entry["name"]
        if Path(name).name!=name:raise ValueError(name)
        out=TASK/"runs"/name
        if out.exists():
            print(f"SKIP existing run {name}",flush=True);continue
        cmd=[sys.executable,"-u","-m","rfd3_grpo.train","--out",str(out),
             "--deadline",a.deadline,*entry["args"]]
        print(f"START {name} {datetime.now(timezone.utc).isoformat()}",flush=True)
        remaining=max(1,(deadline-datetime.now(timezone.utc)).total_seconds())
        with (TASK/f"{name}.log").open("w") as log:
            try:
                code=subprocess.run(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,
                                    timeout=remaining).returncode
            except subprocess.TimeoutExpired:code="deadline"
        history.write(json.dumps(dict(name=name,returncode=code,
                                     finished_utc=datetime.now(timezone.utc).isoformat()))+"\n")
        print(f"END {name}: {code}",flush=True)
        subprocess.run([str(ROOT/".venv/bin/python"),"-m","rfd3_grpo.report"],cwd=ROOT,check=False)
        if code!=0:
            print("Stopped queue for failure review; preserving evidence",flush=True)
            break
    (TASK/"QUEUE_FINISHED.json").write_text(json.dumps(dict(
        finished_utc=datetime.now(timezone.utc).isoformat(),deadline=a.deadline),indent=2)+"\n")


if __name__=="__main__":main()
