"""Launch the real-model smoke immediately after the verified checkpoint arrives."""
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
status=ROOT/"rfd3_grpo/checkpoints/rfd3_latest.download.json"
deadline=datetime.fromisoformat("2026-11-13T10:26:00+00:00")
os.environ.setdefault("WANDB_MODE","disabled")
os.environ.setdefault("XDG_CACHE_HOME",str(ROOT/"rfd3_grpo/.cache"))
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO","0.8")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO","0.6")
while datetime.now(timezone.utc)<deadline:
    try:
        state=json.loads(status.read_text())
        total=(state["size"]+state["block"]-1)//state["block"]
        if len(set(state["completed"]))==total:
            break
    except (OSError,json.JSONDecodeError):
        pass
    time.sleep(10)
else:
    raise RuntimeError("Deadline reached before checkpoint finished")
cmd=[sys.executable,"-u","-m","rfd3_grpo.train","--out",str(ROOT/"rfd3_grpo/runs/smoke"),
     "--steps","8","--length","16","--epochs","1","--group-size","2","--eval-n","2",
     "--lr","0.0001","--check-stock-parity","--evaluate-stock"]
print("Checkpoint complete; launching actual pretrained RFD3 GRPO smoke",flush=True)
print(" ".join(cmd),flush=True)
os.chdir(ROOT)
os.execv(sys.executable,cmd)
