"""Resume the official checkpoint with bounded HTTP range requests, one output file."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import json
import os
from pathlib import Path
import threading
import time
import urllib.request

ROOT = Path(__file__).resolve().parent
URL = "https://files.ipd.uw.edu/pub/rfd3/rfd3_foundry_2025_12_01_remapped.ckpt"


def main():
    target = ROOT/"checkpoints/rfd3_latest.ckpt"
    state_path = target.with_suffix(".download.json")
    lock = target.with_suffix(".lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    with urllib.request.urlopen(urllib.request.Request(URL, method="HEAD"),timeout=60) as response:
        size = int(response.headers["Content-Length"])
        etag = response.headers.get("ETag")
    block = 32*1024*1024
    if state_path.exists():
        state = json.loads(state_path.read_text())
        assert state["size"]==size and state["etag"]==etag
    else:
        prefix = target.stat().st_size if target.exists() else 0
        assert prefix <= size
        state = dict(url=URL,size=size,etag=etag,block=block,
                     completed=list(range(prefix//block)))
    done = set(state["completed"])
    mutex = threading.Lock()
    fd = os.open(target,os.O_CREAT|os.O_RDWR,0o644)
    os.ftruncate(fd,size)
    total = (size+block-1)//block

    def save():
        state["completed"] = sorted(done)
        tmp = state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state,indent=2)+"\n"); tmp.replace(state_path)

    save()
    def fetch(index):
        start,end = index*block,min(size,(index+1)*block)-1
        for attempt in range(4):
            try:
                req = urllib.request.Request(URL,headers={"Range":f"bytes={start}-{end}","If-Range":etag})
                with urllib.request.urlopen(req,timeout=120) as response:
                    assert response.status==206, "Server did not honor byte range"
                    assert response.headers["Content-Range"]==f"bytes {start}-{end}/{size}"
                    offset = start
                    while chunk := response.read(1024*1024):
                        assert offset+len(chunk)<=end+1
                        view = memoryview(chunk)
                        while view:
                            written = os.pwrite(fd,view,offset)
                            offset += written; view = view[written:]
                    assert offset==end+1
                with mutex:
                    done.add(index); save()
                    print(f"Checkpoint blocks {len(done)}/{total}",flush=True)
                return
            except Exception:
                if attempt==3: raise
                time.sleep(2*(attempt+1))
    try:
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(fetch,i) for i in range(total) if i not in done]
            for future in as_completed(futures): future.result()
        os.fsync(fd)
        assert len(done)==total
        print("Official checkpoint download complete",flush=True)
    finally:
        os.close(fd)


if __name__ == "__main__":
    main()
