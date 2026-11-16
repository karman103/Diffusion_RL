"""Recompute reported geometry from saved coordinates; read-only provenance audit."""
import json
from pathlib import Path

import numpy as np
import torch

from .core import geometry, reward, distance_diversity
from .report import read_rows


def main():
    torch.set_num_threads(2)
    count = 0
    for folder in sorted((Path(__file__).parent/"runs").glob("*")):
        if not (folder/"config.json").exists():
            continue
        cfg = json.loads((folder/"config.json").read_text())
        for row in read_rows(folder/"metrics.jsonl"):
            path = folder / f"eval_{row['epoch']:04d}.npz"
            if not path.exists():
                continue  # active run may still be saving an older-code artifact
            with np.load(path, allow_pickle=False) as saved:
                x = torch.from_numpy(saved["coordinates"])
                assert np.isfinite(saved["coordinates"]).all(), path
                recomputed = geometry(x)
                recomputed["reward"] = reward(x,"compact" if cfg["objective"]=="random" else cfg["objective"])
                for key, value in recomputed.items():
                    np.testing.assert_allclose(value.numpy(),saved[key],rtol=1e-5,atol=1e-6,err_msg=f"{path}: {key}")
                    np.testing.assert_allclose(value.mean().item(),row[key],rtol=1e-5,atol=1e-6)
                np.testing.assert_allclose(distance_diversity(x),row["diversity"],rtol=1e-5,atol=1e-6)
                assert len(x)==row["n"]
            count += 1
    print(f"PASS: recomputed geometry, reward and diversity for {count} saved evaluations")


if __name__ == "__main__":
    main()
