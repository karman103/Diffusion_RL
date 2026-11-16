"""Apply the prospectively declared native-video baseline quality gate."""
from __future__ import annotations

import json

from image_video_grpo.video_revisit.preflight import CFG, ROOT


def main():
    rows = []
    for seed in CFG["seeds"]:
        path = ROOT / f"seed_{seed}" / "RESULT.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing native baseline sample: {seed}")
        rows.append(json.loads(path.read_text()))
    passed = [bool(row["gold_digit_confidence"] >= 0.5
                   and 0.01 < row["ink_mass"] < 0.35
                   and row["frame_delta"] > 0.005) for row in rows]
    result = {"seeds": CFG["seeds"], "rows": rows, "passed_per_clip": passed,
              "passed_count": sum(passed), "automatic_gate_passed": sum(passed) >= 3,
              "visual_review_required": True,
              "video_grpo_started": False}
    (ROOT / "GATE.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
