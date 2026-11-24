"""Render ranked evidence from one original video and validate it using the existing Qwen loader."""
import argparse
import csv
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUN = HERE.parents[1] / "run"
sys.dont_write_bytecode = True
sys.path.insert(0, str(RUN))
from infer_mlx import load_manifest, prompt_for


def prepare(source, video, output):
    rows = [json.loads(line) for line in (HERE / "windows.jsonl").read_text().splitlines()]
    item = next(r for r in rows if r["video_id"] == video)
    with (HERE.parent / "media_checks.csv").open() as stream:
        expected = next(r["sha256"] for r in csv.DictReader(stream) if r["video_id"] == video)
    with source.open("rb") as stream:
        checksum = hashlib.file_digest(stream, "sha256").hexdigest()
    if checksum != expected:
        raise ValueError("Original media SHA-256 differs from the fixed source.")
    frames = output / "frames"
    frames.mkdir(parents=True, exist_ok=True)
    costs = {}
    decoded = set()
    begin = time.perf_counter()
    for arm, windows in item["methods"].items():
        segments = []
        for window in sorted(windows, key=lambda w: w["interval_s"][0]):
            lo, hi = window["interval_s"]
            images = []
            for offset in (1.5, 4.5, 7.5):
                sec = round(lo + offset, 3)
                relative = Path("frames") / f"{video}-{round(1000 * sec):08}.png"
                target = output / relative
                if sec not in decoded:
                    subprocess.run([sys.executable, str(RUN / "extract_frame.py"),
                                    str(source), str(sec), str(target)], check=True)
                    decoded.add(sec)
                images.append({"path": relative.as_posix(), "time_s": sec})
            segments.append({"interval_s": [lo, hi], "sound_description": window["sound_description"],
                             "frames": images})
        if not segments:
            costs[arm] = {"windows": 0, "status": "no_evidence", "frame_count": 0}
            continue
        # Only test image/evidence ingestion. Wording and options are fixed, source-independent.
        row = {"pair_id": f"ingestion-{video}", "qid": "probe-first-window",
               "question": "Is a person visible in the final image of the first supplied time window?",
               "options": {"present": "A person is visible.", "absent": "No person is visible."},
               "segments": segments}
        manifest = output / (arm + ".jsonl")
        manifest.write_text(json.dumps(row) + "\n")
        validated, = load_manifest(manifest)
        flat, images, options, events = prompt_for(validated, "flat")
        paired, p_images, p_options, p_events = prompt_for(validated, "paired")
        if (images, options, events) != (p_images, p_options, p_events):
            raise ValueError("Flat and paired conditions did not load identical evidence.")
        if (len(images) != 3 * len(windows) or len(events) != len(windows)
                or "source_qid" in flat or "hint_minute" in paired):
            raise ValueError("Unexpected runner prompt or image count.")
        costs[arm] = {"windows":len(segments), "frames":len(images), "status":"runner_input_validated"}
    metadata = {"video_id":video, "source_sha256":checksum, "frame_offsets_in_window_s":[1.5, 4.5, 7.5],
                "n_distinct_images":len(decoded), "image_space":"448x448 RGB PNG, letterboxed",
                "conditions":costs, "extract_and_validate_seconds":round(time.perf_counter()-begin,3),
                "qwen_generation":"未跑；此处只检验输入和提示构造"}
    (output / "manifest_check.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", type=Path)
    p.add_argument("--video-id", default="video530")
    p.add_argument("--output", type=Path, required=True, help="Use a new directory; frames are never overwritten.")
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    prepare(a.source, a.video_id, a.output)
