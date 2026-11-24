"""Rank full-track acoustic regions and construct uniform controls without annotation hints."""
import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE.parent
K = 4
WIDTH = 10.0
BEFORE_END = 4.0


def read_csv(path):
    with path.open(encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def support(anchor, duration):
    start = round(max(0.0, min(anchor - BEFORE_END, duration - WIDTH)), 3)
    return [start, round(start + WIDTH, 3)]


def uniform(count, duration):
    return [support((i + 0.5) * duration / count, duration) for i in range(count)]


def describe(interval, leads):
    # A common renderer for every arm: only report target-family detections
    # with positive overlap in the displayed interval.
    inside = [lead for lead in leads if lead["start"] < interval[1] and lead["end"] > interval[0]]
    if not inside:
        return "No target-family sound exceeded the detector threshold in this window."
    peak = max(inside, key=lambda x: (x["score"], -x["start"]))
    label = peak["label"].replace("_", " ")
    return f"Sound classifier label: {label}."


def calculate():
    media = read_csv(DATA / "media_checks.csv")
    records = read_csv(DATA / "audio_leads.csv")
    output = []
    for source in media:
        video = source["video_id"]
        duration = float(source["duration_sec"])
        observations = [r for r in records if r["video_id"] == video]
        if not observations or any(r["status"] == "未跑" for r in observations):
            raise ValueError(f"{video}: missing full-track detector result")
        if len({r["cue"] for r in observations}) != 1:
            raise ValueError(f"{video}: expected one fixed cue per video")
        leads = [
            {"start": float(row["start_sec"]), "end": float(row["end_sec"]),
             "score": float(row["peak_confidence"]), "label": row["peak_label"]}
            for row in observations if row["status"] == "机器候选声音"
        ]
        # Source-side overlapping detector windows were already merged in summarize_audio.py.
        # Nonmaximum suppression operates on the actual ten-second downstream windows.
        ranked = sorted(leads, key=lambda x: (-x["score"], x["start"], x["end"], x["label"]))
        acoustic = []
        for lead in ranked:
            interval = support(lead["end"], duration)
            if any(interval[0] < seg["interval_s"][1] and interval[1] > seg["interval_s"][0]
                   for seg in acoustic):
                continue
            acoustic.append({"rank": len(acoustic) + 1, "interval_s": interval,
                             "detector_interval_s": [lead["start"], lead["end"]],
                             "peak_label": lead["label"], "peak_confidence": lead["score"]})
            if len(acoustic) == K:
                break
        methods = {"acoustic_top4": acoustic}
        for name, count in (("uniform4", K), ("uniform_matched", len(acoustic))):
            methods[name] = [{"rank": n, "interval_s": bounds}
                                for n, bounds in enumerate(uniform(count, duration), 1)]
        for windows in methods.values():
            for window in windows:
                window["sound_description"] = describe(window["interval_s"], leads)
        output.append({"video_id": video, "media_id": source["media_id"],
                       "cue": observations[0]["cue"], "duration_s": duration,
                       "duration_source": "media_checks.csv", "raw_regions": len(leads),
                       "acoustic_status": "has_candidates" if acoustic else "no_candidate_over_threshold",
                       "methods": methods})
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=True, parents=True)
    start = time.perf_counter()
    rows = calculate()
    elapsed = time.perf_counter() - start
    out = args.output / "windows.jsonl"
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    keys = ["video_id", "cue", "duration_s", "raw_regions", "acoustic_top4", "uniform4",
            "uniform_matched", "acoustic_status"]
    with (args.output / "counts.csv").open("w") as dst:
        writer = csv.DictWriter(dst, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: len(row["methods"][k]) if k in row["methods"] else row[k] for k in keys})
    inputs = [DATA / "media_checks.csv", DATA / "audio_leads.csv", DATA / "summarize_audio.py"]
    report = {"input_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
              "rank_sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
              "ranking_wall_seconds": round(elapsed, 6),
              "detector_regions": sum(r["raw_regions"] for r in rows),
              "videos_with_two_or_more_acoustic_candidates": sum(len(r["methods"]["acoustic_top4"]) >= 2 for r in rows),
              "counts": {m: sum(len(r["methods"][m]) for r in rows) for m in rows[0]["methods"]},
              "reference_pairs": 0, "recall_at_4": None, "recall_status": "未跑"}
    (args.output / "run.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
