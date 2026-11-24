#!/usr/bin/env python3
"""Prepare a conditional smoke input from one completed EPIC review pair."""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--media-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pair", help="If omitted, use the first completed candidate in queue order")
    args = parser.parse_args()
    media = {
        (row["pair_id"], row["ordinal"]): row
        for row in read_jsonl(args.media_manifest) if "path" in row
    }
    pair = next((
        row for row in read_jsonl(args.queue)
        if (args.pair is None or args.pair == row["pair_id"])
        and all((row["pair_id"], i) in media for i in (1, 2))
    ), None)
    if pair is None:
        parser.exit(1, "No complete candidate pair with two downloadable review windows.\n")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    extract = Path(__file__).with_name("extract_frame.py")
    pair_id = pair["pair_id"]
    segments = []
    options = {}
    for occurrence in pair["occurrences"]:
        ordinal = occurrence["ordinal"]
        window = media[pair_id, ordinal]
        clip = args.media_manifest.parent / window["path"]
        sound_start, sound_end = occurrence["sound_s"]
        action_start, action_end = occurrence["action_s"]
        anchor_time = (sound_start + sound_end) / 2
        action_duration = action_end - action_start
        times = [
            anchor_time,
            action_start + action_duration * 0.25,
            action_start + action_duration * 0.75,
        ]
        clip_start = window["start_s"]
        clip_duration = window["streams"]["video"]
        if any(not 0 <= timestamp - clip_start < clip_duration for timestamp in times):
            times = [clip_start + clip_duration * fraction for fraction in (0.2, 0.5, 0.8)]
            print(f"Using evenly spaced frames for {pair_id} occurrence {ordinal}", file=sys.stderr)
        frames = []
        for index, original_time in enumerate(times, 1):
            path = args.output_dir / f"window-{ordinal}-frame-{index}.png"
            relative_seconds = original_time - clip_start
            subprocess.run(
                [sys.executable, str(extract), str(clip), f"{relative_seconds:.6f}", str(path)],
                check=True,
            )
            frames.append({"path": path.name, "time_s": round(original_time, 4)})
        segments.append({
            "interval_s": [sound_start, sound_end],
            "sound_description": f"Sound label from the source annotation: {pair['sound_class']}.",
            "frames": frames,
        })
        action_id = hashlib.blake2s(occurrence["action_id"].encode(), digest_size=8).hexdigest()
        options[action_id] = occurrence["action_hint"].strip().removesuffix(".") + "."
    if len(options) != 2 or len(set(options.values())) != 2:
        parser.exit(1, "This candidate lacks two distinct action descriptions.\n")

    options = dict(sorted(options.items()))
    with (args.output_dir / "smoke-input.jsonl").open("x", encoding="utf-8") as stream:
        for index, order in ((1, "first"), (2, "second")):
            question = {
                "pair_id": pair_id,
                "qid": f"{pair_id}-smoke-{index}",
                "question": (
                    f"Of the two supplied occurrences of the {pair['sound_class']} sound, "
                    f"what does the person do just after the {order} sound ends?"
                ),
                "options": options,
                "segments": segments,
            }
            stream.write(json.dumps(question, ensure_ascii=False) + "\n")
    print(args.output_dir / "smoke-input.jsonl")


if __name__ == "__main__":
    main()
