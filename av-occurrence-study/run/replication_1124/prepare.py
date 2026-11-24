"""Prepare independent EPIC train candidates using the existing validation-screening rules."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
EPIC = HERE.parents[1] / "epic"
sys.path.insert(0, str(EPIC))
import recover_review_media
import screen
from clip_bounds import review_bounds


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write(path, items):
    with path.open("x", encoding="utf-8") as handle:
        handle.writelines(json.dumps(item, ensure_ascii=False) + "\n" for item in items)


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def screen_train(runtime):
    original = rows(EPIC / "review_queue.jsonl")
    excluded_people = {item["participant_id"] for item in original}
    source = runtime / "epic"
    screen.ROOT = runtime

    def input_rows(name):
        directory = source if name in ("sounds", "actions") else EPIC / "annotations"
        with (directory / f"{name}.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        if name in ("sounds", "actions"):
            rows = [row for row in rows if row["participant_id"] not in excluded_people]
        return rows

    screen.rows = input_rows
    screen.main()
    selected = rows(runtime / "review_queue.jsonl")
    manifest = HERE / "manifest"
    shortlist = [dict(row, candidate_rank=rank) for rank, row in enumerate(selected[:6], 1)]
    write(manifest / "ranked_train_candidates.jsonl", shortlist)
    with (manifest / "data_inputs.json").open("x", encoding="utf-8") as stream:
        json.dump({
            "source_sha256": {
                name: sha(directory / f"{name}.csv")
                for directory, names in [(source, ("sounds", "actions")), (EPIC / "annotations", ("videos", "epic_55_splits", "errata"))]
                for name in names
            },
            "rules_source_sha256": sha(EPIC / "screen.py"),
            "excluded_participants": sorted(excluded_people),
            "selection_rule": "Same rules and greedy class/participant balancing as 11-19. Take first six after filtering. Independently attempt media validation in rank order; select first three fully decoded pairs with distinct participants, no model outputs consulted.",
        }, stream, ensure_ascii=False, indent=2)


def fetch(runtime):
    queue = rows(HERE / "manifest" / "ranked_train_candidates.jsonl")
    recover_review_media.ROOT = HERE
    (HERE / "review_media").mkdir(exist_ok=True)
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(recover_review_media.work, row, runtime / "epic-range-cache") for row in queue]
        for candidate, future in zip(queue, futures):
            print(candidate["candidate_rank"], candidate["pair_id"], future.result(), flush=True)


def freeze():
    queue = rows(HERE / "manifest" / "ranked_train_candidates.jsonl")
    picked = []
    media = []
    participants = set()
    availability = []
    for row in queue:
        pair_id = row["pair_id"]
        windows = [HERE / "review_media" / f"{pair_id}_{e['ordinal']}.mp4" for e in row["occurrences"]]
        present = [path.is_file() for path in windows]
        availability.append(dict(pair_id=pair_id, candidate_rank=row["candidate_rank"], decoded=present))
        if not all(present) or row["participant_id"] in participants or len(picked) >= 3:
            continue
        for event, path in zip(row["occurrences"], windows):
            start, stop = review_bounds(row, event)
            probe = json.loads(subprocess.check_output(["ffprobe","-v","error","-show_entries",
                 "stream=codec_type,duration","-of","json",str(path)]))
            lengths = {s["codec_type"]: float(s["duration"]) for s in probe["streams"]}
            if any(abs(lengths.get(k, 0) - (stop-start)) > .6 for k in ("video","audio")):
                raise ValueError(f"Unexpected window duration: {path} {lengths}")
            media.append(dict(pair_id=pair_id, ordinal=event["ordinal"], path=str(path.relative_to(HERE / "manifest", walk_up=True)),
                 start_s=round(start,3), end_s=round(stop,3), streams=lengths, sha256=sha(path)))
        picked.append(row)
        participants.add(row["participant_id"])
    if len(picked) < 2:
        raise ValueError("Need at least two independent complete pairs")
    write(HERE / "manifest" / "available.jsonl", availability)
    write(HERE / "manifest" / "selected.jsonl", picked)
    write(HERE / "manifest" / "media.jsonl", media)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=("screen","fetch","freeze"))
    parser.add_argument("--runtime", type=Path, default=Path("/private/tmp/av-occurrence-1124-runtime"))
    args = parser.parse_args()
    if args.step == "screen":
        screen_train(args.runtime)
    elif args.step == "fetch":
        fetch(args.runtime)
    else:
        freeze()
