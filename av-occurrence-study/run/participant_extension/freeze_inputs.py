#!/usr/bin/env python3
"""Freeze the extension batch's model inputs for all 72 planned call keys, without generating replies.

The 72 call keys come from planned_calls.csv of the prespecified analysis plan
(commit f653ba6f), copied unchanged into inputs/. Runnable prompts are built by the
unmodified replication_1124/replicate.py freeze step (the preceding batches'
prompt construction, condition mappings and frame rule), which re-extracts every
frame from the local clips through prepare_epic_smoke.py. Each clip, frame and
question input must match the reviewed media_prep/windows.jsonl byte for byte.

A pair without a six-frame question input keeps its rank and its six call keys;
those keys are written as technical_missing with the reason, after confirming that
prepare_epic_smoke.py still stops at the recorded frame. Nothing is replaced and no
model is loaded.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
RUN = HERE.parent
STUDY = RUN.parent
sys.path[:0] = [str(RUN / "replication_1124"), str(HERE)]
import replicate  # noqa: E402
from prepare_media import verify_queue  # noqa: E402

PLAN_COMMIT = "f653ba6fceef7b16a733349041a33823834d0eca"
PLANNED_CALLS = HERE / "inputs/planned_calls.csv"
PLANNED_CALLS_SHA = "a69bf82c89961a310096750d2b78f47efb27e1d0042ad9d6ce1d0e15f6202139"
WINDOWS = HERE / "media_prep/windows.jsonl"
SMOKE = RUN / "prepare_epic_smoke.py"
# Parser pinned by the preceding protocol: PR #2's infer_mlx.py (f6919987, merged as 87ee601).
PARSER_COMMIT = "f6919987b6fe3ecf9746f7d9a5d7ab1edb8db3c4"
PARSER_SHA = "4ebbd521d030f6cfdf9f4b4bead886bd53d076ffb6893cc542202680219fc027"
PROGRAMS = (
    "run/replication_1124/replicate.py", "run/infer_mlx.py", "run/prepare_epic_smoke.py",
    "run/extract_frame.py", "epic/clip_bounds.py", "run/participant_extension/freeze_inputs.py",
)
OUTPUTS = ("selected.jsonl", "media.jsonl", "prompts.jsonl", "input_hashes.json", "calls.jsonl", "provenance.json")
FRAME_COUNT = 6


def sha_file(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def sha_text(text):
    return hashlib.sha256(text.encode()).hexdigest()


def dump_jsonl(rows):
    return "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)


def load_plan(queue):
    if sha_file(PLANNED_CALLS) != PLANNED_CALLS_SHA:
        raise SystemExit(f"{PLANNED_CALLS} differs from the planned calls frozen in {PLAN_COMMIT}")
    with PLANNED_CALLS.open(newline="", encoding="utf-8") as handle:
        plan = list(csv.DictReader(handle))
    expected = [
        dict(run_order=str(6 * index + 3 * (ordinal - 1) + position + 1),
             extension_rank=str(row["extension_rank"]), participant_id=row["participant_id"],
             video_id=row["video_id"], pair_id=row["pair_id"], qid=f"{row['pair_id']}-smoke-{ordinal}",
             ordinal=str(ordinal), condition=condition)
        for index, row in enumerate(queue)
        for ordinal in (1, 2) for position, condition in enumerate(replicate.CONDITIONS)
    ]
    if plan != expected:
        raise SystemExit("planned_calls.csv does not follow the frozen queue, question and condition order")
    return plan


def load_windows(queue, work):
    rows = replicate.read(WINDOWS)
    if [(row["pair_id"], row["ordinal"]) for row in rows] != [
            (pair["pair_id"], ordinal) for pair in queue for ordinal in (1, 2)]:
        raise SystemExit("media_prep/windows.jsonl does not list the queue's 24 windows in order")
    windows = {}
    for row in rows:
        if row["status"] == "decoded":
            clip = work / "review_media" / row["clip"]
            if not clip.is_file() or sha_file(clip) != row["sha256"]:
                raise SystemExit(f"{clip} is missing or differs from media_prep/windows.jsonl; run prepare_media.py first")
        windows[row["pair_id"], row["ordinal"]] = row
    return windows


def runnable(windows, pair_id):
    pair = [windows[pair_id, ordinal] for ordinal in (1, 2)]
    return all(row["status"] == "decoded" and row["frames_status"] == "extracted" and row["smoke_input_sha256"]
               for row in pair)


def media_rows(windows, pairs, work):
    return [
        dict(pair_id=pair_id, ordinal=ordinal, path=str(work / "review_media" / row["clip"]),
             start_s=row["window_s"][0], end_s=row["window_s"][1],
             streams={"video": row["streams"]["video"]["duration_s"], "audio": row["streams"]["audio"]["duration_s"]})
        for pair_id in pairs for ordinal in (1, 2) for row in [windows[pair_id, ordinal]]
    ]


def recorded_frames(windows, pair_id):
    return {frame["file"]: frame for ordinal in (1, 2)
            for frame in windows[pair_id, ordinal].get("frames", [])}


def check_runnable_frames(staging, windows, pairs):
    """Every re-extracted frame and question input must equal the reviewed media preparation."""
    for pair_id in pairs:
        folder = staging / "frames" / pair_id
        expected = {name: frame["sha256"] for name, frame in recorded_frames(windows, pair_id).items()}
        produced = {path.name: sha_file(path) for path in folder.glob("*.png")}
        if len(produced) != FRAME_COUNT or produced != expected:
            raise SystemExit(f"Frames of {pair_id} differ from media_prep/windows.jsonl")
        if sha_file(folder / "smoke-input.jsonl") != windows[pair_id, 1]["smoke_input_sha256"]:
            raise SystemExit(f"Question input of {pair_id} differs from media_prep/windows.jsonl")


def missing_pair(windows, pair_id, queue_row, work, scratch):
    """Explain why a pair has no six-frame input, re-running the frame step where both clips decoded."""
    pair = [windows[pair_id, ordinal] for ordinal in (1, 2)]
    record = dict(pair_id=pair_id, windows=[])
    for row in pair:
        entry = dict(ordinal=row["ordinal"], window_s=row["window_s"], status=row["status"],
                     clip_sha256=row.get("sha256"), frames_status=row["frames_status"])
        for key in ("failure_reason", "frames_failure_reason"):
            if row.get(key):
                entry[key] = row[key]
        entry["frames"] = [
            {key: frame[key] for key in ("file", "role", "requested_original_s", "status", "decoded_original_s", "sha256")
             if key in frame}
            for frame in row.get("frames", [])
        ]
        record["windows"].append(entry)
    record["reason"] = next(
        (f"window {row['ordinal']}: {row.get('failure_reason') or row.get('frames_failure_reason') or row['frames_status']}"
         for row in pair if row["status"] != "decoded" or row["frames_status"] != "extracted"),
        "no six-frame question input")
    if all(row["status"] == "decoded" for row in pair):
        # Confirm the recorded stop with the unchanged frame step on the same clips.
        queue = scratch / f"{pair_id}-queue.jsonl"
        media = scratch / f"{pair_id}-media.jsonl"
        output = scratch / pair_id
        queue.write_text(dump_jsonl([queue_row]), encoding="utf-8")
        media.write_text(dump_jsonl(media_rows(windows, [pair_id], work)), encoding="utf-8")
        result = subprocess.run([sys.executable, str(SMOKE), "--queue", str(queue), "--media-manifest", str(media),
                                 "--output-dir", str(output), "--pair", pair_id], capture_output=True, text=True)
        produced = {path.name: sha_file(path) for path in output.glob("*.png")} if output.is_dir() else {}
        extracted = {name: frame["sha256"] for name, frame in recorded_frames(windows, pair_id).items()
                     if frame["status"] == "extracted"}
        if result.returncode == 0 or (output / "smoke-input.jsonl").exists() or produced != extracted:
            raise SystemExit(f"prepare_epic_smoke.py no longer stops as recorded for {pair_id}")
        record["rechecked"] = (f"prepare_epic_smoke.py exited {result.returncode} without a question input; "
                               f"{len(produced)} frames written before the stop match media_prep/windows.jsonl")
    record["available_frames"] = sum(frame["status"] == "extracted" for w in record["windows"] for frame in w["frames"])
    return record


def build(work, staging):
    queue, queue_sha, combined = verify_queue()
    plan = load_plan(queue)
    windows = load_windows(queue, work)
    by_pair = {row["pair_id"]: row for row in queue}
    ready = [row["pair_id"] for row in queue if runnable(windows, row["pair_id"])]
    missing = {}
    with tempfile.TemporaryDirectory(prefix=".missing-", dir=staging.parent) as scratch:
        for row in queue:
            if row["pair_id"] not in ready:
                missing[row["pair_id"]] = missing_pair(windows, row["pair_id"], row, work, Path(scratch))

    replicate.write(staging / "selected.jsonl", [by_pair[pair_id] for pair_id in ready])
    replicate.write(staging / "media.jsonl", media_rows(windows, ready, work))
    replicate.MANIFEST = staging
    replicate.freeze()
    check_runnable_frames(staging, windows, ready)
    # The local clip paths only served the frame step; keep a portable record instead.
    (staging / "media.jsonl").unlink()
    replicate.write(staging / "media.jsonl", [
        dict(row, path=Path(row["path"]).name, sha256=windows[row["pair_id"], row["ordinal"]]["sha256"])
        for row in media_rows(windows, ready, work)
    ])

    prompts = replicate.read(staging / "prompts.jsonl")
    lines = {(row["qid"], row["condition"]): number for number, row in enumerate(prompts, 1)}
    calls = []
    for key in plan:
        call = dict(run_order=int(key["run_order"]), extension_rank=int(key["extension_rank"]),
                    participant_id=key["participant_id"], video_id=key["video_id"], pair_id=key["pair_id"],
                    qid=key["qid"], ordinal=int(key["ordinal"]), condition=key["condition"])
        line = lines.pop((key["qid"], key["condition"]), None)
        if line is None:
            if key["pair_id"] not in missing:
                raise SystemExit(f"No prompt and no recorded reason for {key['qid']} {key['condition']}")
            call.update(status="technical_missing", stage="frame_extraction",
                        reason=missing[key["pair_id"]]["reason"], prompt_line=None)
        else:
            row = prompts[line - 1]
            if row["pair_id"] != key["pair_id"] or row["ordinal"] != call["ordinal"]:
                raise SystemExit(f"Prompt line {line} does not belong to call {call['run_order']}")
            call.update(status="runnable", prompt_line=line, prompt_sha256=sha_text(row["prompt"]),
                        frames_sha256=[sha_file(staging / path) for path in row["frames"]],
                        option_codes=row["option_codes"], segment_codes=row["segment_codes"])
        calls.append(call)
    if lines:
        raise SystemExit(f"Prompts outside the planned call keys: {sorted(lines)}")
    replicate.write(staging / "calls.jsonl", calls)

    hashes = json.loads((staging / "input_hashes.json").read_text())
    counts = dict(planned_calls=len(calls), runnable=sum(c["status"] == "runnable" for c in calls),
                  technical_missing=sum(c["status"] == "technical_missing" for c in calls),
                  recordings=len(queue), runnable_recordings=len(ready), questions=2 * len(queue),
                  runnable_questions=2 * len(ready), frames=len(hashes["frame_sha256"]))
    if counts["runnable"] != len(prompts) or counts["frames"] != FRAME_COUNT * len(ready):
        raise SystemExit("Prompt or frame counts disagree with the runnable recordings")
    parser_sha = sha_file(RUN / "infer_mlx.py")
    if parser_sha != PARSER_SHA:
        raise SystemExit("run/infer_mlx.py differs from the parser pinned by the preceding protocol")
    provenance = dict(
        plan=dict(commit=PLAN_COMMIT, planned_calls="inputs/planned_calls.csv", planned_calls_sha256=PLANNED_CALLS_SHA),
        queue=dict(queue_sha256=queue_sha, fingerprint=combined),
        media_prep_windows_sha256=sha_file(WINDOWS),
        counts=counts,
        unavailable=list(missing.values()),
        rules=dict(
            prompts="replication_1124/replicate.py freeze (flat prompt; original, options_reversed, "
                    "segment_labels_reversed), unchanged",
            frames="prepare_epic_smoke.py with decoded video duration; no added, moved or replacement frames",
            missing="a recording without both windows' six frames keeps its rank and its six call keys as technical_missing",
        ),
        checkpoint=dict(model="mlx-community/Qwen2.5-VL-7B-Instruct-4bit", revision=replicate.MODEL_REVISION,
                        expected_shard_sha256=replicate.SHARD_SHA),
        generation=dict(temperature=0.0, max_tokens=64, history="none; each call independent"),
        parser=dict(source="run/infer_mlx.py", sha256=parser_sha, pinned_commit=PARSER_COMMIT,
                    primary='parse_reply(..., rule="matched_seconds")',
                    sensitivity='decode(..., rule="strict") (bare evidence code only)'),
        source_sha256={path: sha_file(STUDY / path) for path in PROGRAMS},
        manifest_sha256={name: sha_file(staging / name) for name in OUTPUTS[:-1]},
        frame_manifest_sha256={
            str(path.relative_to(staging)): sha_file(path) for path in sorted((staging / "frames").glob("*/smoke-input.jsonl"))
        },
        tools=dict(python=platform.python_version(),
                   ffmpeg=subprocess.check_output(["ffmpeg", "-version"], text=True).splitlines()[0]),
        predictions="none; inputs frozen before any extension reply is generated",
    )
    (staging / "provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return counts


def text_files(root):
    return sorted(path.relative_to(root) for path in root.rglob("*") if path.is_file() and path.suffix != ".png"
                  and path.name != "README.md")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--work-dir", type=Path, required=True,
                        help="prepare_media.py work directory holding review_media/ with the 24 clips")
    parser.add_argument("--output", type=Path, required=True, help="New directory for the frozen inputs")
    parser.add_argument("--check", type=Path, metavar="DIR",
                        help="Compare every text file (all but PNG frames) byte by byte with DIR")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise SystemExit(f"Output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent, prefix=".freeze-") as directory:
        staging = Path(directory) / "inputs"
        staging.mkdir()
        counts = build(args.work_dir.resolve(), staging)
        staging.rename(output)
    print(f"{counts['runnable']} runnable and {counts['technical_missing']} technically missing of "
          f"{counts['planned_calls']} call keys frozen in {output}", flush=True)
    if args.check:
        mine, theirs = text_files(output), text_files(args.check)
        differ = [str(path) for path in sorted(set(mine) | set(theirs))
                  if path not in mine or path not in theirs
                  or (output / path).read_bytes() != (args.check / path).read_bytes()]
        if differ:
            raise SystemExit(f"differs from {args.check}: {', '.join(differ)}")
        print("identical to", args.check, flush=True)


if __name__ == "__main__":
    main()
