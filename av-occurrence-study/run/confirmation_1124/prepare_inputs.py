#!/usr/bin/env python3
"""Freeze the remaining ranked EPIC train pairs without running the response model."""

import argparse
import json
from pathlib import Path
import platform
import subprocess
import sys
import tempfile


HERE = Path(__file__).resolve().parent
STUDY = HERE.parents[1]
PREVIOUS = STUDY / "run/replication_1124"
sys.path[:0] = [str(PREVIOUS), str(STUDY / "epic")]

import replicate
from clip_bounds import review_bounds
from replicate import content_hash, read, write


RANKED = "run/replication_1124/manifest/ranked_train_candidates.jsonl"
PREVIOUS_SOURCES = (
    "epic/review_queue.jsonl",
    "run/replication_1124/manifest/selected.jsonl",
    "run/epic_smoke_predictions.jsonl",
    "run/epic_smoke_reversed_options.jsonl",
    "run/replication_1124/predictions.jsonl",
)


def select():
    exclusions = {}
    for source in PREVIOUS_SOURCES:
        rows = read(STUDY / source)
        videos = {row["pair_id"].rsplit("-", 1)[0] for row in rows}
        exclusions[source] = {
            "rows": len(rows),
            "sha256": content_hash(STUDY / source),
            "videos": sorted(videos),
            "participants": sorted({video.split("_", 1)[0] for video in videos}),
        }
    used_videos = {video for source in exclusions.values() for video in source["videos"]}
    used_people = {person for source in exclusions.values() for person in source["participants"]}
    queue = read(STUDY / RANKED)
    if [row["candidate_rank"] for row in queue] != list(range(1, 7)):
        raise ValueError("The frozen source queue must contain ranks 1–6 in order")
    selected, decisions = [], []
    for candidate in queue:
        video, person = candidate["video_id"], candidate["participant_id"]
        if video in used_videos:
            decision = "exclude_used_video"
        elif person in used_people:
            decision = "exclude_used_participant"
        else:
            decision = "select"
            used_videos.add(video)
            used_people.add(person)
            selected.append(candidate)
        decisions.append({
            key: candidate[key] for key in ("candidate_rank", "pair_id", "video_id", "participant_id")
        } | {"decision": decision})
    if [row["candidate_rank"] for row in selected] != [4, 5, 6]:
        raise ValueError("The fixed holdout of ranks 4–6 is unavailable; this batch has no replacements")
    return selected, decisions, exclusions


def inspect_media(selected, media_dir, manifest):
    clips = []
    for candidate in selected:
        pair = candidate["pair_id"]
        for event in candidate["occurrences"]:
            ordinal = event["ordinal"]
            path = media_dir / f"{pair}_{ordinal}.mp4"
            start, end = review_bounds(candidate, event)
            result = json.loads(subprocess.check_output([
                "ffprobe", "-v", "error", "-show_entries", "stream=codec_type,duration", "-of", "json", str(path),
            ]))
            durations = {
                stream["codec_type"]: float(stream["duration"])
                for stream in result["streams"] if stream["codec_type"] in ("video", "audio")
            }
            if any(abs(durations.get(kind, 0.0) - (end - start)) > 0.6 for kind in ("video", "audio")):
                raise ValueError(f"Unexpected audio or video duration for {path}: {durations}")
            subprocess.run([
                "ffmpeg", "-nostdin", "-v", "error", "-xerror", "-i", str(path),
                "-map", "0:v:0", "-map", "0:a:0", "-f", "null", "-",
            ], check=True)
            clips.append({
                "pair_id": pair, "ordinal": ordinal, "path": str(path.relative_to(manifest, walk_up=True)),
                "start_s": round(start, 3), "end_s": round(end, 3), "streams": durations,
                "sha256": content_hash(path),
            })
    return clips


def freeze(manifest, media_dir):
    if manifest.exists():
        raise FileExistsError(f"Frozen manifest already exists: {manifest}")
    selected, decisions, exclusions = select()
    clips = inspect_media(selected, media_dir, manifest)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=manifest.parent, prefix=".freeze-") as directory:
        staging = Path(directory)
        write(staging / "selected.jsonl", selected)
        write(staging / "media.jsonl", clips)
        replicate.MANIFEST = staging
        replicate.freeze()
        prompts = read(staging / "prompts.jsonl")
        hashes = json.loads((staging / "input_hashes.json").read_text())
        expected = [
            (row["video_id"], question, condition)
            for row in selected for question in (1, 2) for condition in replicate.CONDITIONS
        ]
        if [(row["video_id"], row["ordinal"], row["condition"]) for row in prompts] != expected:
            raise ValueError("Prompt rows do not match the fixed video, question and condition order")
        if len(hashes["frame_sha256"]) != 18 or len(clips) != 6:
            raise ValueError("Expected six strictly decoded media windows and eighteen extracted frames")
        paths = (
            RANKED, "run/replication_1124/manifest/data_inputs.json", "run/replication_1124/manifest/available.jsonl",
            "epic/clip_bounds.py", "run/infer_mlx.py", "run/extract_frame.py", "run/prepare_epic_smoke.py",
            "run/replication_1124/replicate.py", str(Path(__file__).resolve().relative_to(STUDY)),
        )
        provenance = {
            "selection_rule": "Retain ranks 4–6 of the frozen train shortlist in order after verifying video and participant disjointness; no replacements.",
            "candidate_decisions": decisions,
            "exclusion_sources": exclusions,
            "source_sha256": {path: content_hash(STUDY / path) for path in paths},
            "manifest_sha256": {path.name: content_hash(path) for path in sorted(staging.glob("*.json*"))},
            "frame_manifest_sha256": {
                str(path.relative_to(staging)): content_hash(path)
                for path in sorted((staging / "frames").glob("*/smoke-input.jsonl"))
            },
            "checkpoint": {"revision": replicate.MODEL_REVISION, "expected_shard_sha256": replicate.SHARD_SHA},
            "generation": {"temperature": 0.0, "max_tokens": 64},
            "offline_preflight": {
                "audio_video_windows_decoded": len(clips), "extracted_frames": len(hashes["frame_sha256"]),
                "videos": len(selected), "questions": len(expected) // len(replicate.CONDITIONS),
                "prompts": len(prompts), "condition_isolation": "validated by replication_1124/replicate.py freeze",
                "python": platform.python_version(),
                "ffmpeg": subprocess.check_output(["ffmpeg", "-version"], text=True).splitlines()[0],
            },
        }
        (staging / "provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n")
        staging.rename(manifest)
    print(f"Input manifests and local frames frozen in {manifest}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--media-dir", type=Path,
        default=STUDY.parents[1] / "av-occurrence-study/run/replication_1124/review_media",
        help="Directory containing the six locally recovered EPIC clips",
    )
    parser.add_argument("--output", type=Path, default=HERE / "manifest", help="New directory for frozen inputs")
    args = parser.parse_args()
    freeze(args.output.resolve(), args.media_dir.resolve())
