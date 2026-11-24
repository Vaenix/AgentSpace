"""Prepare local media windows and frames for the frozen participant-extension queue.

The queue is checked against its frozen fingerprint before use. Clipping reuses
epic/recover_review_media.py (resumable byte-range download of the official MP4,
window bounds from epic/clip_bounds.py, H.264/AAC re-encode with a dual-track
check). Frames reuse run/prepare_epic_smoke.py, which calls run/extract_frame.py.
None of those programs are modified here.

Original-media ranges, clips and frames stay under --work-dir. Only the small
text manifests in --output (one row per window plus a summary) are meant for Git.
No question answering or model inference happens in this step.
"""
import argparse
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
RUN = HERE.parent
EPIC = RUN.parent / "epic"
sys.path.insert(0, str(EPIC))
import recover_review_media  # noqa: E402
from clip_bounds import review_bounds  # noqa: E402

QUEUE_DIR = HERE / "manifest"
QUEUE_OUTPUTS = ("queue.jsonl", "exclusion_audit.json", "source_hashes.json")
FROZEN_COMBINED = "faabf53d91a888e59844ff5b22565635aad30ddc040809328aa6fee82c5f7084"
SMOKE = RUN / "prepare_epic_smoke.py"
REUSED = {
    "epic/recover_review_media.py": EPIC / "recover_review_media.py",
    "epic/clip_bounds.py": EPIC / "clip_bounds.py",
    "run/prepare_epic_smoke.py": SMOKE,
    "run/extract_frame.py": RUN / "extract_frame.py",
    "run/participant_extension/prepare_media.py": Path(__file__).resolve(),
}
# Same tolerance as recover_review_media.process and replication_1124/prepare.py.
DURATION_TOLERANCE_S = 0.6
FRAME_ROLES = ("sound_midpoint", "action_25pct", "action_75pct")
FALLBACK_ROLES = ("window_20pct", "window_50pct", "window_80pct")
FALLBACK_NOTE = re.compile(r"Using evenly spaced frames for (\S+) occurrence (\d+)")
OUTPUTS = ("windows.jsonl", "summary.json")
# FFmpeg log prefixes such as "[mov,mp4,m4a,3gp,3g2,mj2 @ 0x74c7054000]" carry a per-process address.
VOLATILE_ADDRESS = re.compile(r" @ 0x[0-9a-fA-F]+(?=\])")


def sha_file(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def stable_text(text, *paths):
    """Drop local paths and per-process addresses so failure reasons replay byte for byte."""
    for path in paths:
        text = text.replace(str(path), Path(path).name)
    return VOLATILE_ADDRESS.sub("", text)


def dump_jsonl(items):
    return "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in items)


def dump_json(item):
    return json.dumps(item, ensure_ascii=False, indent=2) + "\n"


def verify_queue():
    """Refuse to run unless the queue is byte-identical to the reviewed freeze."""
    recorded = json.loads((QUEUE_DIR / "fingerprint.json").read_text(encoding="utf-8"))
    actual = {name: sha_file(QUEUE_DIR / name) for name in QUEUE_OUTPUTS}
    combined = hashlib.sha256("".join(f"{actual[name]}  {name}\n" for name in QUEUE_OUTPUTS).encode()).hexdigest()
    if actual != recorded["files"] or combined != recorded["combined"] or combined != FROZEN_COMBINED:
        raise SystemExit(f"Queue differs from the frozen fingerprint {FROZEN_COMBINED}: {combined}")
    queue = jsonl(QUEUE_DIR / "queue.jsonl")
    if [row["pair_id"] for row in queue] != recorded["order"] or len(queue) != 12:
        raise SystemExit("Queue order differs from fingerprint.json")
    return queue, actual["queue.jsonl"], combined


def run(command, timeout=120):
    return subprocess.run(command, capture_output=True, text=True, timeout=timeout)


def planned_frames(occurrence, clip_start, clip_duration):
    """The frame rule of prepare_epic_smoke.py, restated to explain each outcome."""
    sound_start, sound_end = occurrence["sound_s"]
    action_start, action_end = occurrence["action_s"]
    action_duration = action_end - action_start
    planned = dict(zip(FRAME_ROLES, (
        (sound_start + sound_end) / 2,
        action_start + action_duration * 0.25,
        action_start + action_duration * 0.75,
    )))
    outside = {
        role: seconds for role, seconds in planned.items()
        if not 0 <= seconds - clip_start < clip_duration
    }
    return planned, outside


def substitution_reason(outside, clip_start, clip_duration):
    if not outside:
        return None
    end = clip_start + clip_duration
    parts = [
        f"{role} {seconds:.3f} s {'before window start' if seconds < clip_start else 'at or after window end'}"
        for role, seconds in outside.items()
    ]
    return (f"{'; '.join(parts)} (window {clip_start:.3f}–{end:.3f} s from the decoded video stream); "
            "rule substitutes 3 evenly spaced frames at 20/50/80% of the window")


def probe(clip):
    """Measure stream durations and fully decode both tracks with errors treated as fatal.

    Raises ValueError when the file cannot be probed, e.g. an empty or truncated clip.
    """
    result = run([
        "ffprobe", "-v", "error", "-show_entries",
        "format=start_time,duration:stream=index,codec_type,codec_name,duration,nb_frames,sample_rate,width,height,avg_frame_rate",
        "-of", "json", str(clip),
    ])
    if result.returncode:
        detail = stable_text(result.stderr, clip).strip()[-300:]
        raise ValueError(f"ffprobe exited with status {result.returncode}: {detail}")
    info = json.loads(result.stdout)
    streams = {}
    for stream in info["streams"]:
        kind = stream["codec_type"]
        entry = {"codec": stream.get("codec_name"), "duration_s": float(stream["duration"])}
        if kind == "video":
            entry.update(frames=int(stream["nb_frames"]), width=stream["width"], height=stream["height"],
                         frame_rate=stream["avg_frame_rate"])
        elif kind == "audio":
            entry.update(sample_rate=int(stream["sample_rate"]))
        streams.setdefault(kind, entry)
    decode = run(["ffmpeg", "-nostdin", "-v", "error", "-xerror", "-i", str(clip),
                  "-map", "0:v:0", "-map", "0:a:0", "-f", "null", "-"])
    errors = stable_text(decode.stderr, clip).strip()
    if "audio" in streams:
        # A decodable but silent track would pass the checks above; record its level.
        levels = run(["ffmpeg", "-nostdin", "-hide_banner", "-i", str(clip), "-map", "0:a:0",
                      "-af", "volumedetect", "-f", "null", "-"]).stderr
        for key in ("mean_volume", "max_volume"):
            found = re.search(rf"{key}:\s*(-?[0-9.]+|-inf) dB", levels)
            streams["audio"][f"{key}_db"] = None if found is None or found[1] == "-inf" else float(found[1])
    return streams, float(info["format"].get("start_time", 0)), decode.returncode == 0 and not errors, errors[-400:]


def decoded_frame_time(clip, relative, expected_sha, scratch):
    """Return the presentation time of the frame extract_frame.py produced for `relative`.

    Repeats extract_frame.py's command with -copyts and showinfo, and confirms the
    re-extracted PNG is byte-identical, so the reported timestamp is that frame's.
    """
    png = scratch / "probe.png"
    png.unlink(missing_ok=True)
    result = run([
        "ffmpeg", "-hide_banner", "-loglevel", "info", "-nostdin", "-n", "-copyts",
        "-ss", relative, "-i", str(clip), "-map", "0:v:0", "-frames:v", "1",
        "-vf", "showinfo,scale=448:448:force_original_aspect_ratio=decrease:force_divisible_by=2,"
               "pad=448:448:(ow-iw)/2:(oh-ih)/2",
        "-pix_fmt", "rgb24", "-update", "1", str(png),
    ])
    # showinfo may see one more buffered frame before -frames:v stops; frame n:0 is the one written.
    times = re.findall(r"\bn:\s*0\s+pts:\s*-?\d+\s+pts_time:\s*(-?[0-9.]+)", result.stderr)
    if result.returncode or len(times) != 1 or not png.is_file():
        raise ValueError(f"Could not time the extracted frame at {relative}: {result.stderr[-300:]}")
    if sha_file(png) != expected_sha:
        raise ValueError(f"Re-extracted frame at {relative} differs from the extract_frame.py output")
    return float(times[0])


def fetch_pair(row, cache):
    """Run the original clipper on one pair; return per-window failure reasons."""
    windows = list(recover_review_media.media_windows(row))
    log = io.StringIO()
    source_error = None
    with contextlib.redirect_stdout(log):
        try:
            recover_review_media.process(row, cache, windows)
        except (ValueError, OSError, subprocess.SubprocessError, KeyError, recover_review_media.struct.error) as exc:
            source_error = repr(exc)
    text = log.getvalue()
    sys.stdout.write(text)
    if source_error:
        print("failed source", row["pair_id"], source_error, flush=True)
    reasons = {}
    for line in text.splitlines():
        if line.startswith("failed window "):
            _, _, stem, reason = line.split(" ", 3)
            reasons[stem] = reason
    return windows, reasons, source_error


def failure_stage(reason):
    download = ("curl", "range", "MP4", "remote source", "packets", "metadata", "Truncated", "Malformed")
    return "download" if any(word.lower() in reason.lower() for word in download) else "clip_or_decode"


def last_frame_s(clip):
    packets = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "packet=pts_time",
        "-of", "json", str(clip),
    ], text=True))["packets"]
    return max(float(packet["pts_time"]) for packet in packets)


def extract_frames(row, clips, work):
    """Call prepare_epic_smoke.py unchanged for a pair whose two windows decoded.

    Returns per-window frame records. If the program stops at a frame, the frames
    it wrote before stopping are kept and timed, and the failing frame is explained.
    """
    pair_id = row["pair_id"]
    frames_root = work / "frames"
    frames_root.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{pair_id}-", dir=frames_root) as temp:
        temp = Path(temp)
        media = temp / "media.jsonl"
        media.write_text(dump_jsonl(
            dict(pair_id=pair_id, ordinal=ordinal, path=os.path.relpath(clip["file"], temp),
                 start_s=clip["start_s"], end_s=clip["end_s"], streams={"video": clip["video_s"], "audio": clip["audio_s"]})
            for ordinal, clip in clips.items()
        ), encoding="utf-8")
        output = temp / "out"
        result = run([sys.executable, str(SMOKE), "--queue", str(QUEUE_DIR / "queue.jsonl"),
                      "--media-manifest", str(media), "--output-dir", str(output), "--pair", pair_id], timeout=300)
        substituted = {int(ordinal) for found, ordinal in FALLBACK_NOTE.findall(result.stderr) if found == pair_id}
        stopped_at = None
        if result.returncode:
            match = re.search(r"window-(\d+)-frame-(\d+)\.png'\]' returned non-zero exit status", result.stderr)
            if match is None or not output.is_dir():
                raise ValueError(f"prepare_epic_smoke.py failed before extracting frames: {result.stderr.strip()[-400:]}")
            stopped_at = (int(match[1]), int(match[2]))
        produced = {path.name: sha_file(path) for path in sorted(output.iterdir())}
        final = frames_root / pair_id
        if final.exists():
            existing = {path.name: sha_file(path) for path in sorted(final.iterdir())}
            if existing != produced:
                raise ValueError(f"Frames in {final} differ from a fresh extraction; move them aside to rebuild")
        else:
            output.rename(final)
            output = final
        recorded = {}
        if "smoke-input.jsonl" in produced:
            segments = jsonl(output / "smoke-input.jsonl")[0]["segments"]
            recorded = {frame["path"]: frame["time_s"] for segment in segments for frame in segment["frames"]}
        results = {}
        for occurrence in row["occurrences"]:
            ordinal = occurrence["ordinal"]
            clip = clips[ordinal]
            if stopped_at and ordinal > stopped_at[0]:
                results[ordinal] = dict(frames_status="not_reached: prepare_epic_smoke.py stopped at an earlier window",
                                        frames=[])
                continue
            planned, outside = planned_frames(occurrence, clip["start_s"], clip["video_s"])
            if bool(outside) != (ordinal in substituted):
                raise ValueError(f"Substitution rule disagrees with prepare_epic_smoke.py for {pair_id}_{ordinal}")
            if outside:
                requested = [clip["start_s"] + clip["video_s"] * fraction for fraction in (0.2, 0.5, 0.8)]
                roles = FALLBACK_ROLES
            else:
                requested, roles = list(planned.values()), FRAME_ROLES
            frames, failure = [], None
            for index, (role, seconds) in enumerate(zip(roles, requested), 1):
                name = f"window-{ordinal}-frame-{index}.png"
                relative = f"{seconds - clip['start_s']:.6f}"
                frame = dict(file=name, role=role, requested_original_s=round(seconds, 4), requested_clip_s=float(relative))
                if name not in produced:
                    if stopped_at == (ordinal, index):
                        last = last_frame_s(clip["file"]) - clip["clip_start_time"]
                        if float(relative) > last:
                            failure = (f"extract_frame.py decoded no frame for {role} at clip {relative} s: the last video "
                                       f"frame starts at {last:.3f} s while the video stream lasts {clip['video_s']} s, "
                                       "so the point passes the window check but has no frame at or after it; "
                                       "prepare_epic_smoke.py stops here and writes no question input")
                        else:
                            detail = stable_text(result.stderr, clip["file"], work).strip()[-300:]
                            failure = f"extract_frame.py failed for {role} at clip {relative} s: {detail}"
                        frame.update(status="failed", failure_reason=failure)
                    else:
                        frame.update(status="not_reached")
                    frames.append(frame)
                    continue
                if name in recorded and abs(recorded[name] - round(seconds, 4)) > 1e-9:
                    raise ValueError(f"Unexpected frame time in {pair_id}_{ordinal}: {name} {recorded[name]}")
                digest = produced[name]
                pts = decoded_frame_time(clip["file"], relative, digest, temp)
                clip_seconds = round(pts - clip["clip_start_time"], 6)
                frame.update(status="extracted", decoded_clip_s=clip_seconds,
                             decoded_original_s=round(clip["start_s"] + clip_seconds, 4), size_px=[448, 448], sha256=digest)
                frames.append(frame)
            results[ordinal] = dict(
                frames_status="extracted" if failure is None else "failed",
                substituted=bool(outside),
                substitution_reason=substitution_reason(outside, clip["start_s"], clip["video_s"]),
                frames=frames,
            )
            if failure:
                results[ordinal]["frames_failure_reason"] = failure
        smoke = produced.get("smoke-input.jsonl")
    return results, smoke


def prepare(queue, selected, work):
    cache = work / "epic-range-cache"
    recover_review_media.ROOT = work
    (work / "review_media").mkdir(parents=True, exist_ok=True)
    records = []
    for row in queue:
        attempted = row["pair_id"] in selected
        if attempted:
            windows, reasons, source_error = fetch_pair(row, cache)
        else:
            windows, reasons, source_error = list(recover_review_media.media_windows(row)), {}, None
        clips = {}
        pair_records = []
        for occurrence, (path, _start, _stop, _clean, _action_end) in zip(row["occurrences"], windows):
            ordinal = occurrence["ordinal"]
            start, stop = review_bounds(row, occurrence)
            start, stop = round(start, 3), round(stop, 3)
            action_start, action_end = occurrence["action_s"]
            planned, outside = planned_frames(occurrence, start, stop - start)
            record = dict(
                pair_id=row["pair_id"], extension_rank=row["extension_rank"], participant_id=row["participant_id"],
                video_id=row["video_id"], ordinal=ordinal, sound_class=row["sound_class"],
                sound_s=occurrence["sound_s"], action_s=occurrence["action_s"],
                window_s=[start, stop], planned_duration_s=round(stop - start, 3),
                action_truncated_s=round(max(0, action_end - stop), 3),
                planned_substitution=bool(outside), clip=path.name,
            )
            if not attempted:
                record.update(status="not_attempted")
            elif path.is_file():
                try:
                    streams, clip_start_time, clean_decode, errors = probe(path)
                except (ValueError, KeyError, TypeError, subprocess.SubprocessError) as exc:
                    # An existing clip that cannot be probed fails only this window.
                    record.update(status="decode_failed", bytes=path.stat().st_size, sha256=sha_file(path),
                                  failure_reason=f"existing clip could not be probed: {exc}")
                    pair_records.append(record)
                    continue
                video, audio = streams.get("video"), streams.get("audio")
                lengths_ok = video and audio and all(
                    abs(track["duration_s"] - (stop - start)) <= DURATION_TOLERANCE_S for track in (video, audio))
                record.update(
                    status="decoded" if clean_decode and lengths_ok else "decode_failed",
                    streams=streams, bytes=path.stat().st_size, sha256=sha_file(path),
                )
                if not clean_decode:
                    record["failure_reason"] = f"strict decode of both tracks failed: {errors}"
                elif not lengths_ok:
                    record["failure_reason"] = f"stream durations differ from {stop - start:.3f} s by more than {DURATION_TOLERANCE_S} s"
                else:
                    clips[ordinal] = dict(file=path, start_s=start, end_s=stop, video_s=video["duration_s"],
                                          audio_s=audio["duration_s"], clip_start_time=clip_start_time)
            else:
                reason = reasons.get(path.stem) or source_error or "clip missing without a logged reason"
                reason = stable_text(reason, work)
                record.update(status=f"{failure_stage(reason)}_failed", failure_reason=reason)
            pair_records.append(record)
        if len(clips) == 2:
            frames, smoke = extract_frames(row, clips, work)
            for record in pair_records:
                record.update(smoke_input_sha256=smoke, **frames[record["ordinal"]])
        else:
            for record in pair_records:
                record["frames_status"] = ("not_attempted" if not attempted else
                                           "skipped: prepare_epic_smoke.py needs both windows of the pair")
        records.extend(pair_records)
        status = ", ".join(f"{r['ordinal']}={r['status']}" for r in pair_records)
        print(row["extension_rank"], row["pair_id"], status, flush=True)
    return records


def versions():
    ffmpeg = subprocess.check_output(["ffmpeg", "-version"], text=True).splitlines()[0]
    ffprobe = subprocess.check_output(["ffprobe", "-version"], text=True).splitlines()[0]
    return dict(python=platform.python_version(), ffmpeg=ffmpeg, ffprobe=ffprobe, platform=platform.platform())


def summarize(records, queue_sha, combined):
    count = lambda **match: sum(all(r.get(k) == v for k, v in match.items()) for r in records)
    pairs = {}
    for record in records:
        pairs.setdefault(record["pair_id"], []).append(record)
    return dict(
        queue_sha256=queue_sha, queue_fingerprint=combined,
        windows_planned=len(records), windows_attempted=len(records) - count(status="not_attempted"),
        windows_decoded=count(status="decoded"),
        windows_failed={status: count(status=status) for status in sorted({r["status"] for r in records})
                        if status not in ("decoded", "not_attempted")},
        pairs_planned=len(pairs),
        pairs_complete=sum(all(r["status"] == "decoded" for r in group) for group in pairs.values()),
        windows_with_frames=count(frames_status="extracted"),
        pairs_with_frames=sum(all(r["frames_status"] == "extracted" for r in group) for group in pairs.values()),
        frames_extracted=sum(f["status"] == "extracted" for r in records for f in r.get("frames", [])),
        frames_failed=sum(f["status"] == "failed" for r in records for f in r.get("frames", [])),
        windows_truncating_action=sum(r["action_truncated_s"] > 0 for r in records),
        windows_planned_substitution=count(planned_substitution=True),
        windows_substituted=count(substituted=True),
        program_sha256={name: sha_file(path) for name, path in REUSED.items()},
        tools=versions(),
        predictions="none; this step only prepares media and frames",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--work-dir", type=Path, required=True,
                        help="Local directory for the range cache, clips and frames (kept out of Git)")
    parser.add_argument("--output", type=Path, required=True, help="New directory for windows.jsonl and summary.json")
    parser.add_argument("--pairs", nargs="+", metavar="PAIR_ID",
                        help="Only attempt these pairs; the others are recorded as not_attempted")
    parser.add_argument("--check", type=Path, metavar="DIR",
                        help="Compare the written manifests byte by byte with DIR")
    args = parser.parse_args()
    queue, queue_sha, combined = verify_queue()
    known = [row["pair_id"] for row in queue]
    unknown = set(args.pairs or ()) - set(known)
    if unknown:
        parser.error("unknown pair IDs: " + ", ".join(sorted(unknown)))
    args.output.mkdir(parents=True, exist_ok=False)
    work = args.work_dir.resolve()
    records = prepare(queue, set(args.pairs or known), work)
    files = {"windows.jsonl": dump_jsonl(records), "summary.json": dump_json(summarize(records, queue_sha, combined))}
    for name in OUTPUTS:
        (args.output / name).write_text(files[name], encoding="utf-8")
    summary = json.loads(files["summary.json"])
    print(f"decoded {summary['windows_decoded']}/{summary['windows_planned']} windows, "
          f"{summary['pairs_with_frames']}/{summary['pairs_planned']} pairs with frames", flush=True)
    if args.check:
        differ = [name for name in OUTPUTS if (args.check / name).read_bytes() != files[name].encode()]
        if differ:
            raise SystemExit("differs from " + str(args.check) + ": " + ", ".join(differ))
        print("identical to", args.check, flush=True)


if __name__ == "__main__":
    main()
