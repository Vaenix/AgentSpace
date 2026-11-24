"""Build a continuous low-bitrate review copy from the start of an official EPIC video."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


OUTPUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUTPUT.parent))
from recover_review_media import Download, DEFAULT_CACHE, merge


def cached_ranges(download, file, required):
    """Reuse validated ranges from the short-window download, including partial overlaps."""
    for block in sorted(download.directory.glob("*.bin"), key=lambda p: int(p.stem.split("-")[0])):
        start, last = map(int, block.stem.split("-"))
        end = last + 1
        if block.stat().st_size != end - start:
            continue
        remaining = []
        for low, high in required:
            overlap_start, overlap_end = max(low, start), min(high, end)
            if overlap_start >= overlap_end:
                remaining.append((low, high))
                continue
            with block.open("rb") as stream:
                stream.seek(overlap_start - start)
                os.pwrite(file.fileno(), stream.read(overlap_end - overlap_start), overlap_start)
            if low < overlap_start:
                remaining.append((low, overlap_start))
            if overlap_end < high:
                remaining.append((overlap_end, high))
        required = remaining
    return required


def build(candidate, cache, workers):
    video_id = candidate["video_id"]
    target = OUTPUT / f"{video_id}_prefix.mp4"
    if target.is_file():
        return target
    event = candidate["occurrences"][1]
    stop = min(candidate["duration_s"], max(event["sound_s"][1] + 6, event["action_s"][1] + 2))
    download = Download(candidate["media_url"], cache)
    with tempfile.TemporaryFile() as source, tempfile.TemporaryDirectory(dir=OUTPUT) as temporary:
        download.index(source)
        device = f"/dev/fd/{source.fileno()}"

        def ff(*args, timeout=180):
            completed = subprocess.run(args, pass_fds=(source.fileno(),), capture_output=True, timeout=timeout)
            if completed.returncode:
                raise ValueError(completed.stderr.decode(errors="replace")[-1000:])
            return completed.stdout

        stream_info = json.loads(ff("ffprobe", "-v", "quiet", "-show_entries", "stream=index,codec_type", "-of", "json", device))["streams"]
        streams = {next(entry["index"] for entry in stream_info if entry["codec_type"] == kind) for kind in ("video", "audio")}
        packets = json.loads(ff(
            "ffprobe", "-v", "quiet", "-fflags", "+noparse+nofillin", "-read_intervals", f"0%{stop + 3}",
            "-show_entries", "packet=pos,size,stream_index", "-of", "json", device,
        ))["packets"]
        ranges = merge([
            (int(packet["pos"]), int(packet["pos"]) + int(packet["size"]))
            for packet in packets if packet.get("stream_index") in streams and packet.get("pos", "-1").isdigit()
        ])
        if not ranges:
            raise ValueError(f"No indexed video/audio samples for {video_id}")
        needed = cached_ranges(download, source, ranges)
        remaining_size = sum(end - start for start, end in needed)
        print(f"{video_id}: {remaining_size / 1024**2:.1f} MiB of uncached video/audio samples", flush=True)
        download.write_ranges(source, needed, workers=workers)

        part = Path(temporary) / "prefix.mp4"
        ff(
            "ffmpeg", "-nostdin", "-v", "error", "-xerror", "-i", device, "-t", f"{stop:.3f}",
            "-map", "0:v:0", "-map", "0:a:0", "-map_metadata", "-1", "-map_chapters", "-1",
            "-vf", "fps=8,scale=480:-2", "-c:v", "libx264", "-preset", "veryfast",
            "-crf", "27", "-maxrate", "260k", "-bufsize", "520k", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-ar", "32000", "-b:a", "56k", "-movflags", "+faststart", str(part),
        )
        lengths = {
            track["codec_type"]: float(track["duration"])
            for track in json.loads(ff(
                "ffprobe", "-v", "error", "-show_entries", "stream=codec_type,duration", "-of", "json", str(part),
            ))["streams"]
        }
        if any(abs(lengths.get(kind, 0) - stop) > 0.2 for kind in ("video", "audio")):
            raise ValueError(f"{video_id} expected 0–{stop:.3f} s; got {lengths}")
        ff("ffmpeg", "-v", "error", "-xerror", "-i", str(part), "-map", "0:v:0", "-map", "0:a:0", "-f", "null", "-")
        os.link(part, target)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video_id")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    queue = [json.loads(line) for line in (OUTPUT.parent / "review_queue.jsonl").read_text().splitlines()]
    candidate = next((row for row in queue if row["video_id"] == args.video_id), None)
    if candidate is None:
        parser.error(f"Unknown review video: {args.video_id}")
    try:
        print(build(candidate, args.cache_dir, args.workers), flush=True)
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
