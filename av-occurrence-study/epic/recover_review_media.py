"""Retrieve small public EPIC media windows via a resumable byte-range cache.

Run: python3 recover_review_media.py P30_08

P30_08-19_2 uses 133.133–145.760 s. The source has malformed packets at
132.699–132.716 s and 145.796–145.812 s, outside the annotated sound and action.
Seek to the 133.133 s keyframe and stream-copy the bounded interval with
ffmpeg -ss 133.133 -i SOURCE -t 12.627 -map 0:v:0 -map 0:a:0 -c copy INTERMEDIATE.
Then strictly decode the intermediate and encode H.264/AAC, preserving native
59.94 fps. This keeps the last observed frame beyond the action's 145.730 s end.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import re
import struct
import subprocess
import tempfile
import time
from urllib.parse import urlparse
from clip_bounds import CLEAN_WINDOWS


ROOT = Path(__file__).resolve().parent
DEFAULT_CACHE = Path(tempfile.gettempdir()) / "av-occurrence-epic-range-cache"
RANGE_HEADER = re.compile(r"^content-range: bytes (\d+)-(\d+)/(\d+)\s*$", re.I | re.M)
BLOCK = 768 * 1024
MAX_GAP = 32 * 1024


class Download:
    def __init__(self, url, cache):
        parsed = urlparse(url)
        if parsed.scheme != "https" or parsed.hostname != "data.bris.ac.uk":
            raise ValueError("Unexpected origin for EPIC media")
        self.url = url
        self.directory = cache / hashlib.sha256(url.encode()).hexdigest()[:20]
        self.directory.mkdir(parents=True, exist_ok=True)

    def get(self, start, end):
        chunk = self.directory / f"{start}-{end}.part"
        completed = chunk.with_suffix(".bin")
        totalfile = self.directory / "size"
        if completed.is_file():
            known_total = int(totalfile.read_text())
            data = completed.read_bytes()
            if len(data) == min(known_total - 1, end) - start + 1:
                return start, data, known_total
            # Refuse to silently use an incompatible cached source.
            raise ValueError(f"Invalid cached range {start}-{end}")
        reason = "incomplete range"
        for attempt in range(5):
            have = chunk.stat().st_size if chunk.exists() else 0
            next_byte = start + have
            if totalfile.exists():
                total = int(totalfile.read_text())
                if next_byte == min(total, end + 1):
                    chunk.replace(completed)
                    return start, completed.read_bytes(), total
            with tempfile.TemporaryDirectory(dir=self.directory) as temp:
                headers = Path(temp) / "headers"
                payload = Path(temp) / "payload"
                result = subprocess.run([
                    "/usr/bin/curl", "-fsS", "--connect-timeout", "15", "--max-time", "55",
                    "--speed-limit", "16384", "--speed-time", "20", "--proto", "=https",
                    "--range", f"{next_byte}-{end}", "--dump-header", str(headers),
                    "--output", str(payload), self.url,
                ], capture_output=True, text=True)
                reason = result.stderr.strip() or "server omitted matching range metadata"
                matches = RANGE_HEADER.findall(headers.read_text(errors="replace")) if headers.exists() else []
                if matches:
                    lo, hi, total = map(int, matches[-1])
                    if lo == next_byte and hi == min(end, total - 1) and payload.is_file():
                        received = payload.stat().st_size
                        if received <= hi - lo + 1:
                            if not totalfile.exists():
                                # The initial header request creates this before worker requests.
                                totalfile.write_text(str(total))
                            if int(totalfile.read_text()) != total:
                                raise ValueError("Remote source size changed while downloading")
                            if received:
                                with chunk.open("ab") as current, payload.open("rb") as piece:
                                    current.write(piece.read())
                            if received == hi - lo + 1:
                                chunk.replace(completed)
                                return start, completed.read_bytes(), total
            if attempt < 4:
                time.sleep(min(8, 2**attempt))
        raise ValueError(f"curl range {start}-{end} failed after 5 attempts: {reason}")

    def write_ranges(self, file, ranges, workers=5):
        blocks = [
            (start, min(end, start + BLOCK) - 1)
            for low, end in ranges
            for start in range(low, end, BLOCK)
        ]
        failures = []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for future in as_completed([pool.submit(self.get, *block) for block in blocks]):
                try:
                    start, data, _ = future.result()
                except ValueError as exc:
                    failures.append(str(exc))
                else:
                    os.pwrite(file.fileno(), data, start)
        if failures:
            raise ValueError(f"{len(failures)} failed ranges; {failures[0]}")

    def index(self, file):
        _, head, total = self.get(0, 31)
        file.truncate(total)
        position, metadata = 0, []
        while position < total:
            if position:
                _, head, _ = self.get(position, min(position + 31, total - 1))
            if len(head) < 8:
                raise ValueError(f"Truncated MP4 box at {position}")
            size, box_type = struct.unpack(">I4s", head[:8])
            if size == 1:
                size = struct.unpack(">Q", head[8:16])[0]
            if size == 0:
                size = total - position
            if size < 8 or position + size > total:
                raise ValueError(f"Malformed MP4 box {box_type!r}, offset {position}, size {size}")
            os.pwrite(file.fileno(), head, position)
            if box_type in (b"ftyp", b"moov"):
                metadata.append((position, position + size))
            position += size
        if not metadata:
            raise ValueError("Missing MP4 metadata")
        self.write_ranges(file, metadata)


def merge(ranges):
    result = []
    for start, end in sorted(ranges):
        if result and start <= result[-1][1] + MAX_GAP:
            result[-1][1] = max(end, result[-1][1])
        else:
            result.append([start, end])
    return result


def media_windows(candidate):
    for event in candidate["occurrences"]:
        output = ROOT / "review_media" / f"{candidate['pair_id']}_{event['ordinal']}.mp4"
        start = max(0, event["sound_s"][0] - 3)
        stop = min(candidate["duration_s"], max(event["action_s"][0] + 5, event["sound_s"][1] + 6))
        clean = CLEAN_WINDOWS.get((candidate["pair_id"], event["ordinal"]))
        yield output, start, stop, clean, event["action_s"][1]


def process(candidate, cache, windows=None):
    if windows is None:
        windows = media_windows(candidate)
    missing = [window for window in windows if not window[0].is_file()]
    if not missing:
        return True
    download = Download(candidate["media_url"], cache)
    with tempfile.TemporaryFile() as source:
        download.index(source)
        device = f"/dev/fd/{source.fileno()}"

        def ff(command, seconds=50):
            result = subprocess.run(command, pass_fds=(source.fileno(),), capture_output=True, timeout=seconds)
            if result.returncode:
                raise ValueError(result.stderr.decode(errors="replace")[-400:])
            return result.stdout

        index = json.loads(ff([
            "ffprobe", "-v", "quiet", "-show_entries", "stream=index,codec_type",
            "-of", "json", device,
        ]))["streams"]
        stream_ids = set()
        for kind in ("video", "audio"):
            track = next((item for item in index if item["codec_type"] == kind), None)
            if track is None:
                raise ValueError(f"MP4 lacks a {kind} track")
            stream_ids.add(track["index"])

        success = True
        for output, start, stop, clean, action_end in missing:
            if output.exists():
                continue
            try:
                packets = json.loads(ff([
                    "ffprobe", "-v", "quiet", "-read_intervals", f"{max(0,start-2)}%{stop+3}",
                    "-show_entries", "packet=pos,size,stream_index", "-of", "json", device,
                ]))["packets"]
                selected = [
                    (int(p["pos"]), int(p["pos"]) + int(p["size"]))
                    for p in packets
                    if p.get("stream_index") in stream_ids and p.get("pos", "-1").isdigit()
                ]
                if not selected:
                    raise ValueError("No indexed video or audio packets")
                ranges = merge(selected)
                need = sum(hi - lo for lo, hi in ranges)
                span = ranges[-1][1] - ranges[0][0]
                if need > 200 * 1024**2:
                    raise ValueError(f"Packet ranges above 200 MiB for one short clip ({need})")
                print("fetch", output.stem, f"{need/1024**2:.1f} MiB in {len(ranges)} ranges (span {span/1024**2:.1f} MiB)", flush=True)
                download.write_ranges(source, ranges)
                # Link only a completed, independently checked file, and never overwrite another runner's result.
                with tempfile.TemporaryDirectory(prefix=".range-v2-", dir=output.parent) as temp:
                    part = Path(temp) / "clip.mp4"
                    if clean is None:
                        read_args = ["-ss", str(start), "-i", device, "-t", str(stop - start)]
                        video_filter = "fps=5,scale=480:-2"
                    else:
                        start, stop = clean
                        intermediate = Path(temp) / "bounded.mp4"
                        # Packet-level clipping prevents the threaded decoder reading into corrupted following packets.
                        ff([
                            "ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-ss", str(start), "-i", device,
                            "-t", f"{stop - start:.6f}", "-map", "0:v:0", "-map", "0:a:0",
                            "-c", "copy", str(intermediate),
                        ])
                        read_args = ["-i", str(intermediate)]
                        video_filter = "scale=480:-2"
                    ff([
                        "ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-xerror", *read_args,
                        "-map", "0:v:0", "-map", "0:a:0", "-vf", video_filter, "-c:v", "libx264", "-preset", "veryfast",
                        "-ac", "1", "-ar", "24000", "-c:a", "aac", "-movflags", "+faststart", str(part),
                    ])
                    info = json.loads(ff([
                        "ffprobe", "-v", "error", "-show_entries", "stream=codec_type,duration",
                        "-of", "json", str(part),
                    ]))
                    lengths = {track["codec_type"]: float(track["duration"]) for track in info["streams"]}
                    if any(abs(lengths.get(kind, 0) - (stop - start)) > 0.6 for kind in ("video", "audio")):
                        raise ValueError(f"Output stream durations do not match expected {stop-start:.3f}: {lengths}")
                    ff(["ffmpeg", "-v", "error", "-xerror", "-i", str(part), "-map", "0:v:0", "-map", "0:a:0", "-f", "null", "-"])
                    if clean is not None:
                        frames = json.loads(ff([
                            "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                            "packet=pts_time", "-of", "json", str(part),
                        ]))["packets"]
                        latest = start + max(float(frame["pts_time"]) for frame in frames)
                        if latest < action_end:
                            raise ValueError(f"Latest decoded frame {latest:.6f} falls before action end {action_end:.6f}")
                    try:
                        os.link(part, output)
                    except FileExistsError:
                        pass
                    print("valid", output.stem, f"{start:.3f}–{stop:.3f} s", lengths, flush=True)
            except (ValueError, OSError, subprocess.SubprocessError, KeyError) as exc:
                success = False
                print("failed window", output.stem, repr(exc), flush=True)
        return success


def work(candidate, cache):
    try:
        return process(candidate, cache)
    except (ValueError, OSError, subprocess.SubprocessError, KeyError, struct.error) as exc:
        print("failed source", candidate["pair_id"], repr(exc), flush=True)
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("videos", nargs="*")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    args = parser.parse_args()
    queue = [json.loads(line) for line in (ROOT / "review_queue.jsonl").read_text().splitlines() if line.strip()]
    unknown = set(args.videos) - {candidate["video_id"] for candidate in queue}
    if unknown:
        parser.error("unknown video IDs: " + ", ".join(sorted(unknown)))
    subset = [candidate for candidate in queue if not args.videos or candidate["video_id"] in args.videos]
    (ROOT / "review_media").mkdir(exist_ok=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        success = all(list(pool.map(lambda candidate: work(candidate, args.cache_dir), subset)))
    raise SystemExit(0 if success else 1)
