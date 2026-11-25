"""Check that the public EPIC originals of the confirmation_1124 batch can be reviewed.

For every pair in the frozen queue (manifest/selected.jsonl) this script records,
from the official data.bris.ac.uk URL:

* the HTTP response (status, size, type, byte-range support, ETag);
* the MP4 top-level layout, notably where the index (moov) sits;
* the video and audio tracks declared by the index;
* index-level continuity of both tracks from 0 s to the end of the second
  candidate's following action plus a margin, and the bytes a reviewer must
  receive to reach that point;
* a decode of each candidate window (sound and following action) with the
  audio level inside the annotated sound interval;
* with --full, a strict decode of both tracks from 0 s to that point.

Downloaded bytes are kept in a sparse local mirror under --cache, so repeated
runs resume. Nothing is published, and nothing here judges which sound is heard
or which action follows: those remain the reviewers' decisions.

Run from the AgentSpace root with Python 3.10+, curl and FFmpeg:

    python3 av-occurrence-study/run/confirmation_1124/blind_review_readiness/check_sources.py \
        --output /tmp/blind-review-sources.json [--full]
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import re
import struct
import subprocess
import tempfile
import threading
import time
from urllib.parse import urlparse

HERE = Path(__file__).resolve().parent
BATCH = HERE.parent
QUEUE = BATCH / "manifest" / "selected.jsonl"
PROVENANCE = BATCH / "manifest" / "provenance.json"
DEFAULT_CACHE = Path(tempfile.gettempdir()) / "confirmation-1124-source-mirror"
CHUNK = 16 * 1024 * 1024
RANGE_HEADER = re.compile(r"^content-range: bytes (\d+)-(\d+)/(\d+)\s*$", re.I | re.M)
ETAG_HEADER = re.compile(r'^etag: (".*?")\s*$', re.I | re.M)
WINDOW_LEAD_S = 3.0
WINDOW_TAIL_S = 3.0
KEYFRAME_LEAD_S = 3.0
# Decoders read a few packets past -t; fetch them so no zero-filled packet is parsed.
DECODE_LOOKAHEAD_S = 1.5
# Real bytes at the head of the file for FFmpeg stream probing.
PROBE_BYTES = 8 * 1024 * 1024


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def curl(*args, timeout=120):
    return subprocess.run(["curl", "-sS", "--proto", "=https", "--connect-timeout", "20",
                           "--max-time", str(timeout), *args], capture_output=True)


def head(url):
    started = time.monotonic()
    result = curl("-I", url, timeout=60)
    elapsed = round(time.monotonic() - started, 3)
    if result.returncode:
        return {"ok": False, "error": result.stderr.decode(errors="replace").strip(), "seconds": elapsed}
    text = result.stdout.decode(errors="replace")
    status = int(text.split()[1])
    fields = {}
    for line in text.splitlines()[1:]:
        if ":" in line:
            key, value = line.split(":", 1)
            fields[key.strip().lower()] = value.strip()
    return {
        "ok": status == 200,
        "status": status,
        "content_type": fields.get("content-type"),
        "content_length": int(fields.get("content-length", 0)),
        "accept_ranges": fields.get("accept-ranges"),
        "last_modified": fields.get("last-modified"),
        "etag": fields.get("etag"),
        "seconds": elapsed,
    }


class Mirror:
    """Sparse local copy of a remote file, filled by verified byte ranges."""

    def __init__(self, url, size, etag, cache):
        parsed = urlparse(url)
        if parsed.scheme != "https" or parsed.hostname != "data.bris.ac.uk":
            raise ValueError("Unexpected origin for EPIC media: " + url)
        self.url, self.size, self.etag = url, size, etag
        directory = cache / hashlib.sha256(url.encode()).hexdigest()[:20]
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / Path(parsed.path).name
        self.ledger = directory / "ranges.json"
        identity = {"url": url, "size": size, "etag": etag}
        state = json.loads(self.ledger.read_text()) if self.ledger.exists() else None
        if not state or state["identity"] != identity or not self.path.exists():
            # A different remote object: start afresh instead of mixing bytes.
            with open(self.path, "wb") as handle:
                handle.truncate(size)
            state = {"identity": identity, "ranges": []}
            self.ledger.write_text(json.dumps(state))
        self.ranges = [tuple(r) for r in state["ranges"]]
        self.lock = threading.Lock()
        self.received = 0
        self.fetch_seconds = 0.0

    def _record(self, start, end):
        with self.lock:
            merged = []
            for lo, hi in sorted(self.ranges + [(start, end)]):
                if merged and lo <= merged[-1][1]:
                    merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
                else:
                    merged.append((lo, hi))
            self.ranges = merged
            self.ledger.write_text(json.dumps({"identity": {"url": self.url, "size": self.size, "etag": self.etag},
                                               "ranges": self.ranges}))

    def missing(self, start, end):
        gaps, cursor = [], start
        for lo, hi in self.ranges:
            if hi <= cursor or lo >= end:
                continue
            if lo > cursor:
                gaps.append((cursor, lo))
            cursor = max(cursor, hi)
        if cursor < end:
            gaps.append((cursor, end))
        return gaps

    def _fetch(self, start, end):
        last = "no attempt"
        for attempt in range(6):
            with tempfile.TemporaryDirectory() as temp:
                headers, payload = Path(temp) / "h", Path(temp) / "p"
                began = time.monotonic()
                result = curl("--range", f"{start}-{end - 1}", "--dump-header", str(headers),
                              "--speed-limit", "16384", "--speed-time", "30",
                              "--output", str(payload), self.url, timeout=900)
                text = headers.read_text(errors="replace") if headers.exists() else ""
                ranges, etags = RANGE_HEADER.findall(text), ETAG_HEADER.findall(text)
                data = payload.read_bytes() if payload.exists() else b""
                if (result.returncode == 0 and ranges and tuple(map(int, ranges[-1])) == (start, end - 1, self.size)
                        and (not etags or etags[-1] == self.etag) and len(data) == end - start):
                    with open(self.path, "r+b") as handle:
                        handle.seek(start)
                        handle.write(data)
                    with self.lock:
                        self.received += len(data)
                        self.fetch_seconds += time.monotonic() - began
                    self._record(start, end)
                    return
                last = result.stderr.decode(errors="replace").strip() or "range metadata mismatch"
            time.sleep(2 * (attempt + 1))
        raise RuntimeError(f"{self.url} bytes {start}-{end - 1}: {last}")

    def ensure(self, start, end, connections=8):
        jobs = []
        for lo, hi in self.missing(start, min(end, self.size)):
            jobs += [(a, min(a + CHUNK, hi)) for a in range(lo, hi, CHUNK)]
        with ThreadPoolExecutor(max(1, connections)) as pool:
            for future in [pool.submit(self._fetch, a, b) for a, b in jobs]:
                future.result()
        return len(jobs)


def atoms(mirror):
    """Walk top-level MP4 boxes with small range requests."""
    layout, offset = [], 0
    while offset < mirror.size:
        mirror.ensure(offset, min(offset + 16, mirror.size), 1)
        with open(mirror.path, "rb") as handle:
            handle.seek(offset)
            raw = handle.read(16)
        size, kind = struct.unpack(">I4s", raw[:8])
        if size == 1:
            size = struct.unpack(">Q", raw[8:16])[0]
        elif size == 0:
            size = mirror.size - offset
        layout.append({"type": kind.decode(errors="replace"), "offset": offset, "size": size})
        offset += size
    return layout, offset == mirror.size


def index_tables(mirror, moov):
    """Summarise each track's handler, sync-sample count and edit list from the moov box."""
    with open(mirror.path, "rb") as handle:
        handle.seek(moov["offset"])
        data = handle.read(moov["size"])
    tracks = []

    def walk(start, end):
        while start + 8 <= end:
            size, kind = struct.unpack(">I4s", data[start:start + 8])
            header = 8
            if size == 1:
                size, header = struct.unpack(">Q", data[start + 8:start + 16])[0], 16
            if size < header:
                return
            kind = kind.decode(errors="replace")
            body = start + header
            if kind == "trak":
                tracks.append({})
            elif kind == "hdlr" and tracks:
                tracks[-1]["handler"] = data[body + 8:body + 12].decode(errors="replace")
            elif kind == "stss" and tracks:
                count = struct.unpack(">I", data[body + 4:body + 8])[0]
                first = [struct.unpack(">I", data[body + 8 + 4 * i:body + 12 + 4 * i])[0] for i in range(min(count, 3))]
                tracks[-1]["sync_samples"] = count
                tracks[-1]["first_sync_samples"] = first
            elif kind == "elst" and tracks:
                version, count = data[body], struct.unpack(">I", data[body + 4:body + 8])[0]
                width = 8 if version else 4
                entries = []
                for i in range(count):
                    at = body + 8 + i * (2 * width + 4)
                    duration = int.from_bytes(data[at:at + width], "big")
                    media_time = int.from_bytes(data[at + width:at + 2 * width], "big", signed=True)
                    entries.append({"segment_duration": duration, "media_time": media_time})
                tracks[-1]["edit_list"] = entries
            if kind in ("moov", "trak", "mdia", "minf", "stbl", "edts"):
                walk(body, start + size)
            start += size

    walk(0, len(data))
    return tracks


def ffprobe(*args):
    result = subprocess.run(["ffprobe", "-v", "error", *args], capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(result.stderr.strip())
    return result.stdout


def packets(path, selector):
    text = ffprobe("-select_streams", selector, "-show_entries",
                   "packet=pts_time,duration_time,pos,size,flags", "-of", "csv=p=0", str(path))
    rows = []
    # ffprobe prints packet fields in its own order: pts_time, duration_time, size, pos, flags.
    for pts, duration, size, pos, _flags in csv.reader(io.StringIO(text)):
        if pts != "N/A":
            rows.append((float(pts), float(duration), int(pos), int(size)))
    return sorted(rows)


def continuity(rows, until):
    used = [row for row in rows if row[0] <= until]
    gaps = [{"after_s": a[0], "gap_s": round(b[0] - a[0] - a[1], 6)}
            for a, b in zip(used, used[1:]) if abs(b[0] - a[0] - a[1]) > 0.002]
    return {
        "packets": len(used),
        "first_pts_s": used[0][0] if used else None,
        "last_pts_s": used[-1][0] if used else None,
        "gaps_over_2ms": gaps,
        "byte_end": max(p[2] + p[3] for p in used) if used else 0,
    }


def window_bytes(video, audio, start_s, end_s):
    # Keyframe flags read from a sparse mirror are unreliable (parsers inspect the
    # zero-filled payload), so take a lead longer than the ~1 s sync-sample spacing.
    begin = max(0.0, start_s - KEYFRAME_LEAD_S)
    chosen = [row for row in video + audio if begin <= row[0] <= end_s + DECODE_LOOKAHEAD_S]
    return min(p[2] for p in chosen), max(p[2] + p[3] for p in chosen)


def ffmpeg(args, level="error"):
    result = subprocess.run(["ffmpeg", "-hide_banner", "-nostdin", "-v", level, "-stats_period", "1",
                             "-progress", "pipe:1", *args], capture_output=True, text=True)
    progress = {}
    for line in result.stdout.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            progress[key] = value
    errors = [line for line in result.stderr.splitlines() if line.strip()]
    return result.returncode, progress, errors


def decode_check(path, start_s, duration_s, stream):
    """Decode one stream strictly; report decoded frames and decoder messages."""
    args = ["-ss", f"{start_s:.3f}", "-t", f"{duration_s:.3f}", "-i", str(path),
            "-map", f"0:{stream}:0", "-f", "null", "-"]
    code, progress, errors = ffmpeg(args)
    return {"exit": code, "frames": int(progress.get("frame", 0) or 0) if stream == "v" else None,
            "out_time_s": round(int(progress.get("out_time_us", 0) or 0) / 1e6, 3),
            "decoder_messages": len(errors), "first_messages": errors[:5]}


def loudness(path, start_s, end_s):
    code, _, errors = ffmpeg(["-ss", f"{start_s:.3f}", "-t", f"{end_s - start_s:.3f}", "-i", str(path),
                              "-map", "0:a:0", "-af", "volumedetect", "-f", "null", "-"], level="info")
    found = {}
    for line in errors:
        match = re.search(r"(mean|max)_volume: (-?[\d.]+|-inf) dB", line)
        if match:
            found[match.group(1) + "_db"] = None if match.group(2) == "-inf" else float(match.group(2))
    return found


def stream_info(path):
    data = json.loads(ffprobe("-show_entries",
                              "format=duration:stream=index,codec_type,codec_name,codec_tag_string,profile,"
                              "width,height,avg_frame_rate,sample_rate,channels,duration,start_time",
                              "-of", "json", str(path)))
    keep = []
    for stream in data["streams"]:
        if stream.get("codec_type") in ("video", "audio"):
            keep.append({key: stream[key] for key in ("index", "codec_type", "codec_name", "codec_tag_string",
                                                       "profile", "width", "height", "avg_frame_rate",
                                                       "sample_rate", "channels", "start_time", "duration")
                         if key in stream})
    return float(data["format"]["duration"]), keep


def check(pair, cache, margin, full, connections):
    report = {"pair_id": pair["pair_id"], "video_id": pair["video_id"], "media_url": pair["media_url"]}
    response = head(pair["media_url"])
    report["http"] = response
    if not response["ok"]:
        report["failure"] = "source unreachable: " + str(response.get("error") or response.get("status"))
        return report
    mirror = Mirror(pair["media_url"], response["content_length"], response["etag"], cache)
    layout, complete = atoms(mirror)
    report["mp4_boxes"] = layout
    report["mp4_boxes_cover_file"] = complete
    moov = next(box for box in layout if box["type"] == "moov")
    mdat = next(box for box in layout if box["type"] == "mdat")
    report["index_position"] = "start" if moov["offset"] < mdat["offset"] else "end"
    mirror.ensure(0, min(PROBE_BYTES, mirror.size), 1)
    mirror.ensure(moov["offset"], moov["offset"] + moov["size"], 1)
    duration, streams = stream_info(mirror.path)
    report["index_tracks"] = index_tables(mirror, moov)
    report["duration_s"], report["streams"] = duration, streams

    video, audio = packets(mirror.path, "v:0"), packets(mirror.path, "a:0")
    occurrences = pair["occurrences"]
    target = max(o["action_s"][1] for o in occurrences) + margin
    report["review_span_s"] = [0.0, round(target, 3)]
    report["continuity"] = {"video": continuity(video, target), "audio": continuity(audio, target)}
    report["bytes_to_review_end"] = max(report["continuity"]["video"]["byte_end"],
                                        report["continuity"]["audio"]["byte_end"])
    report["mean_bitrate_mbit_s"] = round(response["content_length"] * 8 / duration / 1e6, 2)

    windows = []
    for occurrence in occurrences:
        start_s = occurrence["sound_s"][0] - WINDOW_LEAD_S
        end_s = occurrence["action_s"][1] + WINDOW_TAIL_S
        low, high = window_bytes(video, audio, start_s, end_s)
        mirror.ensure(low, high, connections)
        windows.append({
            "ordinal": occurrence["ordinal"],
            "sound_s": occurrence["sound_s"],
            "action_s": occurrence["action_s"],
            "window_s": [round(start_s, 3), round(end_s, 3)],
            "bytes": [low, high],
            "video": decode_check(mirror.path, start_s, end_s - start_s, "v"),
            "audio": decode_check(mirror.path, start_s, end_s - start_s, "a"),
            "sound_interval_level": loudness(mirror.path, *occurrence["sound_s"]),
        })
    report["windows"] = windows

    if full:
        lookahead = continuity(video + audio, target + DECODE_LOOKAHEAD_S)["byte_end"]
        before, began = mirror.received, time.monotonic()
        mirror.ensure(0, lookahead, connections)
        fetched = time.monotonic() - began
        began = time.monotonic()
        prefix_video = decode_check(mirror.path, 0.0, target, "v")
        prefix_audio = decode_check(mirror.path, 0.0, target, "a")
        report["full_prefix"] = {
            "span_s": [0.0, round(target, 3)],
            "fetched_bytes_this_run": mirror.received - before,
            "fetch_wall_seconds": round(fetched, 1),
            "decode_wall_seconds": round(time.monotonic() - began, 1),
            "index_video_packets": report["continuity"]["video"]["packets"],
            "video": prefix_video,
            "audio": prefix_audio,
        }
    report["throughput_mb_s_per_connection"] = (round(mirror.received / mirror.fetch_seconds / 1e6, 3)
                                                 if mirror.fetch_seconds else None)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--queue", type=Path, default=QUEUE)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--margin", type=float, default=10.0, help="seconds kept after the last action")
    parser.add_argument("--full", action="store_true", help="fetch and decode 0 s to the review end")
    parser.add_argument("--connections", type=int, default=8)
    args = parser.parse_args()

    frozen = json.loads(PROVENANCE.read_text())["manifest_sha256"]["selected.jsonl"]
    queue_sha = sha256(args.queue)
    if args.queue.resolve() == QUEUE.resolve() and queue_sha != frozen:
        raise SystemExit(f"selected.jsonl does not match the frozen SHA-256 {frozen}")
    pairs = [json.loads(line) for line in args.queue.read_text().splitlines() if line.strip()]
    args.cache.mkdir(parents=True, exist_ok=True)
    results = []
    for pair in pairs:
        try:
            results.append(check(pair, args.cache, args.margin, args.full, args.connections))
        except Exception as error:  # keep other pairs; record the failure itself
            results.append({"pair_id": pair["pair_id"], "video_id": pair["video_id"],
                            "media_url": pair["media_url"], "failure": f"{type(error).__name__}: {error}"})
        print(json.dumps({k: results[-1].get(k) for k in ("pair_id", "failure", "bytes_to_review_end")}),
              flush=True)
    output = {
        "checked_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "queue": str(args.queue.relative_to(BATCH)) if args.queue.is_relative_to(BATCH) else str(args.queue),
        "queue_sha256": queue_sha,
        "margin_s": args.margin,
        "full_prefix_decoded": args.full,
        "ffmpeg": subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True).stdout.split("\n")[0],
        "pairs": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=1) + "\n")


if __name__ == "__main__":
    main()
