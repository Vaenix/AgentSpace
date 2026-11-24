"""Audit the 20 proposed occurrences using official annotation clocks and local audio."""
import array
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess

from screen import interval


root = Path(__file__).resolve().parent
read_jsonl = lambda p: [json.loads(line) for line in p.read_text().splitlines()]
clips = {(r["pair_id"], r["ordinal"]): r for r in read_jsonl(root / "available_media.jsonl")}
with (root / "annotations/actions.csv").open() as file:
    actions = [(r["video_id"], r["narration_id"], *interval(r)) for r in csv.DictReader(file)]
with (root / "annotations/sounds.csv").open() as file:
    sounds = [(r["video_id"], r["class"], *interval(r)) for r in csv.DictReader(file)]
rows = []
for pair in read_jsonl(root / "review_queue.jsonl"):
    vid, cue = pair["video_id"], pair["sound_class"]
    for event in pair["occurrences"]:
        clip = clips[pair["pair_id"], event["ordinal"]]
        source = root / clip["path"]
        if hashlib.sha256(source.read_bytes()).hexdigest() != clip["sha256"]:
            raise ValueError(f"Changed media file {source}")
        onset, offset = event["sound_s"]
        a, b = event["action_s"]
        pcm = subprocess.check_output([
            "ffmpeg", "-nostdin", "-v", "error", "-xerror", "-i", str(source),
            "-ss", str(onset - clip["start_s"]), "-t", str(offset - onset), "-map", "0:a:0",
            "-ac", "1", "-ar", "16000", "-f", "s16le", "-",
        ])
        samples = array.array("h", pcm)
        if not samples or abs(len(samples) / 16000 - (offset - onset)) > 0.08:
            raise ValueError(f"Missing audio inside {source.name} target interval.")
        rms = math.sqrt(sum(x*x for x in samples) / len(samples))
        other_sounds = sorted({kind for v,kind,t0,t1 in sounds
                               if v == vid and kind != cue and max(t0,onset) < min(t1,offset)})
        active = [id_ for v,id_,t0,t1 in actions if v == vid and t0 < a < t1 and id_ != event["action_id"]]
        rows.append(dict(pair_id=pair["pair_id"], ordinal=event["ordinal"], cue=cue,
                         action_delay_s=round(a-offset,3), other_sound_classes="|".join(other_sounds),
                         ongoing_actions_at_target_start=len(active), original_clip_tail_s=round(clip["end_s"]-b,3),
                         anchor_audio_rms_dbfs=round(20*math.log10(rms/32768),2) if rms else "-inf"))
with (root / "candidate_audit_1124.csv").open("w", newline="") as out:
    writer = csv.DictWriter(out, fieldnames=rows[0])
    writer.writeheader()
    writer.writerows(rows)
print(f"{len(rows)} anchor and action windows checked")
