"""Scan one fixed public TraceAV annotation snapshot; no model prediction or media decoding."""
import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

REVISION = "3b99330137750531919616c4c37156a40473148c"
REPO = "https://huggingface.co/datasets/Heinz217/TraceAV-Bench"
CUES = {
    "bell": r"\b(?:bells?|chimes?|ding(?:ing|s)?)\b",
    "beep": r"\b(?:beeps?|beeping|buzzers?)\b",
    "applause": r"\b(?:applaud\w*|applause|clapping)\b",
    "whistle": r"\bwhistl(?:es?|ing)\b",
    "door_knock": r"\bknock(?:ing|s)?\b",
    "door_slam": r"\b(?:door\w* (?:slams?|slam(?:med|ming))|slamm?\w* (?:the |a )?door)\b",
    "siren": r"\bsirens?\b",
    "horn": r"\b(?:horns?|honks?|honking)\b",
    "animal": r"\b(?:barks?|barking|meows?|meowing)\b",
    "splash": r"\bsplash(?:es|ing)?\b",
    "gunshot": r"\b(?:gunshots?|gunfire)\b",
    "explosion": r"\bexplosions?\b",
    "telephone": r"\b(?:telephone|phone) (?:rings?|ringing)\b",
    "laughter": r"\b(?:laughter|laughing)\b",
}
def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

def scan(root, out):
    files = sorted(root.glob("*.jsonl"))
    if len(files) != 16:
        raise ValueError("Expected 15 task files and the media mapping.")
    maps = {r["video_id"]:r for r in read_rows(root/"video_name_mapping.jsonl")}
    evidence = defaultdict(list)
    manifest = {}
    seen_videos = set()
    n_questions = 0
    n_hops = 0
    for file in files:
        rows = read_rows(file)
        manifest[file.name] = {"rows":len(rows),"sha256":hashlib.sha256(file.read_bytes()).hexdigest()}
        if file.name == "video_name_mapping.jsonl":
            continue
        for row in rows:
            n_questions += 1
            seen_videos.add(row["video_id"])
            hops = row["trajectory_with_timestamps"]
            if isinstance(hops, str):
                hops = json.loads(hops)
            for hop in hops:
                n_hops += 1
                minute = hop["timestamp_minute"]
                for cue, regex in CUES.items():
                    if re.search(regex,hop["evidence"],re.I):
                        evidence[(row["video_id"],cue,minute)].append({
                            "source_task":row["task_type"],"source_qid":row["question_id"],"event_id":hop["event_id"],
                            "text":hop["evidence"],"label":hop["label"]
                        })
    by_video_cue = defaultdict(dict)
    for (video,cue,minute), mentions in evidence.items():
        by_video_cue[video,cue][minute]=mentions
    hits = []
    for (video,cue), minutes in sorted(by_video_cue.items()):
        if len(minutes)<2:
            continue
        src=maps[video]
        hits.append({"video_id":video,"source":src["source"],"media_id":src["id"],"cue_family":cue,"minute_mentions":minutes})
    out.mkdir(exist_ok=True,parents=True)
    (out/"scan_hits.jsonl").write_text("".join(json.dumps(x,ensure_ascii=False)+"\n" for x in hits))
    count = dict(questions=n_questions, unique_question_videos=len(seen_videos), input_evidence_mentions=n_hops,
                 distinct_video_cue_hits=len(hits), distinct_hit_videos=len({x['video_id'] for x in hits}))
    (out/"snapshot.json").write_text(json.dumps(dict(dataset=REPO,revision=REVISION,files=manifest,counts=count),indent=2)+"\n")
    print(count)
if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("cache",type=Path);p.add_argument("output",type=Path);a=p.parse_args()
    scan(a.cache,a.output)
