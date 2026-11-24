"""Project audio votes on the same grid and compute equal-weight reciprocal rank fusion."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import time

HERE = Path(__file__).resolve().parent


def read_csv(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    with path.open("w") as f:
        writer = csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def competition_ranks(values):
    ordered = sorted({v for v in values if v is not None}, reverse=True)
    position = 1
    ranks = {}
    for val in ordered:
        ranks[val] = position
        position += values.count(val)
    return [ranks.get(x) for x in values]


def export_votes(tsv, rules, dest):
    votes = []
    with tsv.open() as f:
        for lo, duration, label, confidence in csv.reader(f,delimiter="\t"):
            if label in rules["audio_classes"] and float(confidence) >= rules["audio_threshold"]:
                votes.append({"start_s":float(lo),"end_s":float(lo)+float(duration),
                              "label":label,"confidence":float(confidence)})
    write_csv(dest,votes)


def rank(inputs, output, detection=None):
    rules = json.loads((HERE/"rules.json").read_text())
    output.mkdir(exist_ok=True,parents=True)
    votes_path = inputs/"audio_votes.csv"
    if detection is not None:
        votes_path = output/"audio_votes.csv"
        export_votes(detection,rules,votes_path)
    votes = read_csv(votes_path)
    media = json.loads((inputs/"runtime.json").read_text())
    cosines = {int(r["requested_s"]):float(r["clip_cosine"]) for r in read_csv(inputs/"frame_cosines.csv")}
    start = time.perf_counter()
    rows = []
    frame_times = {}
    for i in range(media["grid_windows"]):
        lo = i * rules["stride_seconds"]
        hi = min(lo+rules["grid_seconds"],media["video_duration_s"])
        frames = [lo+j for j in rules["frame_offsets_seconds"]]
        if any(t >= hi for t in frames):
            raise ValueError("Frame time falls outside its grid window.")
        visual = sum(cosines[t] for t in frames) / len(frames)
        match = []
        for r in votes:
            a,b = float(r["start_s"]),float(r["end_s"])
            overlap = min(b,hi) - max(a,lo)
            if overlap>0:
                match.append(float(r["confidence"])*overlap/(b-a))
        ident = f"W{i+1:03}"
        frame_times[ident]=frames
        rows.append({"window_id":ident,"start_s":lo,"end_s":hi,"clip_mean_cosine":visual,
                     "audio_overlap_score":max(match,default=None)})
    for key, rank_key in (("clip_mean_cosine","vision_rank"),("audio_overlap_score","audio_rank")):
        for row, pos in zip(rows,competition_ranks([r[key] for r in rows])):
            row[rank_key] = pos
    for r in rows:
        a=r["audio_rank"]
        r["audio_contribution"] = 0 if a is None else 0.5/(60+a)
        r["fusion_score"] = 0.5/(60+r["vision_rank"])+r["audio_contribution"]
    for r, pos in zip(rows,competition_ranks([r["fusion_score"] for r in rows])):
        r["fusion_rank"]=pos
    selected = {}
    for arm, key in (("vision_only","clip_mean_cosine"),("vision_plus_sound","fusion_score")):
        chosen = []
        for r in sorted(rows,key=lambda x:(-x[key],x["start_s"])):
            if not any(r["start_s"]<c["end_s"] and r["end_s"]>c["start_s"] for c in chosen):
                chosen.append(r)
            if len(chosen)==rules["top_k"]:
                break
        selected[arm]=[dict(r,selection_rank=p+1,frame_times_s=frame_times[r["window_id"]])
                       for p,r in enumerate(chosen)]
    elapsed = time.perf_counter()-start
    write_csv(output/"grid_scores.csv",rows)
    (output/"top4.json").write_text(json.dumps(selected,indent=2)+"\n")
    files = [HERE/"rules.json",inputs/"frame_cosines.csv",votes_path,output/"grid_scores.csv",output/"top4.json"]
    log={"audio_vote_rows":len(votes),"windows_with_audio_match":sum(r["audio_rank"] is not None for r in rows),
         "fusion_and_ranking_seconds":round(elapsed,6),"sha256":{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in files},
         "formal_recall_at_4":None,"reference_status":"未跑"}
    (output/"rank_check.json").write_text(json.dumps(log,ensure_ascii=False,indent=2)+"\n")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--input",type=Path,default=HERE)
    p.add_argument("--output",type=Path,default=HERE);p.add_argument("--detector-tsv",type=Path)
    a=p.parse_args();rank(a.input,a.output,a.detector_tsv)
