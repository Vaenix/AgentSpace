"""Select proposed pairs from EPIC validation annotations, never verified ground truth."""
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOUNDS = {"beep":"提示音","water":"流水声","rustle":"窸窣声","cut / chop":"切剁声",
          "sizzling / boiling":"滋滋声或沸腾声","pour":"倾倒声","footstep":"脚步声"}
def rows(name):
    with (ROOT / "annotations" / (name + ".csv")).open() as file:
        return list(csv.DictReader(file))
def seconds(ts):
    h,m,s=ts.split(":")
    return int(h)*3600 + int(m)*60 + float(s)
def interval(row):
    return [seconds(row[k]) for k in ("start_timestamp","stop_timestamp")]

def main():
    sounds, actions = rows("sounds"), rows("actions")
    duration={r["video_id"]:float(r["duration"]) for r in rows("videos")}
    splits={r["video_id"]:r["split"] for r in rows("epic_55_splits")}
    errata={r["rdsf_path"]:r["dropbox_path"] for r in rows("errata")}
    per_video=defaultdict(list)
    for a in actions:
        per_video[a["video_id"]].append(a)
    groups=defaultdict(list)
    for s in sounds:
        if s["class"] in SOUNDS and duration.get(s["video_id"],0)>=600:
            groups[s["video_id"],s["class"]].append(s)
    candidates=[]
    for (vid,kind), group in sorted(groups.items()):
        if len(group)<2: continue
        anchors=sorted(group,key=lambda s:interval(s)[0])[:2]
        times=[interval(s) for s in anchors]
        if times[1][0]-times[0][1]<10 or any(end-start<0.5 or end-start>25 for start,end in times): continue
        matched=[]
        for start,end in times:
            following=sorted((a for a in per_video[vid] if end<=interval(a)[0]<=end+4),
                             key=lambda a:interval(a)[0])
            if not following: break
            if len(following)>1 and interval(following[1])[0]-interval(following[0])[0]<0.6: break
            matched.append(following[0])
        if len(matched)!=2: continue
        a,b=matched
        # Drop incomplete narrations and continued activities without a new onset.
        if any(x["narration"].lower().split()[-1] in {"the","a","an"} or
               x["narration"].lower().startswith("continue ") for x in matched): continue
        if a['verb_class']==b['verb_class'] or a['noun_class']==b['noun_class']: continue
        person,seq=vid.split('_')
        if len(seq)==3:
            base='https://data.bris.ac.uk/datasets/2g1n6qdydwa9u22shpxqzp0t8m/'
            path=f'{person}/videos/{vid}.MP4'
        else:
            base='https://data.bris.ac.uk/datasets/3h91syskeag572hl6tvuovwv4d/'
            path=f'videos/{splits[vid]}/{person}/{vid}.MP4'
        cand={"pair_id":f'{vid}-{anchors[0]["class_id"]}',"video_id":vid,"participant_id":person,
              "duration_s":duration[vid],"sound_class":kind,"media_url":errata.get(path,base+path),
              "review_status":"annotation_candidate","occurrences":[]}
        for i,(s,a) in enumerate(zip(anchors,matched),1):
            cand["occurrences"].append({"ordinal":i,"sound_id":s["annotation_id"],"sound_s":interval(s),
                "action_id":a["narration_id"],"action_s":interval(a),"action_hint":a["narration"],
                "suggested_question":f'在这段视频里，第{["一","二"][i-1]}次{SOUNDS[kind]}结束后，拍摄者接下来开始做的第一个可见动作是什么？'})
        candidates.append(cand)
    # Choose diversity without observing answers or model predictions.
    selected=[]; used_vid=set(); class_counts=Counter(); people=Counter()
    while len(selected)<10:
        remaining=[c for c in candidates if c["video_id"] not in used_vid]
        if not remaining: break
        c=min(remaining,key=lambda c:(people[c["participant_id"]],class_counts[c["sound_class"]],
                                      hashlib.sha256(c['pair_id'].encode()).hexdigest()))
        selected.append(c);used_vid.add(c["video_id"]);class_counts[c["sound_class"]]+=1;people[c["participant_id"]]+=1
    for name,entries in [('all_candidates',candidates),('review_queue',selected)]:
        (ROOT/(name+'.jsonl')).write_text(''.join(json.dumps(c,ensure_ascii=False)+'\n' for c in entries))
    print(f'{len(sounds)} sound rows; {len(actions)} action rows; {len(candidates)} candidate pairs; '
          f'{len(set(c["video_id"] for c in candidates))} candidate videos; selected {len(selected)}')
    for c in selected:
        print(c['video_id'],c['sound_class'],[(x['sound_s'],x['action_hint']) for x in c['occurrences']])
if __name__=='__main__':
    main()
