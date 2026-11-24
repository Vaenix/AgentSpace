"""Describe usable local review clips independently of any download attempt."""
import hashlib
import json
import subprocess
from pathlib import Path
from clip_bounds import review_bounds

root=Path(__file__).resolve().parent
rows=[]
for line in (root/'review_queue.jsonl').read_text().splitlines():
    pair=json.loads(line)
    for e in pair['occurrences']:
        video=root/'review_media'/f'{pair["pair_id"]}_{e["ordinal"]}.mp4'
        if not video.is_file(): continue
        start,end=review_bounds(pair,e)
        info=subprocess.check_output(['ffprobe','-v','error','-show_entries','stream=codec_type,duration','-of','json',str(video)])
        streams={s['codec_type']:float(s['duration']) for s in json.loads(info)['streams']}
        if any(abs(streams.get(k,0)-(end-start))>0.6 for k in ['video','audio']):
            raise ValueError(f'{video.name}: expected {end-start}s, got {streams}')
        rows.append(dict(pair_id=pair['pair_id'],ordinal=e['ordinal'],path=str(video.relative_to(root)),
                         start_s=round(start,3),end_s=round(end,3),streams=streams,
                         sha256=hashlib.sha256(video.read_bytes()).hexdigest()))
out=root/'available_media.jsonl'
out.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
print(f'{len(rows)} validated clips')
