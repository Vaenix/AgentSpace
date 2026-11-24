"""Fetch only candidate review windows from the official media. Keep their audio."""
import concurrent.futures
import json
import subprocess
from pathlib import Path
from clip_bounds import CLEAN_WINDOWS, review_bounds
from recover_review_media import DEFAULT_CACHE, process

root = Path(__file__).resolve().parent
media = root / 'review_media'
media.mkdir(exist_ok=True)
def fetch(job):
    c, event = job
    start, stop = review_bounds(c, event)
    out = media / f'{c["pair_id"]}_{event["ordinal"]}.mp4'
    temp = out.with_suffix('.partial.mp4')
    try:
        if not out.exists() and (c["pair_id"],event["ordinal"]) in CLEAN_WINDOWS:
            process(dict(c, occurrences=[event]), DEFAULT_CACHE)
        if not out.exists():
            command=['ffmpeg','-nostdin','-y','-hide_banner','-loglevel','error','-rw_timeout','12000000',
                     '-ss',f'{start:.3f}','-i',c['media_url'],'-t',f'{stop-start:.3f}',
                     '-map','0:v:0','-map','0:a:0','-vf','fps=5,scale=480:-2','-c:v','libx264','-preset','veryfast',
                     '-ac','1','-ar','24000','-c:a','aac','-movflags','+faststart',str(temp)]
            subprocess.run(command,check=True,timeout=100,stderr=subprocess.PIPE)
        target=out if out.exists() else temp
        info=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_entries','stream=codec_type,duration','-of','json',str(target)]))
        lengths={s['codec_type']:float(s['duration']) for s in info['streams']}
        if any(abs(lengths.get(k,0)-(stop-start))>0.6 for k in ('video','audio')):
            raise ValueError(f'Incomplete streams {lengths}')
        if target==temp: temp.replace(out)
        return dict(pair_id=c['pair_id'],ordinal=event['ordinal'],path=str(out.relative_to(root)),
                    start_s=round(start,3),end_s=round(stop,3),streams=lengths)
    except Exception as exc:
        temp.unlink(missing_ok=True)
        reason=exc.stderr.decode(errors='replace')[-380:] if isinstance(exc,subprocess.CalledProcessError) and exc.stderr else type(exc).__name__
        return dict(pair_id=c['pair_id'],ordinal=event['ordinal'],error=reason)
if __name__=='__main__':
    pairs=[json.loads(l) for l in (root/'review_queue.jsonl').read_text().splitlines()]
    jobs=[(c,e) for c in pairs for e in c['occurrences']]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        with (root/'media_manifest.jsonl').open('w') as manifest:
            for row in pool.map(fetch,jobs):
                manifest.write(json.dumps(row,ensure_ascii=False)+'\n');manifest.flush()
                print(row['pair_id'],row['ordinal'],row.get('error','ok'),flush=True)
