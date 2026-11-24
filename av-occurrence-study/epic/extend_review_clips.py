"""Preserve the original pilot clips and extend clips ending before the annotated action."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys

from recover_review_media import DEFAULT_CACHE, ROOT, process


def main():
    cache = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CACHE
    out = ROOT / "review_extended_1124"
    out.mkdir(exist_ok=True)
    originals = {(r["pair_id"], r["ordinal"]): r for r in
                 map(json.loads, (ROOT / "available_media.jsonl").read_text().splitlines())}
    jobs, manifest = [], []
    for pair in map(json.loads, (ROOT / "review_queue.jsonl").read_text().splitlines()):
        windows = []
        for event in pair["occurrences"]:
            old = originals[pair["pair_id"], event["ordinal"]]
            action_end = event["action_s"][1]
            if old["end_s"] >= action_end:
                continue
            target = out / Path(old["path"]).name
            end = min(pair["duration_s"], action_end + 2)
            windows.append((target, old["start_s"], end, None, action_end))
            manifest.append(dict(pair_id=pair["pair_id"], ordinal=event["ordinal"],
                                 path=str(target.relative_to(ROOT)), start_s=old["start_s"], end_s=end,
                                 original_end_s=old["end_s"], annotated_action_end_s=action_end))
        if windows:
            jobs.append((pair, cache, windows))
    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(lambda args: process(*args), jobs))
    if not all(outcomes):
        raise SystemExit("Some review extensions could not be retrieved.")
    for row in manifest:
        row["sha256"] = hashlib.sha256((ROOT / row["path"]).read_bytes()).hexdigest()
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
