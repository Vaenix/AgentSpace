"""Freeze the next EPIC train-source candidate order: one pair per remaining participant.

Only public training annotations and video metadata are read. No model output,
reply text or media availability enters the ranking; the prior batches are used
only to list participants that must be excluded.
"""
import argparse
import contextlib
import csv
import hashlib
import io
import json
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
STUDY = HERE.parents[1]
EPIC = STUDY / "epic"
sys.path.insert(0, str(EPIC))
import screen  # noqa: E402  (reused unchanged; its SHA-256 is checked below)

MAX_PARTICIPANTS = 12
SCREEN_SHA256 = "d3a953c5be5c410ecd875751decbf8a81ea7a172c24c7cef3e91fe5050101fc0"

# Same public files and hashes as run/replication_1124/manifest/data_inputs.json.
SOURCES = {
    "sounds": ("https://raw.githubusercontent.com/epic-kitchens/epic-sounds-annotations/main/EPIC_Sounds_train.csv",
               "225427ab288cc5c72c1680abb29f5ab8ccd3608c3f4cf409541d2c5cf720dd32"),
    "actions": ("https://raw.githubusercontent.com/epic-kitchens/epic-kitchens-100-annotations/master/EPIC_100_train.csv",
                "a3a7ef2e397bd8af12bd0496b09d61b1f2c1f3f6e1726fa86af50a27f4855e87"),
    "videos": ("https://raw.githubusercontent.com/epic-kitchens/epic-kitchens-100-annotations/master/EPIC_100_video_info.csv",
               "75fd040f6662cb407b4ca2eed4811ca94df280a1aba00b6bf8a5cd286deff382"),
    "epic_55_splits": ("https://raw.githubusercontent.com/epic-kitchens/epic-kitchens-download-scripts/master/data/epic_55_splits.csv",
                       "5acf8b924a51819b7dda91cb56b867f6fc08d367dd0e3f6c99ecdb0f8a9c45e4"),
    "errata": ("https://raw.githubusercontent.com/epic-kitchens/epic-kitchens-download-scripts/master/data/errata.csv",
               "5919d70ba14d352022afac3e08fb9076306cbeec772f78aeaa96e74a40aa0d65"),
}

# Every participant here already has a pair in a queue or batch; all their videos are excluded.
# Paths are relative to av-occurrence-study/. Hashes are the ones recorded at freeze 86c0044.
EXCLUSIONS = [
    ("validation_queue", "epic/review_queue.jsonl",
     "35cbbebe4b9cc9285a7bfb80010d4a034aeaa4e636401e64b379a9df66f5500d"),
    ("train_batch_1_replication_1124", "run/replication_1124/manifest/selected.jsonl",
     "0d55975787a41241012bf670e24823237f06d9050475ba24dcea139ec9a1aad8"),
    ("train_batch_2_confirmation_1124", "run/participant_extension/inputs/confirmation_1124_selected.jsonl",
     "5d9a9158d5ba42b9c8c6eb9bcd04a32575cd321ca6734e6cf10ad089d2e915a5"),
    ("train_shortlist_1124", "run/replication_1124/manifest/ranked_train_candidates.jsonl",
     "15c1c09674bb20d2a92e5625a544cde7225655f1b8e3509b0896f51f71e7eab1"),
    ("smoke_predictions", "run/epic_smoke_predictions.jsonl",
     "e31bae613130eebd461e58537b62e0d6c92ffb04d687b3c571db97042950b432"),
    ("smoke_reversed_options", "run/epic_smoke_reversed_options.jsonl",
     "c17ac6833b7b6e32d4b9c66f3998b1a18223c751eb3cfc790f072c1975cfac4d"),
]
# Canonical location once the confirmation freeze lands on main; must match the copy above.
CONFIRMATION_CANONICAL = STUDY / "run/confirmation_1124/manifest/selected.jsonl"

KEEP = ("pair_id", "participant_id", "video_id", "duration_s", "sound_class", "media_url", "review_status")
OUTPUTS = ("queue.jsonl", "exclusion_audit.json", "source_hashes.json")


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha_file(path):
    return sha_bytes(Path(path).read_bytes())


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def dump_jsonl(items):
    return "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in items)


def dump_json(item):
    return json.dumps(item, ensure_ascii=False, indent=2) + "\n"


def fetch_sources(cache):
    """Download each public CSV once into cache with curl and refuse any byte difference."""
    cache.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, (url, expected) in SOURCES.items():
        path = cache / url.rsplit("/", 1)[1]
        if not path.is_file():
            partial = path.with_name(path.name + ".partial")
            subprocess.run(["curl", "-fsSL", "--retry", "3", "--connect-timeout", "20", "-o", str(partial), url], check=True)
            if sha_file(partial) != expected:
                raise SystemExit(f"SHA-256 mismatch for downloaded {url}")
            partial.replace(path)
        if sha_file(path) != expected:
            raise SystemExit(f"SHA-256 mismatch for {path}; expected {expected}")
        paths[name] = path
    return paths


def run_screen(paths, drop_participants=frozenset()):
    """Run epic/screen.py unchanged on the train CSVs; return (all candidates, its top-10 queue)."""
    def rows(name):
        with paths[name].open(newline="") as stream:
            table = list(csv.DictReader(stream))
        if name in ("sounds", "actions"):
            table = [row for row in table if row["participant_id"] not in drop_participants]
        return table

    saved = screen.ROOT, screen.rows
    with tempfile.TemporaryDirectory() as directory:
        screen.ROOT, screen.rows = Path(directory), rows
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                screen.main()
            return jsonl(Path(directory) / "all_candidates.jsonl"), jsonl(Path(directory) / "review_queue.jsonl")
        finally:
            screen.ROOT, screen.rows = saved


def load_exclusions():
    if CONFIRMATION_CANONICAL.is_file() and CONFIRMATION_CANONICAL.read_bytes() != (STUDY / EXCLUSIONS[2][1]).read_bytes():
        raise SystemExit(f"{CONFIRMATION_CANONICAL} differs from the frozen copy used here")
    sources, reasons = {}, {}
    for label, relative, expected in EXCLUSIONS:
        path = STUDY / relative
        actual = sha_file(path)
        if actual != expected:
            raise SystemExit(f"SHA-256 mismatch for {relative}: {actual}")
        items = jsonl(path)
        pairs = sorted({item["pair_id"] for item in items})
        people = sorted({pair.split("_", 1)[0] for pair in pairs})
        sources[label] = dict(path=relative, sha256=actual, rows=len(items), pair_ids=pairs, participants=people)
        for person in people:
            reasons.setdefault(person, []).append(label)
    # The two train batches are exactly the six-item shortlist, ranks 1-3 then 4-6.
    batches = sources["train_batch_1_replication_1124"]["pair_ids"] + sources["train_batch_2_confirmation_1124"]["pair_ids"]
    if sorted(batches) != sources["train_shortlist_1124"]["pair_ids"] or len(set(batches)) != 6:
        raise SystemExit("The two train batches do not cover the six-item shortlist exactly")
    return sources, reasons


def check_shortlist_reproduces(paths, validation_people):
    """Prove the reused screen and CSVs still give the frozen 11-24 shortlist byte for byte."""
    _, queue = run_screen(paths, frozenset(validation_people))
    shortlist = [dict(row, candidate_rank=rank) for rank, row in enumerate(queue[:6], 1)]
    actual = sha_bytes(dump_jsonl(shortlist).encode())
    if actual != EXCLUSIONS[3][2]:
        raise SystemExit(f"Screen no longer reproduces ranked_train_candidates.jsonl ({actual})")
    return actual


def select(eligible):
    """One pair per participant: fewest picks of its sound class in this batch, then SHA-256(pair_id)."""
    class_counts, chosen, queue = Counter(), set(), []
    while len(queue) < MAX_PARTICIPANTS:
        remaining = [c for c in eligible if c["participant_id"] not in chosen]
        if not remaining:
            break
        key = lambda c: (class_counts[c["sound_class"]], sha_bytes(c["pair_id"].encode()))
        pick = min(remaining, key=key)
        row = {field: pick[field] for field in KEEP}
        row["occurrences"] = pick["occurrences"]
        row["extension_rank"] = len(queue) + 1
        row["selection_key"] = dict(class_count_before=class_counts[pick["sound_class"]],
                                    pair_id_sha256=key(pick)[1],
                                    eligible_participants_left=len({c["participant_id"] for c in remaining}),
                                    eligible_pairs_left=len(remaining))
        queue.append(row)
        chosen.add(pick["participant_id"])
        class_counts[pick["sound_class"]] += 1
    return queue


def participants(path):
    with path.open(newline="") as stream:
        return {row["participant_id"] for row in csv.DictReader(stream)}


def build(cache):
    if sha_file(EPIC / "screen.py") != SCREEN_SHA256:
        raise SystemExit("epic/screen.py changed; the reused screening rule is no longer the frozen one")
    paths = fetch_sources(cache)
    sources, reasons = load_exclusions()
    shortlist_sha = check_shortlist_reproduces(paths, sources["validation_queue"]["participants"])

    candidates, _ = run_screen(paths)
    with_candidates = sorted({c["participant_id"] for c in candidates})
    eligible = [c for c in candidates if c["participant_id"] not in reasons]
    eligible_people = sorted({c["participant_id"] for c in eligible})
    queue = select(eligible)
    picked = {row["pair_id"] for row in queue}
    picked_people = {row["participant_id"] for row in queue}

    if len(picked_people) != len(queue) or picked_people & set(reasons):
        raise SystemExit("Queue violates one pair per participant or overlaps an excluded participant")

    audit = dict(
        rule=("epic/screen.py candidates on the public train CSVs; drop every participant in any exclusion source; "
              "then repeatedly take, among participants not yet picked, the pair with the fewest picks of its sound "
              f"class in this batch (from zero), ties by SHA-256(pair_id) ascending; at most {MAX_PARTICIPANTS} participants."),
        counts=dict(
            train_sound_participants=len(participants(paths["sounds"])),
            train_action_participants=len(participants(paths["actions"])),
            candidate_pairs=len(candidates),
            candidate_videos=len({c["video_id"] for c in candidates}),
            participants_with_candidates=len(with_candidates),
            excluded_participants=len(set(with_candidates) & set(reasons)),
            excluded_pairs=len(candidates) - len(eligible),
            eligible_participants=len(eligible_people),
            eligible_pairs=len(eligible),
            eligible_videos=len({c["video_id"] for c in eligible}),
            selected_participants=len(queue),
            participant_cap=MAX_PARTICIPANTS,
        ),
        excluded_participants={person: reasons[person] for person in sorted(reasons)},
        participants=[
            dict(participant_id=person,
                 candidate_pairs=sum(c["participant_id"] == person for c in candidates),
                 decision="excluded" if person in reasons else ("selected" if person in picked_people else "eligible_not_selected"),
                 reasons=reasons.get(person, []))
            for person in with_candidates
        ],
        pairs=[
            dict(pair_id=c["pair_id"], participant_id=c["participant_id"], sound_class=c["sound_class"],
                 pair_id_sha256=sha_bytes(c["pair_id"].encode()),
                 decision=("excluded_participant" if c["participant_id"] in reasons else
                           "selected" if c["pair_id"] in picked else "participant_already_selected"))
            for c in sorted(candidates, key=lambda c: c["pair_id"])
        ],
        checks=dict(
            one_pair_per_participant=len(picked_people) == len(queue),
            overlap_with_excluded_participants=sorted(picked_people & set(reasons)),
            shortlist_reproduced_sha256=shortlist_sha,
            model_outputs_used_for_ranking=False,
            media_availability_used=False,
        ),
    )
    hashes = dict(
        annotations={name: dict(url=url, file=paths[name].name, sha256=expected)
                     for name, (url, expected) in SOURCES.items()},
        rules={"epic/screen.py": SCREEN_SHA256,
               "run/participant_extension/build_queue.py": sha_file(Path(__file__))},
        exclusion_sources={label: dict(path=item["path"], sha256=item["sha256"], rows=item["rows"])
                           for label, item in sources.items()},
    )
    return {"queue.jsonl": dump_jsonl(queue), "exclusion_audit.json": dump_json(audit),
            "source_hashes.json": dump_json(hashes)}


def fingerprints(files):
    each = {name: sha_bytes(files[name].encode()) for name in OUTPUTS}
    combined = sha_bytes("".join(f"{each[name]}  {name}\n" for name in OUTPUTS).encode())
    return dict(files=each, combined=combined, order=[json.loads(line)["pair_id"] for line in files["queue.jsonl"].splitlines()])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=EPIC / "annotations" / "official_train",
                        help="directory for the public CSVs (downloaded if missing, always hash-checked)")
    parser.add_argument("--output", type=Path, required=True, help="new directory to write the manifest into")
    parser.add_argument("--check", action="store_true",
                        help="also compare every output byte for byte with the frozen manifest in this directory")
    args = parser.parse_args()
    files = build(args.cache)
    prints = fingerprints(files)
    args.output.mkdir(parents=True, exist_ok=False)
    for name in OUTPUTS:
        (args.output / name).write_text(files[name], encoding="utf-8")
    (args.output / "fingerprint.json").write_text(dump_json(prints), encoding="utf-8")
    print(json.dumps(prints, indent=2))
    if args.check:
        frozen = HERE / "manifest"
        differ = [name for name in (*OUTPUTS, "fingerprint.json")
                  if (frozen / name).read_bytes() != (args.output / name).read_bytes()]
        if differ:
            raise SystemExit(f"Rebuild differs from frozen manifest: {differ}")
        print(f"identical to {frozen.relative_to(STUDY.parent)}")


if __name__ == "__main__":
    main()
