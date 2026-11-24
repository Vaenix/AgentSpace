"""Compare the frozen video530 top-4 windows with a top-4 that skips repeats of one machine-detected event.

Reads only committed files: ../visual/grid_scores.csv, ../visual/top4.json, ../visual/rules.json,
../visual/audio_votes.csv and the persisted merged detector intervals in ../../audio_leads.csv.
No media, model or network access is needed.
"""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
VISUAL = HERE.parent / "visual"
LEADS = HERE.parent.parent / "audio_leads.csv"
VIDEO_ID = "video530"
ARMS = (("vision_only", "clip_mean_cosine", "vision_rank"),
        ("vision_plus_sound", "fusion_score", "fusion_rank"))
VARIANTS = ("original", "event_dedup")


def read_csv(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def csv_text(fields, rows):
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue()


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def overlap(a_lo, a_hi, b_lo, b_hi):
    """Positive overlap in seconds; touching endpoints give zero, as in the frozen ranking."""
    return max(0.0, min(a_hi, b_hi) - max(a_lo, b_lo))


def competition_ranks(values):
    ordered = sorted({v for v in values if v is not None}, reverse=True)
    ranks, position = {}, 1
    for v in ordered:
        ranks[v] = position
        position += values.count(v)
    return [ranks.get(v) for v in values]


def merge_votes(votes):
    """Re-merge 3 s detector votes with the summarize_audio.py rule (start <= current end joins)."""
    merged = []
    for v in sorted(votes, key=lambda r: (float(r["start_s"]), float(r["end_s"]))):
        lo, hi = float(v["start_s"]), float(v["end_s"])
        if merged and lo <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])
    return [tuple(m) for m in merged]


def load_events(leads_path, video_id):
    rows = [r for r in read_csv(leads_path) if r["video_id"] == video_id and r["status"] == "机器候选声音"]
    rows.sort(key=lambda r: (float(r["start_sec"]), float(r["end_sec"])))
    return [{"event_id": f"E{i}", "start": float(r["start_sec"]), "end": float(r["end_sec"]),
             "start_s": r["start_sec"], "end_s": r["end_sec"], "peak_label": r["peak_label"],
             "peak_confidence": r["peak_confidence"]} for i, r in enumerate(rows, 1)]


def attach_events(windows, events):
    for w in windows:
        hits = [(e["event_id"], overlap(w["start"], w["end"], e["start"], e["end"])) for e in events]
        w["events"] = [(eid, ov) for eid, ov in hits if ov > 0]


def order(windows, key):
    """Fixed tie rule: higher score first, then earlier start, then window id."""
    return sorted(windows, key=lambda w: (-w[key], w["start"], w["id"]))


def select(ranked, budget, dedup):
    """Greedy pass over one ranked list.

    Both variants skip a candidate with positive temporal overlap with an already selected window.
    event_dedup additionally skips a candidate that overlaps any machine event already overlapped by a
    selected window. Windows without detection overlap are never skipped by the event rule.
    Returns the selected windows and the decision for every candidate examined before the budget filled.
    """
    chosen, covered, log = [], {}, []
    for pos, w in enumerate(ranked, 1):
        if len(chosen) == budget:
            break
        clash = [c["id"] for c in chosen if overlap(w["start"], w["end"], c["start"], c["end"]) > 0]
        same = sorted({covered[eid] for eid, _ in w["events"] if eid in covered})
        if clash:
            decision, against = "skip_time_overlap", clash
        elif dedup and same:
            decision, against = "skip_same_event", same
        else:
            decision, against = "selected", []
            chosen.append(w)
            for eid, _ in w["events"]:
                covered.setdefault(eid, w["id"])
        log.append((pos, w, decision, against))
    return chosen, log


def event_cell(w):
    return "|".join(eid for eid, _ in w["events"])


def overlap_cell(w):
    return "|".join(f"{eid}:{round(ov, 6):g}" for eid, ov in w["events"])


def summarize(arm, variant, chosen, budget, events):
    repeats = {e["event_id"]: [w["id"] for w in chosen if e["event_id"] in dict(w["events"])] for e in events}
    hit = [eid for eid, ws in repeats.items() if ws]
    return {
        "arm": arm, "variant": variant, "budget": budget, "selected": len(chosen),
        "shortfall": budget - len(chosen),
        "window_seconds": "|".join(f"{w['start_s']}-{w['end_s']}" for w in chosen),
        "events_in_video": len(events), "unique_events_hit": len(hit),
        "repeats_per_event": "|".join(f"{eid}:{len(ws)}" for eid, ws in repeats.items()),
        "extra_windows_on_hit_events": sum(len(ws) - 1 for ws in repeats.values() if ws),
        "windows_without_detection": sum(not w["events"] for w in chosen),
    }, repeats


def run_cases(budget):
    """Synthetic fixtures for paths the real video530 lists may not exercise. Not study data."""
    def win(i, lo, hi, s):
        return {"id": i, "start": lo, "end": hi, "start_s": f"{lo:g}", "end_s": f"{hi:g}", "score": s}
    ev = [{"event_id": "E1", "start": 10.0, "end": 20.0}, {"event_id": "E2", "start": 50.0, "end": 55.0}]
    cases = [
        ("adjacent_windows_same_event",
         "two touching windows both overlap E1; dedup keeps the higher one and fills with the next distinct window",
         [win("A", 8, 16, 0.9), win("B", 16, 24, 0.8), win("C", 48, 56, 0.7), win("D", 80, 88, 0.6),
          win("F", 100, 108, 0.5)], ["A", "B", "C", "D"], ["A", "C", "D", "F"]),
        ("window_without_detection",
         "windows that overlap no event are never removed by the event rule and are counted separately",
         [win("A", 0, 8, 0.9), win("B", 30, 38, 0.8), win("C", 60, 68, 0.7), win("D", 90, 98, 0.6)],
         ["A", "B", "C", "D"], ["A", "B", "C", "D"]),
        ("fewer_than_four_candidates",
         "only three candidates survive dedup; the fourth slot is reported as shortfall, not refilled",
         [win("A", 8, 16, 0.9), win("B", 16, 24, 0.8), win("C", 48, 56, 0.7), win("D", 92, 100, 0.6)],
         ["A", "B", "C", "D"], ["A", "C", "D"]),
        ("equal_scores",
         "equal scores are ordered by earlier start, so the earlier of two windows on E1 is kept",
         [win("B", 16, 24, 0.8), win("A", 8, 16, 0.8), win("C", 48, 56, 0.7), win("D", 80, 88, 0.6),
          win("F", 100, 108, 0.5)], ["A", "B", "C", "D"], ["A", "C", "D", "F"]),
        ("window_spanning_new_and_hit_event",
         "a window touching an already hit event is skipped even if it also reaches a new one",
         [win("A", 8, 16, 0.9), win("B", 18, 52, 0.8), win("C", 60, 68, 0.7), win("D", 80, 88, 0.6),
          win("F", 100, 108, 0.5)], ["A", "B", "C", "D"], ["A", "C", "D", "F"]),
    ]
    rows = []
    for name, rule, wins, want_orig, want_dedup in cases:
        attach_events(wins, ev)
        ranked = order(wins, "score")
        got = {v: [w["id"] for w in select(ranked, budget, v == "event_dedup")[0]] for v in VARIANTS}
        sel = select(ranked, budget, True)[0]
        ok = got["original"] == want_orig and got["event_dedup"] == want_dedup
        rows.append({"case": name, "rule_checked": rule,
                     "windows": "|".join(f"{w['id']}:{w['start_s']}-{w['end_s']}:{w['score']}" for w in wins),
                     "expected_original": "|".join(want_orig), "observed_original": "|".join(got["original"]),
                     "expected_event_dedup": "|".join(want_dedup), "observed_event_dedup": "|".join(got["event_dedup"]),
                     "event_dedup_shortfall": budget - len(sel),
                     "event_dedup_windows_without_detection": sum(not w["events"] for w in sel),
                     "pass": "yes" if ok else "no"})
    return rows


def main(output):
    rules = json.loads((VISUAL / "rules.json").read_text())
    budget = rules["top_k"]
    if rules["video_id"] != VIDEO_ID:
        raise SystemExit("rules.json is not for video530")
    grid = read_csv(VISUAL / "grid_scores.csv")
    frozen = json.loads((VISUAL / "top4.json").read_text())
    votes = read_csv(VISUAL / "audio_votes.csv")
    events = load_events(LEADS, VIDEO_ID)

    if merge_votes(votes) != [(e["start"], e["end"]) for e in events]:
        raise SystemExit("audio_votes.csv does not merge to the persisted audio_leads.csv intervals")

    windows = []
    for r in grid:
        audio = r["audio_overlap_score"]
        windows.append({"id": r["window_id"], "start": float(r["start_s"]), "end": float(r["end_s"]),
                        "start_s": r["start_s"], "end_s": r["end_s"],
                        "clip_mean_cosine": float(r["clip_mean_cosine"]),
                        "audio": float(audio) if audio else None,
                        "fusion_score": float(r["fusion_score"]),
                        "vision_rank": int(r["vision_rank"]), "fusion_rank": int(r["fusion_rank"])})
    # The stored ranks and fusion scores must follow from the stored scores under rules.json.
    vr = competition_ranks([w["clip_mean_cosine"] for w in windows])
    ar = competition_ranks([w["audio"] for w in windows])
    for w, v, a in zip(windows, vr, ar):
        fused = 0.5 / (60 + v) + (0 if a is None else 0.5 / (60 + a))
        if v != w["vision_rank"] or fused != w["fusion_score"]:
            raise SystemExit(f"{w['id']}: stored rank or fusion score does not recompute")
    fr = competition_ranks([w["fusion_score"] for w in windows])
    if fr != [w["fusion_rank"] for w in windows]:
        raise SystemExit("stored fusion ranks do not recompute")
    attach_events(windows, events)

    sel_rows, dec_rows, sum_rows, cov_rows, match = [], [], [], [], {}
    for arm, key, rank_key in ARMS:
        ranked = order(windows, key)
        for variant in VARIANTS:
            chosen, log = select(ranked, budget, variant == "event_dedup")
            if variant == "original":
                match[arm] = [w["id"] for w in chosen] == [x["window_id"] for x in frozen[arm]]
                if not match[arm]:
                    raise SystemExit(f"{arm}: recomputed top-4 differs from top4.json")
            for slot in range(1, budget + 1):
                if slot <= len(chosen):
                    w = chosen[slot - 1]
                    sel_rows.append({"arm": arm, "variant": variant, "slot": slot, "status": "selected",
                                     "window_id": w["id"], "start_s": w["start_s"], "end_s": w["end_s"],
                                     "score": repr(w[key]), "method_rank": w[rank_key],
                                     "event_ids": event_cell(w), "event_overlap_s": overlap_cell(w),
                                     "has_detection_overlap": "yes" if w["events"] else "no"})
                else:
                    sel_rows.append({"arm": arm, "variant": variant, "slot": slot, "status": "shortfall",
                                     "window_id": "", "start_s": "", "end_s": "", "score": "",
                                     "method_rank": "", "event_ids": "", "event_overlap_s": "",
                                     "has_detection_overlap": ""})
            for pos, w, decision, against in log:
                dec_rows.append({"arm": arm, "variant": variant, "sorted_position": pos, "window_id": w["id"],
                                 "start_s": w["start_s"], "end_s": w["end_s"], "score": repr(w[key]),
                                 "method_rank": w[rank_key], "event_ids": event_cell(w),
                                 "decision": decision, "conflicts_with": "|".join(against)})
            summary, repeats = summarize(arm, variant, chosen, budget, events)
            sum_rows.append(summary)
            for e in events:
                ws = repeats[e["event_id"]]
                cov_rows.append({"arm": arm, "variant": variant, "event_id": e["event_id"],
                                 "detector_start_s": e["start_s"], "detector_end_s": e["end_s"],
                                 "peak_label": e["peak_label"], "peak_confidence": e["peak_confidence"],
                                 "windows_hitting": len(ws), "window_ids": "|".join(ws)})

    case_rows = run_cases(budget)
    if any(r["pass"] != "yes" for r in case_rows):
        raise SystemExit("a synthetic edge case failed: " + ", ".join(r["case"] for r in case_rows if r["pass"] != "yes"))

    output.mkdir(parents=True, exist_ok=True)
    outputs = {
        "summary.csv": csv_text(list(sum_rows[0]), sum_rows),
        "selections.csv": csv_text(list(sel_rows[0]), sel_rows),
        "event_coverage.csv": csv_text(list(cov_rows[0]), cov_rows),
        "decisions.csv": csv_text(list(dec_rows[0]), dec_rows),
        "edge_cases.csv": csv_text(list(case_rows[0]), case_rows),
    }
    for name, text in outputs.items():
        (output / name).write_text(text)
    inputs = [VISUAL / "grid_scores.csv", VISUAL / "top4.json", VISUAL / "rules.json",
              VISUAL / "audio_votes.csv", LEADS]
    check = {
        "video_id": VIDEO_ID, "budget": budget, "events_in_video": len(events),
        "event_source": "audio_leads.csv merged SoundAnalysis intervals (applause|clapping >= 0.5)",
        "audio_votes_merge_to_events": True,
        "stored_ranks_and_fusion_recompute": True,
        "recomputed_original_matches_top4_json": match,
        "edge_cases_passed": f"{sum(r['pass'] == 'yes' for r in case_rows)}/{len(case_rows)}",
        "input_sha256": {str(p.relative_to(HERE.parent.parent)): sha256(p) for p in inputs},
        "output_sha256": {name: hashlib.sha256(text.encode()).hexdigest() for name, text in outputs.items()},
        "human_recall_at_4": None, "reference_status": "未跑",
    }
    (output / "check.json").write_text(json.dumps(check, ensure_ascii=False, indent=2) + "\n")
    print((output / "summary.csv").read_text(), end="")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=HERE)
    main(p.parse_args().output)
