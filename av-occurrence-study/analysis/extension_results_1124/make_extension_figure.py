"""Rebuild the per-recording extension-batch figure from the archived text results.

Offline and read-only: reads run/participant_extension/results_1124/statistics.json and
replies.jsonl plus the frozen extension queue. No media, model or network is needed.

    python3 av-occurrence-study/analysis/extension_results_1124/make_extension_figure.py          # regenerate
    python3 av-occurrence-study/analysis/extension_results_1124/make_extension_figure.py --check  # verify committed outputs
"""
import argparse
from collections import Counter
import csv
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile

HERE = Path(__file__).resolve().parent
EXTENSION = HERE.parent.parent / "run" / "participant_extension"
RESULTS = EXTENSION / "results_1124"
QUEUE = EXTENSION / "manifest" / "queue.jsonl"
RULE = "matched_seconds"
INTERVENTIONS = ("options_reversed", "segment_labels_reversed")
ORDINALS = (1, 2)
ABSTENTIONS = ("both_abstained", "answer_abstained", "event_abstained")
TARGET_CELL = "action_stable_window_changed"
CELLS = ("action_stable_window_stable", TARGET_CELL, "action_changed_window_stable", "action_changed_window_changed")
CATEGORIES = ("technical_missing", "not_comparable_abstention", "comparable_no_event", "comparable_target_event")
FIGURE = "fig_extension_per_recording"
DATA_OUTPUTS = ("figure_data.csv", "figure_totals.csv", "caption.md", "verification.json")
FIGURE_OUTPUTS = (f"{FIGURE}.pdf", f"{FIGURE}.svg")
INTERVENTION_LABEL = {
    "options_reversed": "Reversed action choices",
    "segment_labels_reversed": "Reversed evidence codes",
}
INTERVENTION_ROLE = {"options_reversed": "primary contrast", "segment_labels_reversed": "control"}
STATE_LABEL = {"complete": "complete", "both_abstained": "both null", "answer_abstained": "answer null",
               "event_abstained": "evidence null"}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class Checks:
    def __init__(self):
        self.passed = []

    def __call__(self, name, ok):
        if not ok:
            raise SystemExit(f"Verification failed: {name}")
        self.passed.append({"check": name, "passed": True})


# ---------------------------------------------------------------- data

def load(check):
    stats = json.loads((RESULTS / "statistics.json").read_text(encoding="utf-8"))
    queue = jsonl(QUEUE)
    check("replies.jsonl SHA-256 equals statistics.json replies_sha256",
          sha256(RESULTS / "replies.jsonl") == stats["replies_sha256"])
    check("attempts.jsonl SHA-256 equals statistics.json attempts_sha256",
          sha256(RESULTS / "attempts.jsonl") == stats["attempts_sha256"])
    check("statistics.json names matched_seconds as the primary rule", stats["primary_rule"] == RULE)
    fingerprint = json.loads((EXTENSION / "manifest" / "fingerprint.json").read_text(encoding="utf-8"))
    check("queue.jsonl SHA-256 equals the frozen manifest/fingerprint.json entry",
          sha256(QUEUE) == fingerprint["files"]["queue.jsonl"])
    order = [row["pair_id"] for row in sorted(queue, key=lambda r: r["extension_rank"])]
    check("queue extension_rank is 1..12 without gaps",
          sorted(r["extension_rank"] for r in queue) == list(range(1, 13)))
    check("statistics.json recordings follow the frozen queue order", stats["recordings"] == order)
    return stats, queue


def classify(original, intervention):
    """Figure category of one question-intervention comparison under the primary parser."""
    statuses = (original["status"], intervention["status"])
    if all(s == "complete" for s in statuses):
        moved = original["physical_window"] != intervention["physical_window"]
        same = original["answer_id"] == intervention["answer_id"]
        cell = f"action_{'stable' if same else 'changed'}_window_{'changed' if moved else 'stable'}"
        return ("comparable_target_event" if cell == TARGET_CELL else "comparable_no_event"), cell
    if "technical_missing" in statuses:
        return "technical_missing", None
    if any(s in ABSTENTIONS for s in statuses) and all(s in ABSTENTIONS + ("complete",) for s in statuses):
        return "not_comparable_abstention", None
    # Invalid fields, format errors and code/seconds conflicts do not occur under the primary
    # parser in this batch; the figure has no category for them, so refuse to draw.
    raise SystemExit(f"Unhandled non-comparable state pair {statuses}")


def build_rows(stats, queue, check):
    rule = stats["rules"][RULE]
    replies = {(r["qid"], r["condition"]): r for r in rule["replies"]}
    check("statistics.json holds 72 distinct primary-parser reply keys", len(replies) == len(rule["replies"]) == 72)
    ledger = {(r["qid"], r["condition"]): r for r in jsonl(RESULTS / "replies.jsonl")}
    check("every received primary-parser reply carries the verbatim replies.jsonl text",
          len(ledger) == 66 and all(replies[key].get("raw_response") == row["raw_response"] for key, row in ledger.items()))
    check("every key absent from replies.jsonl is technical missing in statistics.json",
          all((key in ledger) != (row["status"] == "technical_missing") for key, row in replies.items()))
    rows = []
    for item in sorted(queue, key=lambda r: r["extension_rank"]):
        pair = item["pair_id"]
        for intervention in INTERVENTIONS:
            frozen = rule["comparisons"][intervention]["recordings"][pair]["questions"]
            for ordinal in ORDINALS:
                qid = f"{pair}-smoke-{ordinal}"
                original, test = replies[qid, "original"], replies[qid, intervention]
                category, cell = classify(original, test)
                stored = next(q for q in frozen if q["ordinal"] == ordinal)
                same_windows = cell is None or (stored["original_window"], stored["intervention_window"]) == (
                    original["physical_window"], test["physical_window"])
                check(f"{qid} {intervention}: recomputed comparison equals statistics.json",
                      stored["qid"] == qid
                      and (stored["original_status"], stored["intervention_status"]) == (original["status"], test["status"])
                      and stored["comparable"] == category.startswith("comparable")
                      and stored["cell"] == cell
                      and stored["event"] == (None if cell is None else cell == TARGET_CELL)
                      and same_windows)
                rows.append(dict(
                    queue_rank=item["extension_rank"], pair_id=pair, video_id=item["video_id"],
                    sound_class=item["sound_class"], intervention=intervention, question=ordinal, qid=qid,
                    original_run_order=original["run_order"], intervention_run_order=test["run_order"],
                    original_status=original["status"], intervention_status=test["status"],
                    category=category, cell=cell or "",
                    original_action=original.get("action") or "", intervention_action=test.get("action") or "",
                    original_window=original.get("physical_window") if cell else "",
                    intervention_window=test.get("physical_window") if cell else "",
                ))
    return rows


def totals(rows, stats, check):
    rule = stats["rules"][RULE]
    out = []
    for intervention in INTERVENTIONS:
        subset = [r for r in rows if r["intervention"] == intervention]
        categories = Counter(r["category"] for r in subset)
        cells = Counter(r["cell"] for r in subset if r["cell"])
        events = [r for r in subset if r["category"] == "comparable_target_event"]
        sources = sorted({r["pair_id"] for r in events}, key=lambda p: next(r["queue_rank"] for r in subset if r["pair_id"] == p))
        stored = rule["comparisons"][intervention]["questions"]
        comparable = categories["comparable_no_event"] + categories["comparable_target_event"]
        check(f"{intervention}: 24 planned questions partition into the four figure categories",
              len(subset) == stored["planned_questions"] == 24 and sum(categories.values()) == 24)
        check(f"{intervention}: comparable count and four cells equal statistics.json",
              comparable == stored["comparable_questions"] and all(cells[c] == stored["cells"][c] for c in CELLS))
        check(f"{intervention}: target events equal statistics.json joint_events",
              len(events) == stored["joint_events"])
        pairs = Counter((r["original_status"], r["intervention_status"])
                        for r in subset if not r["category"].startswith("comparable"))
        check(f"{intervention}: non-comparable ordered state pairs equal statistics.json",
              pairs == Counter({(p["original_status"], p["intervention_status"]): p["count"]
                                for p in stored["non_comparable_status_pairs"]}))
        out.append(dict(
            rule=RULE, intervention=intervention, role=INTERVENTION_ROLE[intervention], planned_questions=len(subset),
            comparable_questions=comparable, technical_missing=categories["technical_missing"],
            not_comparable_abstention=categories["not_comparable_abstention"],
            comparable_no_event=categories["comparable_no_event"], target_events=len(events),
            **{f"cell_{c}": cells[c] for c in CELLS},
            target_event_recordings=" ".join(sources),
            target_event_questions=" ".join(f"{r['pair_id']}:Q{r['question']}" for r in events),
        ))
        check(f"{intervention}: technical-missing comparisons are exactly the two questions of P11_104-7",
              sorted(r["qid"] for r in subset if r["category"] == "technical_missing")
              == ["P11_104-7-smoke-1", "P11_104-7-smoke-2"])
    check("the two interventions' target events come from disjoint recordings",
          not set(out[0]["target_event_recordings"].split()) & set(out[1]["target_event_recordings"].split()))
    missing_calls = rule["technical_missing"]
    check("6 of 72 planned calls are technical missing and stay in the planned keys",
          missing_calls["planned_calls"] == 72 and missing_calls["missing_calls"] == 6
          and sum(r["status"] == "technical_missing" for r in rule["replies"]) == 6)
    return out


def frame_gap(stats, check):
    """Describe the technical-missing recording from the frozen frame record (frozen_inputs/provenance.json)."""
    path = EXTENSION / "frozen_inputs" / "provenance.json"
    check("frozen_inputs/provenance.json SHA-256 equals statistics.json frozen_input_sha256",
          sha256(path) == stats["frozen_input_sha256"]["frozen_inputs/provenance.json"])
    unavailable = json.loads(path.read_text(encoding="utf-8"))["unavailable"]
    missing = stats["rules"][RULE]["technical_missing"]
    check("exactly one recording lacks a complete input and it owns every technical-missing call",
          len(unavailable) == 1
          and {key["qid"].rsplit("-smoke-", 1)[0] for key in missing["keys"]} == {unavailable[0]["pair_id"]})
    windows = sorted(unavailable[0]["windows"], key=lambda w: w["window_s"][0])
    frames = [(rank, index, frame) for rank, window in enumerate(windows, 1)
              for index, frame in enumerate(window["frames"], 1)]
    failed = [(rank, index) for rank, index, frame in frames if frame["status"] != "extracted"]
    check("the technical-missing recording has two windows and exactly one frame not extracted",
          len(windows) == 2 and len(failed) == 1)
    (window, index), = failed
    ordinal = {1: "first", 2: "second", 3: "third"}[index]
    return dict(
        pair_id=unavailable[0]["pair_id"], extracted=len(frames) - 1, planned_frames=len(frames),
        window="late" if window == 2 else "early", frame=ordinal,
        missing_calls=missing["missing_calls"], planned_calls=missing["planned_calls"],
        conditions=len(missing["keys"]) // len(ORDINALS),
    )


# ---------------------------------------------------------------- text outputs

def write_csv(path, rows):
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    path.write_text(buffer.getvalue(), encoding="utf-8")


def sources_text(total):
    return " and ".join(total["target_event_recordings"].split())


def caption(total_rows, gap):
    opt, lab = total_rows
    return f"""# Caption

**Extension batch: per-recording comparison states under the prespecified primary parser (`{RULE}`).**
Each row is one recording, in the frozen extension-queue order (rank 1–12).
For each of the two questions (Q1: action after the first sound; Q2: after the second) the figure shows how the reply under each intervention compares with the reply to the original prompt.
*Reversed action choices* (primary contrast) swaps which action each option letter denotes; *reversed evidence codes* (control) swaps which physical window each evidence code denotes.
A comparison is comparable only when both replies give a valid action and a valid window.
The target event is the same semantic action with a changed physical window (W1 and W2 are the two candidate windows in source-time order).
Non-comparable cells are split into technical missing and abstention.
For {gap['pair_id']}, {gap['extracted']} of its {gap['planned_frames']} frames were extracted; only the {gap['frame']} frame of the {gap['window']} window was not, so all {gap['missing_calls']} of its calls ({len(ORDINALS)} questions × {gap['conditions']} prompt conditions) lacked a complete input and received no reply.
These calls count among the {gap['planned_calls']} planned calls, and the {opt['technical_missing']} affected questions count among the {opt['planned_questions']} planned questions of each intervention, but not among its comparable questions.
Abstention means the answer, or both the answer and the evidence, was null on at least one side; the cell lists the original | intervention reply state.
Reversed action choices give {opt['comparable_questions']}/{opt['planned_questions']} comparable questions and {opt['target_events']}/{opt['comparable_questions']} target events (from {sources_text(opt)}); reversed evidence codes give {lab['comparable_questions']}/{lab['planned_questions']} and {lab['target_events']}/{lab['comparable_questions']} (from {sources_text(lab)}), so the events of the two interventions come from different recordings.
The two questions of a recording share frames, options and candidate windows, so they are not independent.
No human labels are used; the figure describes reply consistency, not correctness.
"""


# ---------------------------------------------------------------- figure

def draw(rows, total_rows, queue, gap, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 6, "svg.fonttype": "none", "svg.hashsalt": "extension-figure",
        "pdf.fonttype": 42,
    })
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    ink, grey, rule_color = "#1a202c", "#4a5568", "#cbd5e0"
    style = {
        "technical_missing": dict(fc="#2d3748", ec="#2d3748", ls="-", text="white"),
        "not_comparable_abstention": dict(fc="#edf2f7", ec="#a0aec0", ls="--", text=grey),
        "comparable_no_event": dict(fc="white", ec="#4a5568", ls="-", text=ink),
        "comparable_target_event": dict(fc="#d95f02", ec="#d95f02", ls="-", text="white"),
    }
    rank_x, rec_x, cell_x0 = 0.08, 0.36, 1.55
    cell_w, cell_gap, group_gap = 1.18, 0.05, 0.16
    source_w = 0.95
    cell_x = [cell_x0 + g * (2 * (cell_w + cell_gap) + group_gap) + q * (cell_w + cell_gap) for g in range(2) for q in range(2)]
    source_x = cell_x[-1] + cell_w + group_gap
    width = source_x + source_w + 0.05
    top_h, header_h, row_h = 0.24, 0.52, 0.30
    footer_h, legend_h = 0.56, 0.87
    n = len(queue)
    height = top_h + header_h + n * row_h + footer_h + legend_h
    fig = plt.figure(figsize=(width, height))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.axis("off")

    ax.text(rank_x, top_h / 2, f"Primary parser: {RULE} (prespecified)", weight="bold", size=6.6, va="center", color=ink)
    ax.text(width - 0.05, top_h / 2, "12 recordings × 2 questions × 2 interventions = 48 comparisons; no correctness labels",
            ha="right", va="center", size=5.4, color=grey)

    y0 = top_h
    ax.text(rank_x, y0 + 0.38, "Rank", weight="bold", size=6, va="center")
    ax.text(rec_x, y0 + 0.38, "Recording (sound)", weight="bold", size=6, va="center")
    for g, intervention in enumerate(INTERVENTIONS):
        left, right = cell_x[2 * g], cell_x[2 * g + 1] + cell_w
        ax.text((left + right) / 2, y0 + 0.12, f"{INTERVENTION_LABEL[intervention]} vs. original",
                weight="bold", size=6.4, ha="center", va="center")
        ax.text((left + right) / 2, y0 + 0.25, f"({INTERVENTION_ROLE[intervention]})", size=5.6, ha="center",
                va="center", color=grey, style="italic")
        ax.plot([left, right], [y0 + 0.31] * 2, color=ink, lw=0.5)
        for q in ORDINALS:
            x = cell_x[2 * g + q - 1] + cell_w / 2
            ax.text(x, y0 + 0.42, f"Q{q}: after {'first' if q == 1 else 'second'} sound", size=5.6, ha="center",
                    va="center")
    ax.text(source_x + 0.04, y0 + 0.31, "Target event under", weight="bold", size=6, va="center")
    ax.plot([0.05, width - 0.05], [y0 + header_h - 0.02] * 2, color=ink, lw=0.8)

    index = {(r["pair_id"], r["intervention"], r["question"]): r for r in rows}
    event_sources = {t["intervention"]: set(t["target_event_recordings"].split()) for t in total_rows}
    y = top_h + header_h
    for item in sorted(queue, key=lambda r: r["extension_rank"]):
        pair = item["pair_id"]
        mid = y + row_h / 2
        ax.text(rank_x + 0.12, mid, str(item["extension_rank"]), ha="right", va="center", size=6)
        ax.text(rec_x, mid, pair, va="center", size=6, weight="bold", family="DejaVu Sans Mono")
        ax.text(rec_x + 0.66, mid, f"({item['sound_class']})", va="center", size=5.4, color=grey)
        for g, intervention in enumerate(INTERVENTIONS):
            for q in ORDINALS:
                r = index[pair, intervention, q]
                s = style[r["category"]]
                x = cell_x[2 * g + q - 1]
                ax.add_patch(Rectangle((x, y + 0.035), cell_w, row_h - 0.07, facecolor=s["fc"], edgecolor=s["ec"],
                                       lw=0.6, ls=s["ls"]))
                first, second = cell_text(r, gap)
                ax.text(x + cell_w / 2, mid - 0.055, first, ha="center", va="center", size=5.6, color=s["text"],
                        weight="bold" if r["category"] == "comparable_target_event" else "normal")
                ax.text(x + cell_w / 2, mid + 0.06, second, ha="center", va="center", size=5.0, color=s["text"])
        tags = [INTERVENTION_LABEL[i].replace("Reversed ", "") for i in INTERVENTIONS if pair in event_sources[i]]
        ax.text(source_x + 0.04, mid, ", ".join(tags) if tags else "–", va="center", size=5.6,
                color=style["comparable_target_event"]["fc"] if tags else rule_color,
                weight="bold" if tags else "normal")
        y += row_h
        ax.plot([0.05, width - 0.05], [y, y], color="#e2e8f0", lw=0.4, zorder=0)
    ax.plot([0.05, width - 0.05], [y, y], color=ink, lw=0.8)

    for g, total in enumerate(total_rows):
        left, right = cell_x[2 * g], cell_x[2 * g + 1] + cell_w
        center = (left + right) / 2
        ax.text(center, y + 0.13, f"Comparable: {total['comparable_questions']}/{total['planned_questions']} questions",
                ha="center", va="center", size=6, weight="bold")
        ax.text(center, y + 0.28, f"Target events: {total['target_events']}/{total['comparable_questions']} comparable "
                f"({total['target_events']}/{total['planned_questions']} planned)", ha="center", va="center", size=6,
                color=style["comparable_target_event"]["fc"], weight="bold")
        ax.text(center, y + 0.43, "from " + ", ".join(total["target_event_recordings"].split()) + " (Q1 and Q2)"
                if all(f"{p}:Q{q}" in total["target_event_questions"].split()
                       for p in total["target_event_recordings"].split() for q in ORDINALS)
                else "from " + total["target_event_questions"].replace(":", " "),
                ha="center", va="center", size=5.6, color=grey)
    ax.text(source_x + 0.04, y + 0.2, "No recording has", va="center", size=5.6, color=grey, style="italic")
    ax.text(source_x + 0.04, y + 0.32, "events under both", va="center", size=5.6, color=grey, style="italic")
    ax.text(rec_x, y + 0.13, "Totals", va="center", size=6, weight="bold")

    ly = y + footer_h + 0.12
    legend = [
        ("comparable_target_event", "Target event", "same action, window changed"),
        ("comparable_no_event", "Comparable, no target event", "window kept or action changed"),
        ("not_comparable_abstention", "Not comparable: abstention", "original | intervention state"),
        ("technical_missing", "Technical missing", "incomplete input; planned, not comparable"),
    ]
    lx = rank_x
    for category, title, note in legend:
        s = style[category]
        ax.add_patch(Rectangle((lx, ly - 0.07), 0.26, 0.15, facecolor=s["fc"], edgecolor=s["ec"], lw=0.6, ls=s["ls"]))
        ax.text(lx + 0.32, ly - 0.03, title, va="center", size=5.6, weight="bold")
        ax.text(lx + 0.32, ly + 0.08, note, va="center", size=5.0, color=grey)
        lx += 1.95 if category != "not_comparable_abstention" else 1.85
    ax.text(rank_x, ly + 0.32, "Comparable = valid action and window on both sides. W1/W2 = candidate windows in "
            "source-time order. Abstention states: complete, answer null, both null (answer and evidence).",
            va="center", size=5.0, color=grey)
    planned = total_rows[0]["planned_questions"]
    ax.text(rank_x, ly + 0.45, f"Technical missing: {gap['pair_id']} yielded {gap['extracted']} of {gap['planned_frames']} "
            f"frames ({gap['frame']} frame of the {gap['window']} window missing), so its {gap['missing_calls']} calls had no "
            f"complete input. They count among the {gap['planned_calls']} planned calls and as "
            f"{total_rows[0]['technical_missing']} of the {planned} planned", va="center", size=5.0, color=grey)
    ax.text(rank_x, ly + 0.58, "questions per intervention, not among the comparable questions. "
            "Rows follow the frozen extension-queue rank.", va="center", size=5.0, color=grey)

    fig.savefig(out_dir / f"{FIGURE}.pdf", metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(out_dir / f"{FIGURE}.svg", metadata={"Date": None})
    plt.close(fig)
    return matplotlib.__version__


def cell_text(r, gap):
    category = r["category"]
    if category == "technical_missing":
        return "technical missing", f"incomplete input: {gap['extracted']}/{gap['planned_frames']} frames"
    if category == "not_comparable_abstention":
        return "abstention", f"{STATE_LABEL[r['original_status']]} | {STATE_LABEL[r['intervention_status']]}"
    same = r["original_action"] == r["intervention_action"]
    windows = (f"W{r['original_window']} → W{r['intervention_window']}"
               if r["original_window"] != r["intervention_window"] else f"W{r['original_window']} kept")
    if category == "comparable_target_event":
        return "same action", windows
    return ("same action" if same else "action changed"), windows


# ---------------------------------------------------------------- driver

def generate(out_dir, figures=True):
    check = Checks()
    stats, queue = load(check)
    rows = build_rows(stats, queue, check)
    total_rows = totals(rows, stats, check)
    gap = frame_gap(stats, check)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "figure_data.csv", rows)
    write_csv(out_dir / "figure_totals.csv", total_rows)
    (out_dir / "caption.md").write_text(caption(total_rows, gap), encoding="utf-8")
    import matplotlib
    report = {
        "source": "av-occurrence-study/run/participant_extension/results_1124 (read-only)",
        "rule": RULE,
        "input_sha256": {
            "run/participant_extension/results_1124/statistics.json": sha256(RESULTS / "statistics.json"),
            "run/participant_extension/results_1124/replies.jsonl": sha256(RESULTS / "replies.jsonl"),
            "run/participant_extension/results_1124/attempts.jsonl": sha256(RESULTS / "attempts.jsonl"),
            "run/participant_extension/manifest/queue.jsonl": sha256(QUEUE),
        },
        "matplotlib": matplotlib.__version__,
        "checks": check.passed,
    }
    (out_dir / "verification.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if figures:
        draw(rows, total_rows, queue, gap, out_dir)
    return report


def same_output(name, committed_dir, fresh_dir):
    committed, fresh = committed_dir / name, fresh_dir / name
    if not committed.exists():
        return False
    if name == "verification.json":  # the recorded matplotlib version may legitimately differ
        strip = lambda p: {k: v for k, v in json.loads(p.read_text()).items() if k != "matplotlib"}
        return strip(committed) == strip(fresh)
    return committed.read_bytes() == fresh.read_bytes()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true",
                        help="regenerate into a temporary directory and compare with the committed outputs")
    parser.add_argument("--out", type=Path, default=HERE, help="output directory (default: this directory)")
    args = parser.parse_args()
    if not args.check:
        report = generate(args.out)
        print(f"{len(report['checks'])} checks passed; wrote {len(DATA_OUTPUTS + FIGURE_OUTPUTS)} files to {args.out}")
        return
    recorded = json.loads((HERE / "verification.json").read_text())["matplotlib"]
    with tempfile.TemporaryDirectory() as tmp:
        report = generate(Path(tmp))
        compared = DATA_OUTPUTS + (FIGURE_OUTPUTS if report["matplotlib"] == recorded else ())
        differing = [name for name in compared if not same_output(name, HERE, Path(tmp))]
    print(f"{len(report['checks'])} recomputation checks passed")
    if differing:
        sys.exit("Committed outputs differ from regeneration: " + ", ".join(differing))
    if report["matplotlib"] != recorded:
        print(f"matplotlib {report['matplotlib']} differs from the recorded {recorded}: figure files not compared")
    print(f"{len(compared)} committed outputs are byte-identical to a fresh regeneration")


if __name__ == "__main__":
    main()
