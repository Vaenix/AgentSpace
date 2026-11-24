"""Rebuild the per-question pilot figure, count table and captions from frozen replication_1124 artifacts.

Offline and read-only: the frozen raw replies, prompt mappings and summaries under
run/replication_1124 are only read. Both parsing rules are re-implemented here, so the
outputs do not depend on any later change to the inference parser.

    python3 av-occurrence-study/analysis/pilot_figures/make_pilot_figures.py          # regenerate
    python3 av-occurrence-study/analysis/pilot_figures/make_pilot_figures.py --check  # verify committed outputs
"""
import argparse
from collections import defaultdict
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import sys
import tempfile

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent.parent / "run" / "replication_1124"
CONDITIONS = ("original", "options_reversed", "segment_labels_reversed")
INTERVENTIONS = CONDITIONS[1:]
VIDEOS = ("P37_102", "P20_01", "P03_117")
ANALYSES = ("strict_preregistered", "posthoc_format_sensitivity")
COUNT_KEYS = ("all_questions", "comparable_questions", "action_stable",
              "physical_window_changed", "action_stable_and_window_changed")
FENCED = re.compile(r"```(?:json)?\s*([\s\S]+?)\s*```", re.I)
CODE_WITH_SECONDS = re.compile(r"(E0[12])\s+(\d+(?:\.\d+)?)\s*[–-]\s*(\d+(?:\.\d+)?)\s*s", re.I)
FIGURE = "fig_pilot_per_question"
OUTPUTS = (f"{FIGURE}.pdf", f"{FIGURE}.svg", "pilot_responses.csv", "pilot_comparisons.csv",
           "table_pilot_counts.csv", "table_pilot_counts.md", "table_pilot_counts.tex",
           "captions.md", "verification.json")
LABEL = {
    "original": "Original",
    "options_reversed": "Options reversed",
    "segment_labels_reversed": "Segment labels reversed",
    "strict_preregistered": "Pre-registered strict parsing",
    "posthoc_format_sensitivity": "Post-hoc format sensitivity",
}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def seconds(value):
    return f"{value:.3f}"


def interval_text(interval):
    return f"{seconds(interval[0])}–{seconds(interval[1])} s"


# ---------------------------------------------------------------- parsing rules

def strict_parse(raw, option_codes, segment_codes):
    """Pre-registered rule: answer must be an option letter and event a bare segment code (or null)."""
    text = raw.strip()
    fenced = FENCED.fullmatch(text)
    try:
        reply = json.loads(fenced[1] if fenced else text)
    except json.JSONDecodeError:
        return None, None, None, "invalid_json"
    if not isinstance(reply, dict) or set(reply) != {"answer", "event"}:
        return reply, None, None, "invalid_keys"
    answer, event = reply["answer"], reply["event"]
    answer_ok = answer is None or isinstance(answer, str) and answer in option_codes
    event_ok = event is None or isinstance(event, str) and event in segment_codes
    answer_id = option_codes.get(answer) if isinstance(answer, str) else None
    interval = segment_codes.get(event) if isinstance(event, str) else None
    return reply, answer_id, interval, "ok" if answer_ok and event_ok else "invalid_value"


def posthoc_recover(reply, segment_codes):
    """Post-hoc rule: accept "E0x a–bs" only if the code and both printed bounds match the frozen mapping."""
    event = reply.get("event") if isinstance(reply, dict) else None
    match = CODE_WITH_SECONDS.fullmatch(event.strip()) if isinstance(event, str) else None
    if not match:
        return None
    code, left, right = match.groups()
    mapping = segment_codes.get(code.upper())
    if mapping and (float(left), float(right)) == (float(f"{mapping[0]:g}"), float(f"{mapping[1]:g}")):
        return mapping
    return None


# ---------------------------------------------------------------- recomputation

def load_inputs():
    predictions = jsonl(SOURCE / "predictions.jsonl")
    prompts = jsonl(SOURCE / "manifest" / "prompts.jsonl")
    frozen_hashes = json.loads((SOURCE / "manifest" / "input_hashes.json").read_text())
    if sha256(SOURCE / "manifest" / "prompts.jsonl") != frozen_hashes["prompts_sha256"]:
        raise SystemExit("prompts.jsonl differs from its frozen SHA-256")
    keys = [(row["qid"], row["condition"]) for row in predictions]
    prompt_index = {(row["qid"], row["condition"]): row for row in prompts}
    if len(keys) != 18 or len(set(keys)) != 18 or set(keys) != set(prompt_index):
        raise SystemExit("Expected exactly 18 distinct question-condition replies matching the frozen prompts")
    return predictions, prompt_index


def build_responses(predictions, prompt_index):
    responses = []
    for row in predictions:
        item = prompt_index[row["qid"], row["condition"]]
        reply, answer_id, interval, fmt = strict_parse(row["raw_response"], item["option_codes"], item["segment_codes"])
        order = item["physical_window_order"]
        window = order.index(interval) + 1 if interval is not None else None
        frozen = (row["answer_id"], row["interval_s"], row["physical_window"], row["result_format"])
        if (answer_id, interval, window, fmt) != frozen:
            raise SystemExit(f"Strict re-parse disagrees with frozen fields for {row['qid']} {row['condition']}")
        raw_answer = reply.get("answer") if isinstance(reply, dict) else None
        raw_event = reply.get("event") if isinstance(reply, dict) else None
        if fmt != "ok":
            strict_status = "format_not_comparable"
        elif raw_answer is None and raw_event is None:
            strict_status = "abstained"
        elif raw_answer is None or raw_event is None:
            strict_status = "partially_abstained"
        else:
            strict_status = "valid"
        recovered = posthoc_recover(reply, item["segment_codes"]) if fmt == "invalid_value" and answer_id and interval is None else None
        posthoc_interval = interval if recovered is None else recovered
        posthoc_window = order.index(posthoc_interval) + 1 if posthoc_interval is not None else None
        posthoc_status = "recovered" if recovered is not None else strict_status
        actions = item["semantic_actions"]
        action_ids = list(actions)  # alpha = first listed action, beta = second (fixed across conditions)
        responses.append(dict(
            video_id=row["video_id"], qid=row["qid"], ordinal=row["ordinal"], condition=row["condition"],
            raw_answer=raw_answer, raw_event=raw_event, raw_response=row["raw_response"],
            option_map=" ".join(f"{k}={'αβ'[action_ids.index(v)]}" for k, v in item["option_codes"].items()),
            segment_map=" ".join(f"{k}=W{order.index(v) + 1}" for k, v in sorted(item["segment_codes"].items())),
            action_id=answer_id, action_symbol="αβ"[action_ids.index(answer_id)] if answer_id else None,
            action=actions[answer_id].rstrip(".") if answer_id else None,
            strict_format=fmt, strict_status=strict_status, strict_window=window,
            strict_interval_s=interval, posthoc_status=posthoc_status, posthoc_window=posthoc_window,
            posthoc_interval_s=posthoc_interval,
        ))
    return responses


def compare(responses):
    grouped = defaultdict(dict)
    for row in responses:
        grouped[row["qid"]][row["condition"]] = row
    comparisons = []
    for qid in sorted(grouped, key=lambda q: (VIDEOS.index(grouped[q]["original"]["video_id"]), q)):
        group = grouped[qid]
        ref = group["original"]
        for analysis in ANALYSES:
            prefix = "strict" if analysis == ANALYSES[0] else "posthoc"
            for intervention in INTERVENTIONS:
                test = group[intervention]
                sides = (ref, test)
                if analysis == ANALYSES[0]:
                    comparable = all(r["strict_status"] == "valid" for r in sides)
                else:
                    comparable = all(r["action_id"] is not None and r["posthoc_window"] is not None for r in sides)
                if comparable:
                    reason = ""
                elif any(r[f"{prefix}_status"] in ("abstained", "partially_abstained") for r in sides):
                    reason = "abstention"
                else:
                    reason = "format"
                same = ref["action_id"] == test["action_id"] if comparable else None
                moved = ref[f"{prefix}_window"] != test[f"{prefix}_window"] if comparable else None
                comparisons.append(dict(
                    analysis=analysis, video_id=ref["video_id"], qid=qid, ordinal=ref["ordinal"],
                    intervention=intervention, comparable=comparable, not_comparable_reason=reason,
                    original_action=ref["action"], test_action=test["action"],
                    original_window=ref[f"{prefix}_window"] if comparable else None,
                    test_window=test[f"{prefix}_window"] if comparable else None,
                    original_interval_s=ref[f"{prefix}_interval_s"] if comparable else None,
                    test_interval_s=test[f"{prefix}_interval_s"] if comparable else None,
                    action_stable=same, physical_window_changed=moved,
                    action_stable_and_window_changed=(same and moved) if comparable else None,
                ))
    return comparisons


def count(comparisons):
    counts = {}
    for analysis in ANALYSES:
        for intervention in INTERVENTIONS:
            for scope in VIDEOS + ("all",):
                rows = [c for c in comparisons if c["analysis"] == analysis and c["intervention"] == intervention
                        and scope in ("all", c["video_id"])]
                counts[analysis, intervention, scope] = dict(
                    all_questions=len(rows),
                    comparable_questions=sum(c["comparable"] for c in rows),
                    not_comparable_abstention=sum(c["not_comparable_reason"] == "abstention" for c in rows),
                    not_comparable_format=sum(c["not_comparable_reason"] == "format" for c in rows),
                    action_stable=sum(bool(c["action_stable"]) for c in rows),
                    physical_window_changed=sum(bool(c["physical_window_changed"]) for c in rows),
                    action_stable_and_window_changed=sum(bool(c["action_stable_and_window_changed"]) for c in rows),
                )
    return counts


def verify(responses, comparisons, counts):
    """Cross-check every recomputed number against the frozen summaries; abort on any disagreement."""
    checks = []

    def check(name, ok):
        checks.append({"check": name, "passed": bool(ok)})
        if not ok:
            raise SystemExit(f"Verification failed: {name}")

    def subset(scope_counts):
        return {k: scope_counts[k] for k in COUNT_KEYS}

    check("18 distinct question-condition replies match the frozen prompt manifest", len(responses) == 18)
    check("strict re-parse reproduces answer_id, interval_s, physical_window and result_format of all 18 replies", True)
    summary = json.loads((SOURCE / "summary.json").read_text())
    for intervention in INTERVENTIONS:
        check(f"strict totals equal summary.json ({intervention})",
              subset(counts[ANALYSES[0], intervention, "all"]) == summary[intervention])
    with (SOURCE / "comparisons.csv").open(newline="") as handle:
        frozen_rows = {(r["qid"], r["intervention"]): r for r in csv.DictReader(handle)}

    def as_csv(value):
        return "" if value is None else str(value)

    strict_rows = [c for c in comparisons if c["analysis"] == ANALYSES[0]]
    check("strict per-question comparisons equal comparisons.csv", len(strict_rows) == len(frozen_rows) == 12 and all(
        [as_csv(c[k]) for k in ("comparable", "action_stable", "physical_window_changed", "action_stable_and_window_changed")]
        == [frozen_rows[c["qid"], c["intervention"]][k] for k in ("comparable", "action_stable", "physical_window_changed",
                                                                   "action_stable_and_window_changed")]
        for c in strict_rows))
    sensitivity = json.loads((SOURCE / "format_sensitivity.json").read_text())
    recovered = sorted((r["qid"], r["condition"], r["raw_event"], tuple(r["posthoc_interval_s"]))
                       for r in responses if r["posthoc_status"] == "recovered")
    frozen_recovered = sorted((r["qid"], r["condition"], r["raw_event"], tuple(r["interval_s"]))
                              for r in sensitivity["recovered_outputs"])
    check("post-hoc recovered replies equal format_sensitivity.json recovered_outputs", recovered == frozen_recovered)
    for analysis, total_key, video_key in ((ANALYSES[0], "strict_metrics", "strict_metrics_by_video"),
                                           (ANALYSES[1], "metrics", "metrics_by_video")):
        for intervention in INTERVENTIONS:
            check(f"{analysis} totals equal format_sensitivity.json {total_key} ({intervention})",
                  subset(counts[analysis, intervention, "all"]) == sensitivity[total_key][intervention])
            for video in VIDEOS:
                check(f"{analysis} {video} equals format_sensitivity.json {video_key} ({intervention})",
                      subset(counts[analysis, intervention, video]) == sensitivity[video_key][video][intervention])
    for key, value in counts.items():
        check(f"denominator partition {'/'.join(key)}", value["all_questions"] == value["comparable_questions"]
              + value["not_comparable_abstention"] + value["not_comparable_format"])
    return checks


# ---------------------------------------------------------------- tables and captions

def ratio(c):
    if c["comparable_questions"] == 0:
        return "no comparable questions (0 of %d)" % c["all_questions"]
    return f"{c['action_stable_and_window_changed']} / {c['comparable_questions']}"


def write_csv(path, rows, fields):
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, list) else ("" if v is None else v)
                         for k, v in ((k, row[k]) for k in fields)})
    path.write_text(buffer.getvalue(), encoding="utf-8")


def table_rows(counts):
    rows = []
    for analysis in ANALYSES:
        for intervention in INTERVENTIONS:
            for scope in VIDEOS + ("all",):
                rows.append(dict(analysis=analysis, intervention=intervention, scope=scope,
                                 **counts[analysis, intervention, scope]))
    return rows


def example_rows(responses):
    return [r for r in responses if r["video_id"] == "P03_117" and r["ordinal"] == 1]


def markdown_table(counts, responses, meta):
    lines = ["# Table: behavioural consistency counts, replication_1124 (3 videos, 6 frozen questions)", "",
             "Counts are numbers of questions. A question is comparable only if both the original reply and the "
             "intervention reply yield an option and a physical time window under the stated rule. "
             "No gold answers are used, so no count is a measure of correctness.", ""]
    for analysis in ANALYSES:
        lines += [f"## {LABEL[analysis]}" + (" (primary)" if analysis == ANALYSES[0] else " (supplementary)"), "",
                  "| Intervention | Scope | All questions | Comparable | Not comparable: abstention | Not comparable: format | "
                  "Action stable | Window changed | Action stable and window changed / comparable |",
                  "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
        for intervention in INTERVENTIONS:
            for scope in VIDEOS + ("all",):
                c = counts[analysis, intervention, scope]
                lines.append(f"| {LABEL[intervention]} | {'All videos' if scope == 'all' else scope} | {c['all_questions']} | "
                             f"{c['comparable_questions']} | {c['not_comparable_abstention']} | {c['not_comparable_format']} | "
                             f"{c['action_stable']} | {c['physical_window_changed']} | {ratio(c)} |")
        lines.append("")
    lines += ["## Example: P03_117, question 1 (two cited physical intervals)", "",
              "| Condition | Option map | Segment-code map | Raw answer | Raw event | Action | Strict window | Post-hoc window |",
              "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in example_rows(responses):
        lines.append(f"| {LABEL[r['condition']]} | {r['option_map']} | {r['segment_map']} | `{r['raw_answer']}` | "
                     f"`{r['raw_event']}` | {r['action']} | {window_text(r, 'strict')} | {window_text(r, 'posthoc')} |")
    lines += ["", f"α = {meta['P03_117']['actions'][0]}; β = {meta['P03_117']['actions'][1]}. "
              f"W1 = {interval_text(meta['P03_117']['windows'][0])}, W2 = {interval_text(meta['P03_117']['windows'][1])} "
              "(source-video time). Question 2 of P03_117 has identical replies.", ""]
    return "\n".join(lines)


def window_text(r, prefix):
    window = r[f"{prefix}_window"]
    if window is None:
        return "abstained" if r[f"{prefix}_status"] in ("abstained", "partially_abstained") else "not decodable"
    return f"W{window} ({interval_text(r[f'{prefix}_interval_s'])})"


def latex_table(counts, responses, meta):
    def esc(text):
        return str(text).replace("_", r"\_").replace("–", r"\textendash{}").replace("α", r"$\alpha$").replace("β", r"$\beta$")

    out = [r"% Generated by make_pilot_figures.py; do not edit by hand. Requires \usepackage{booktabs}.",
           r"\begin{table*}[t]", r"\centering", r"\small",
           r"\caption{Behavioural consistency counts for replication\_1124 (3 videos, 6 frozen questions). "
           r"Entries are numbers of questions; the last column is action-stable-and-window-changed over comparable "
           r"questions. The pre-registered strict analysis is primary; the post-hoc format sensitivity analysis is "
           r"supplementary. The two questions of each video share one clip and are not independent.}",
           r"\label{tab:pilot-counts}",
           r"\begin{tabular}{llrrrrl}", r"\toprule",
           r"Intervention & Scope & All & Comparable & Abstention & Format & Stable \& changed / comparable \\",
           r"\midrule"]
    for analysis in ANALYSES:
        out.append(r"\multicolumn{7}{l}{\textit{" + LABEL[analysis] + (" (primary)" if analysis == ANALYSES[0] else
                                                                      " (supplementary)") + r"}} \\")
        for intervention in INTERVENTIONS:
            for scope in VIDEOS + ("all",):
                c = counts[analysis, intervention, scope]
                cell = ratio(c) if c["comparable_questions"] else r"-- (0 comparable)"
                out.append(f"{LABEL[intervention] if scope == VIDEOS[0] else ''} & {esc('All videos' if scope == 'all' else scope)} & "
                           f"{c['all_questions']} & {c['comparable_questions']} & {c['not_comparable_abstention']} & "
                           f"{c['not_comparable_format']} & {cell} \\\\")
        out.append(r"\midrule" if analysis == ANALYSES[0] else r"\bottomrule")
    out += [r"\end{tabular}", "", r"\vspace{4pt}", r"\begin{tabular}{lllll}", r"\toprule",
            r"P03\_117 Q1 condition & Raw answer & Raw event & Action & Cited window (strict / post hoc) \\", r"\midrule"]
    for r in example_rows(responses):
        out.append(f"{LABEL[r['condition']]} & \\texttt{{{esc(r['raw_answer'])}}} & \\texttt{{{esc(r['raw_event'])}}} & "
                   f"{esc(r['action'])} & {esc(window_text(r, 'strict'))} / {esc(window_text(r, 'posthoc'))} \\\\")
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    return "\n".join(out)


def captions(counts, meta, responses):
    s_opt, s_lab = (counts[ANALYSES[0], i, "all"] for i in INTERVENTIONS)
    p_opt, p_lab = (counts[ANALYSES[1], i, "all"] for i in INTERVENTIONS)
    p03 = meta["P03_117"]["windows"]
    by_key = {(r["video_id"], r["ordinal"], r["condition"]): r for r in responses}
    # The prose below states these observations; fail instead of printing a stale caption.
    if sum(r["video_id"] == "P20_01" and r["strict_status"] == "abstained" for r in responses) != 6:
        raise SystemExit("Caption assumes P20_01 abstained in all six replies")
    for q in (1, 2):
        o, x, s_ = (by_key["P03_117", q, c] for c in CONDITIONS)
        if not (o["action_id"] == x["action_id"] == s_["action_id"] and o["posthoc_window"] == s_["posthoc_window"] == 1
                and x["posthoc_window"] == 2 and o["raw_event"].startswith("E01") and s_["raw_event"].startswith("E02")):
            raise SystemExit("Caption's P03_117 example no longer matches the replies")

    def strict_line(name, c):
        if c["comparable_questions"] == 0:
            return f"{name}: no comparable question (0 of {c['all_questions']}; {c['not_comparable_abstention']} abstention, " \
                   f"{c['not_comparable_format']} format)"
        return f"{name}: {c['action_stable_and_window_changed']} of {c['comparable_questions']} comparable questions " \
               f"(of {c['all_questions']}; {c['not_comparable_abstention']} abstention, {c['not_comparable_format']} format)"

    return f"""# Captions

## Figure (fig_pilot_per_question)

**Per-question replies under option and segment-label swaps (replication_1124; 3 videos, 6 questions, 18 replies).**
Each row is one frozen question (Q1: action after the first sound occurrence; Q2: after the second).
Each condition cell shows the verbatim reply (option letter · segment code), the action it denotes (α or β, fixed per video) and the physical time window it cites (W1 or W2, source-video seconds in each video header).
*Options reversed* swaps which action the letters A/B denote; *segment labels reversed* swaps which window the codes E01/E02 denote; frames, their physical order and timestamps are unchanged.
A filled box marks a window decoded by the pre-registered strict rule; a hatched box marks a window recovered only by the post-hoc rule (segment code plus printed start and end seconds that exactly match the frozen mapping); dashed boxes mark abstention (answer and event both null).
The right-hand columns compare each intervention with the original under both rules.
P20_01 abstained in all six replies.
In P03_117 the action stayed the same while the cited window moved from {interval_text(p03[0])} to {interval_text(p03[1])} under option reversal; under segment-label reversal the code changed from E01 to E02 but the cited window stayed at {interval_text(p03[0])}.
The two questions of each video share the same clip, frames and options, so their outcomes are correlated: the figure shows three clusters, not six independent observations.
Model outputs are behavioural consistency records; no gold answers are used.

## Table (table_pilot_counts)

**Behavioural consistency counts (number of questions that were action-stable with a changed physical window, over comparable questions).**
Pre-registered strict parsing (primary) — {strict_line("option reversal", s_opt)}; {strict_line("segment-label reversal", s_lab)}.
Post-hoc format sensitivity (supplementary; rule defined after the replies were seen) — {strict_line("option reversal", p_opt)}; {strict_line("segment-label reversal", p_lab)}.
Non-comparable questions stay in every denominator, split by cause.
Both questions of a video share one clip and option set, so per-video counts are dependent; the sample supports no estimate of correctness or of generalisation across videos.
"""


# ---------------------------------------------------------------- figure

def draw_figure(responses, comparisons, meta, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 6, "svg.fonttype": "none", "svg.hashsalt": "pilot-figures",
        "pdf.fonttype": 42, "hatch.linewidth": 0.6,
    })
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    colors = {"α": "#2b6cb0", "β": "#c05621"}
    grey, light, changed_color = "#4a5568", "#a0aec0", "#b83280"
    mono = {"family": "DejaVu Sans Mono", "size": 5.6}
    cond_w, sub_w = 1.47, 0.73
    cols = [(0.05, 0.85)] + [(0.92 + i * cond_w, cond_w) for i in range(3)]
    comp_x = 0.92 + 3 * cond_w + 0.04
    sub_cols = [(comp_x + i * sub_w, sub_w) for i in range(4)]
    header_h, video_h, row_h, legend_h = 0.68, 0.30, 0.56, 0.42
    height = header_h + len(VIDEOS) * (video_h + 2 * row_h) + legend_h
    width = comp_x + 4 * sub_w + 0.05
    fig = plt.figure(figsize=(width, height))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.axis("off")

    titles = [("Question", ""),
              ("Original", "A=α B=β  |  E01=W1 E02=W2"),
              ("Options reversed", "A=β B=α  |  E01=W1 E02=W2"),
              ("Segment labels reversed", "A=α B=β  |  E01=W2 E02=W1")]
    for (x, w), (title, sub) in zip(cols, titles):
        ax.text(x + 0.04, 0.22, title, weight="bold", size=6.4, va="center")
        if sub:
            ax.text(x + 0.04, 0.5, sub, va="center", color=grey, **mono)
    for group, title in enumerate(("Options reversed\nvs. original", "Segment labels reversed\nvs. original")):
        x = sub_cols[2 * group][0]
        ax.text(x + 0.04, 0.22, title, weight="bold", size=6.2, va="center", linespacing=1.15)
        for k, name in enumerate(("strict (primary)", "post hoc")):
            ax.text(sub_cols[2 * group + k][0] + 0.04, 0.5, name, va="center", size=5.4, color=grey,
                    style="italic" if k else "normal")
    ax.plot([0.05, width - 0.05], [header_h - 0.02] * 2, color="black", lw=0.8)

    by_key = {(r["qid"], r["condition"]): r for r in responses}
    cmp_key = {(c["qid"], c["analysis"], c["intervention"]): c for c in comparisons}

    def window_boxes(x, y, r):
        for index in (1, 2):
            bx = x + (index - 1) * 0.3
            strict = r["strict_window"] == index
            recovered = r["posthoc_status"] == "recovered" and r["posthoc_window"] == index
            abstained = r["strict_status"] in ("abstained", "partially_abstained")
            color = colors.get(r["action_symbol"], grey)
            ax.add_patch(Rectangle((bx, y), 0.26, 0.15, facecolor=color if strict else "white",
                                   edgecolor=color if (strict or recovered) else light, lw=0.7,
                                   hatch="//////" if recovered else None, ls="--" if abstained else "-"))
            ax.text(bx + 0.13, y + 0.077, f"W{index}", ha="center", va="center", size=5.2,
                    color="white" if strict else (color if recovered else light),
                    weight="bold" if strict or recovered else "normal",
                    bbox=dict(boxstyle="square,pad=0.05", fc="white", ec="none") if recovered else None)

    def outcome(c):
        if not c["comparable"]:
            return ("not comparable", f"({c['not_comparable_reason']})"), light, "italic"
        action = "same action" if c["action_stable"] else "action changed"
        if c["physical_window_changed"]:
            return (action, f"W{c['original_window']} → W{c['test_window']}"), changed_color, "normal"
        return (action, f"W{c['original_window']} kept"), grey, "normal"

    y = header_h
    for video in VIDEOS:
        m = meta[video]
        ax.add_patch(Rectangle((0.05, y + 0.02), width - 0.1, video_h - 0.04, facecolor="#edf2f7", edgecolor="none"))
        ax.text(0.09, y + video_h / 2, video, weight="bold", size=6.6, va="center")
        ax.text(cols[1][0] + 0.04, y + video_h / 2,
                f"sound: {m['sound']}     α = {m['actions'][0]}     β = {m['actions'][1]}     "
                f"W1 = {interval_text(m['windows'][0])}     W2 = {interval_text(m['windows'][1])}",
                va="center", size=5.8, color="#1a202c")
        y += video_h
        for ordinal in (1, 2):
            qid = m["qids"][ordinal - 1]
            ax.text(0.09, y + 0.2, f"Q{ordinal}", weight="bold", size=6.2, va="center")
            ax.text(0.09, y + 0.37, f"after {'first' if ordinal == 1 else 'second'} sound", size=5.2,
                    color=grey, va="center")
            for column, condition in enumerate(CONDITIONS, 1):
                r = by_key[qid, condition]
                x = cols[column][0] + 0.04
                raw_event = "null" if r["raw_event"] is None else f'"{r["raw_event"]}"'
                raw_answer = "null" if r["raw_answer"] is None else f'"{r["raw_answer"]}"'
                ax.text(x, y + 0.11, f"{raw_answer} · {raw_event}", va="center", **mono)
                if r["strict_status"] in ("abstained", "partially_abstained"):
                    ax.text(x, y + 0.26, "abstained", va="center", color=grey, style="italic", size=5.8)
                else:
                    ax.text(x, y + 0.26, f"{r['action_symbol']}: {r['action']}", va="center",
                            color=colors[r["action_symbol"]], size=5.8)
                tag = {"valid": "valid", "format_not_comparable": "strict: not decodable",
                       "abstained": "", "partially_abstained": ""}[r["strict_status"]]
                window_boxes(x, y + 0.34, r)
                ax.text(x + 0.62, y + 0.415, tag, va="center", size=4.9, color=grey)
            for group, intervention in enumerate(INTERVENTIONS):
                for k, analysis in enumerate(ANALYSES):
                    (first, second), color, style = outcome(cmp_key[qid, analysis, intervention])
                    x = sub_cols[2 * group + k][0] + 0.04
                    ax.text(x, y + 0.2, first, va="center", size=5.4, color=color, style=style)
                    ax.text(x, y + 0.36, second, va="center", size=5.4, color=color, style=style,
                            weight="bold" if color == changed_color else "normal")
            y += row_h
            ax.plot([0.05, width - 0.05], [y, y], color="#e2e8f0", lw=0.5)
    for x in [c[0] for c in cols[1:]] + [c[0] for c in sub_cols[::2]]:
        ax.plot([x - 0.02, x - 0.02], [0.05, y], color="#cbd5e0", lw=0.5, zorder=0)
    for x in [c[0] for c in sub_cols[1::2]]:
        ax.plot([x - 0.02, x - 0.02], [0.42, y], color="#e2e8f0", lw=0.4, ls=":", zorder=0)

    ly = y + 0.14
    lx = 0.09
    ax.add_patch(Rectangle((lx, ly - 0.07), 0.26, 0.14, facecolor=colors["α"], edgecolor=colors["α"], lw=0.7))
    ax.text(lx + 0.32, ly, "window cited, strict rule (pre-registered)", va="center", size=5.4)
    lx += 2.0
    ax.add_patch(Rectangle((lx, ly - 0.07), 0.26, 0.14, facecolor="white", edgecolor=colors["α"], lw=0.7, hatch="//////"))
    ax.text(lx + 0.32, ly, "window recovered by post-hoc rule only", va="center", size=5.4)
    lx += 1.95
    ax.add_patch(Rectangle((lx, ly - 0.07), 0.26, 0.14, facecolor="white", edgecolor=light, lw=0.7, ls="--"))
    ax.text(lx + 0.32, ly, "abstained (answer and event null)", va="center", size=5.4)
    lx += 1.75
    ax.text(lx, ly, "colour = action:", va="center", size=5.4)
    ax.text(lx + 0.62, ly, "α", va="center", size=6, color=colors["α"], weight="bold")
    ax.text(lx + 0.75, ly, "β", va="center", size=6, color=colors["β"], weight="bold")
    ax.text(0.09, ly + 0.2, "Raw replies are verbatim (answer · event). Letters and segment codes are prompt labels; "
            "α/β and W1/W2 are the fixed actions and physical source-video windows they map to in each condition.",
            va="center", size=5.0, color=grey)

    fig.savefig(out_dir / f"{FIGURE}.pdf", metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(out_dir / f"{FIGURE}.svg", metadata={"Date": None})
    plt.close(fig)
    return matplotlib.__version__


# ---------------------------------------------------------------- driver

def video_meta(prompt_index):
    selected = {row["video_id"]: row for row in jsonl(SOURCE / "manifest" / "selected.jsonl")}
    meta = {}
    for (qid, condition), item in sorted(prompt_index.items(), key=lambda kv: (kv[1]["ordinal"], kv[0])):
        if condition != "original":
            continue
        m = meta.setdefault(item["video_id"], dict(
            sound=selected[item["video_id"]]["sound_class"],
            actions=[a.rstrip(".") for a in item["semantic_actions"].values()],
            windows=item["physical_window_order"], qids=[]))
        m["qids"].append(qid)
    return meta


def generate(out_dir):
    predictions, prompt_index = load_inputs()
    responses = build_responses(predictions, prompt_index)
    comparisons = compare(responses)
    counts = count(comparisons)
    checks = verify(responses, comparisons, counts)
    meta = video_meta(prompt_index)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "pilot_responses.csv", responses, list(responses[0]))
    write_csv(out_dir / "pilot_comparisons.csv", comparisons, list(comparisons[0]))
    rows = table_rows(counts)
    write_csv(out_dir / "table_pilot_counts.csv", rows, list(rows[0]))
    (out_dir / "table_pilot_counts.md").write_text(markdown_table(counts, responses, meta), encoding="utf-8")
    (out_dir / "table_pilot_counts.tex").write_text(latex_table(counts, responses, meta), encoding="utf-8")
    (out_dir / "captions.md").write_text(captions(counts, meta, responses), encoding="utf-8")
    mpl_version = draw_figure(responses, comparisons, meta, out_dir)
    inputs = ["predictions.jsonl", "manifest/prompts.jsonl", "manifest/input_hashes.json", "manifest/selected.jsonl",
              "summary.json", "comparisons.csv", "format_sensitivity.json"]
    report = {
        "source": "av-occurrence-study/run/replication_1124 (read-only)",
        "input_sha256": {name: sha256(SOURCE / name) for name in inputs},
        "matplotlib": mpl_version,
        "replies": {"total": len(responses),
                    "strict_valid": sum(r["strict_status"] == "valid" for r in responses),
                    "abstained": sum(r["strict_status"] == "abstained" for r in responses),
                    "strict_not_decodable": sum(r["strict_status"] == "format_not_comparable" for r in responses),
                    "posthoc_recovered": sum(r["posthoc_status"] == "recovered" for r in responses)},
        "denominators": {f"{a}/{i}": {k: counts[a, i, "all"][k] for k in ("all_questions", "comparable_questions",
                         "not_comparable_abstention", "not_comparable_format", "action_stable_and_window_changed")}
                         for a in ANALYSES for i in INTERVENTIONS},
        "checks": checks,
    }
    (out_dir / "verification.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="regenerate into a temporary directory and compare "
                        "with the committed outputs byte for byte")
    parser.add_argument("--out", type=Path, default=HERE, help="output directory (default: this directory)")
    args = parser.parse_args()
    if not args.check:
        report = generate(args.out)
        print(f"{len(report['checks'])} checks passed; wrote {len(OUTPUTS)} files to {args.out}")
        return
    with tempfile.TemporaryDirectory() as tmp:
        report = generate(Path(tmp))
        differing = [name for name in OUTPUTS if not (HERE / name).exists()
                     or (HERE / name).read_bytes() != (Path(tmp) / name).read_bytes()]
    print(f"{len(report['checks'])} recomputation checks passed")
    if differing:
        sys.exit("Committed outputs differ from regeneration: " + ", ".join(differing))
    print(f"All {len(OUTPUTS)} committed outputs are byte-identical to a fresh regeneration")


if __name__ == "__main__":
    main()
