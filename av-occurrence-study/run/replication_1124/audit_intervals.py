"""Report a supplemental format sensitivity check without modifying frozen predictions."""
from collections import Counter, defaultdict
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
CODE_WITH_SECONDS = re.compile(r"(E0[12])\s+(\d+(?:\.\d+)?)\s*[–-]\s*(\d+(?:\.\d+)?)\s*s", re.I)
FENCED = re.compile(r"```(?:json)?\s*([\s\S]+?)\s*```", re.I)


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def summarize(groups):
    total = defaultdict(Counter)
    per_video = defaultdict(lambda: defaultdict(Counter))
    for (video_id, _), group in groups.items():
        reference_answer, reference_interval = group["original"]
        for name in ("options_reversed", "segment_labels_reversed"):
            answer, interval = group[name]
            eligible = None not in (reference_answer, reference_interval, answer, interval)
            same = eligible and answer == reference_answer
            moved = eligible and interval != reference_interval
            counts = dict(all_questions=1, comparable_questions=int(eligible), action_stable=int(same),
                          physical_window_changed=int(moved), action_stable_and_window_changed=int(same and moved))
            total[name].update(counts)
            per_video[video_id][name].update(counts)
    return total, per_video


def main():
    predictions = rows(HERE / "predictions.jsonl")
    prompts = rows(HERE / "manifest" / "prompts.jsonl")
    expected = {(row["qid"], row["condition"]): row for row in prompts}
    if len(predictions) != len(prompts) or {(r["qid"], r["condition"]) for r in predictions} != set(expected):
        raise ValueError("This audit requires the complete original prediction batch")
    groups = defaultdict(dict)
    strict_groups = defaultdict(dict)
    recovered = []
    for row in predictions:
        interval = row["interval_s"]
        item = expected[row["qid"], row["condition"]]
        if row["result_format"] == "invalid_value" and row["answer_id"] is not None and interval is None:
            reply = row["raw_response"].strip()
            fenced = FENCED.fullmatch(reply)
            reply = json.loads(fenced[1] if fenced else reply)
            event = reply.get("event")
            match = CODE_WITH_SECONDS.fullmatch(event.strip()) if isinstance(event, str) else None
            if match:
                code, left, right = match.groups()
                mapping = item["segment_codes"].get(code.upper())
                if mapping and (float(left), float(right)) == (float(f"{mapping[0]:g}"), float(f"{mapping[1]:g}")):
                    interval = mapping
                    recovered.append(dict(qid=row["qid"], condition=row["condition"], raw_event=event, interval_s=interval))
        key = row["video_id"], row["qid"]
        groups[key][row["condition"]] = (row["answer_id"], interval)
        strict_groups[key][row["condition"]] = (
            (row["answer_id"], row["interval_s"]) if row["result_format"] == "ok" else (None, None)
        )
    metrics, video_metrics = summarize(groups)
    strict_metrics, strict_per_video = summarize(strict_groups)
    report = {"analysis": "post_hoc_code_and_matching_printed_seconds", "recovered_outputs": recovered,
              "metrics": metrics, "metrics_by_video": video_metrics, "strict_metrics": strict_metrics,
              "strict_metrics_by_video": strict_per_video}
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    with (HERE / "format_sensitivity.json").open("x") as stream:
        stream.write(text)
    print(text, end="")


if __name__ == "__main__":
    main()
