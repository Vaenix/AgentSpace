"""Replay the frozen 2026-11-24 replies offline under a selectable parse rule.

Reads only predictions.jsonl and manifest/prompts.jsonl; no model, video or GPU.
Re-decodes every stored raw response with the strict rule first and stops if it
disagrees with the stored fields or summary.json, then decodes with the chosen
rule and writes the new statistics to a separate, newly created JSON file.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "manifest"
sys.path.insert(0, str(HERE.parent))
from infer_mlx import PARSE_RULES, interval_label, parse_reply  # noqa: E402
from replicate import CONDITIONS, content_hash, read  # noqa: E402

METRICS = (
    "all_questions", "comparable_questions", "action_stable",
    "physical_window_changed", "action_stable_and_window_changed",
)


def index_prompts(prompts):
    indexed = {}
    for item in prompts:
        key = item["qid"], item["condition"]
        if key in indexed:
            raise ValueError(f"duplicate prompt row: {key}")
        for code, interval in item["segment_codes"].items():
            if f"\n{code} {interval_label(interval)} |" not in item["prompt"]:
                raise ValueError(f"{key}: prompt does not display {code} as {interval_label(interval)}")
        indexed[key] = item
    return indexed


def outcome(parsed):
    """Reply class shared by both rules; strict reports explicit nulls as ok."""
    statuses = parsed["answer_status"], parsed["event_status"]
    if parsed["result_format"] != "ok":
        return parsed["result_format"]
    if statuses == ("null", "null"):
        return "abstained"
    if "null" in statuses:
        return "partially_abstained"
    return "ok"


def decode_all(predictions, prompts, rule):
    rows = []
    for stored in predictions:
        item = prompts[stored["qid"], stored["condition"]]
        parsed = parse_reply(stored["raw_response"], item["option_codes"], item["segment_codes"], rule)
        interval = parsed["interval_s"]
        rows.append({
            "qid": stored["qid"], "video_id": item["video_id"], "ordinal": item["ordinal"],
            "condition": stored["condition"], "raw_response": stored["raw_response"],
            "answer_id": parsed["answer_id"], "action": item["semantic_actions"].get(parsed["answer_id"]),
            "interval_s": interval,
            "physical_window": item["physical_window_order"].index(interval) + 1 if interval is not None else None,
            "result_format": parsed["result_format"], "answer_status": parsed["answer_status"],
            "event_status": parsed["event_status"], "outcome": outcome(parsed),
            "stored_strict_result_format": stored["result_format"],
        })
    return rows


def compare(rows):
    """Intervention metrics, as in replicate.py summarize, plus reasons for the rest."""
    by_qid = defaultdict(dict)
    for row in rows:
        by_qid[row["qid"]][row["condition"]] = row
    overall = {condition: Counter() for condition in CONDITIONS[1:]}
    reasons = {condition: Counter() for condition in CONDITIONS[1:]}
    by_video = defaultdict(lambda: {condition: Counter() for condition in CONDITIONS[1:]})
    for group in by_qid.values():
        original = group["original"]
        for condition in CONDITIONS[1:]:
            test = group[condition]
            comparable = all(row["outcome"] == "ok" and row["physical_window"] is not None for row in (original, test))
            action_same = comparable and original["answer_id"] == test["answer_id"]
            changed = comparable and original["physical_window"] != test["physical_window"]
            counts = dict(all_questions=1, comparable_questions=int(comparable), action_stable=int(action_same),
                          physical_window_changed=int(changed), action_stable_and_window_changed=int(action_same and changed))
            overall[condition].update(counts)
            by_video[original["video_id"]][condition].update(counts)
            if not comparable:
                reasons[condition][f"original={original['outcome']}, test={test['outcome']}"] += 1
    as_dict = lambda counter: {key: counter[key] for key in METRICS}  # noqa: E731
    return (
        {condition: as_dict(counter) for condition, counter in overall.items()},
        {condition: dict(sorted(counter.items())) for condition, counter in reasons.items()},
        {video: {condition: as_dict(counter) for condition, counter in group.items()} for video, group in by_video.items()},
    )


def replay(predictions, prompt_rows, rule, strict_summary=None):
    prompts = index_prompts(prompt_rows)
    keys = [(row["qid"], row["condition"]) for row in predictions]
    if len(set(keys)) != len(keys) or set(keys) != set(prompts):
        raise ValueError("Predictions do not cover the frozen prompt rows exactly once")

    strict = decode_all(predictions, prompts, "strict")
    mismatched = [
        (row["qid"], row["condition"]) for row, stored in zip(strict, predictions)
        if (row["answer_id"], row["interval_s"], row["result_format"])
        != (stored["answer_id"], stored["interval_s"], stored["result_format"])
    ]
    if mismatched:
        raise ValueError(f"Strict re-decoding differs from stored predictions: {mismatched}")
    strict_metrics = compare(strict)[0]
    if strict_summary is not None and strict_metrics != strict_summary:
        raise ValueError("Strict re-decoding does not reproduce summary.json")

    rows = strict if rule == "strict" else decode_all(predictions, prompts, rule)
    metrics, reasons, by_video = compare(rows)
    outcomes = {condition: Counter() for condition in CONDITIONS}
    for row in rows:
        outcomes[row["condition"]].update(total=1, **{row["outcome"]: 1})
    return {
        "parse_rule": rule,
        "strict_recheck": {
            "replies_rechecked": len(strict), "mismatched_replies": 0,
            "reproduces_summary_json": strict_summary is not None, "metrics": strict_metrics,
        },
        "reply_outcomes_by_condition": {
            condition: {"total": counter["total"], **dict(sorted((k, v) for k, v in counter.items() if k != "total"))}
            for condition, counter in outcomes.items()
        },
        "metrics": metrics,
        "not_comparable_reasons": reasons,
        "metrics_by_video": by_video,
        "replies": rows,
    }


def frozen_replay(rule):
    prompt_file = MANIFEST / "prompts.jsonl"
    prediction_file = HERE / "predictions.jsonl"
    frozen = json.loads((MANIFEST / "input_hashes.json").read_text())
    if content_hash(prompt_file) != frozen["prompts_sha256"]:
        raise ValueError("Prompt manifest differs from its frozen fingerprint")
    summary = json.loads((HERE / "summary.json").read_text())
    result = replay(read(prediction_file), read(prompt_file), rule, summary)
    result["inputs"] = {
        "predictions": "predictions.jsonl", "predictions_sha256": content_hash(prediction_file),
        "prompts": "manifest/prompts.jsonl", "prompts_sha256": content_hash(prompt_file),
    }
    if rule == "matched_seconds":
        post_hoc = json.loads((HERE / "format_sensitivity.json").read_text())
        result["matches_post_hoc_format_sensitivity"] = (
            result["metrics"] == post_hoc["metrics"] and result["metrics_by_video"] == post_hoc["metrics_by_video"]
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rule", choices=PARSE_RULES, default="matched_seconds")
    parser.add_argument("--output", type=Path, required=True, help="New JSON file for the replay statistics")
    args = parser.parse_args()
    try:
        result = frozen_replay(args.rule)
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    except (OSError, ValueError) as exc:
        parser.exit(1, f"{exc}\n")
    print(f"rule={args.rule}; strict re-decoding matched {result['strict_recheck']['replies_rechecked']} stored replies")
    for condition, counts in result["reply_outcomes_by_condition"].items():
        print(f"{condition}: " + ", ".join(f"{key}={value}" for key, value in counts.items()))
    for condition, counts in result["metrics"].items():
        reasons = "; ".join(f"{key} ×{value}" for key, value in result["not_comparable_reasons"][condition].items())
        print(f"{condition}: comparable {counts['comparable_questions']}/{counts['all_questions']}, "
              f"action stable and window changed {counts['action_stable_and_window_changed']}"
              + (f"; not comparable: {reasons}" if reasons else ""))


if __name__ == "__main__":
    main()
