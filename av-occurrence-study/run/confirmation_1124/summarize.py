#!/usr/bin/env python3
"""Replay a finalized confirmation run from frozen text inputs and raw replies."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
import subprocess
from types import ModuleType


HERE = Path(__file__).resolve().parent
STUDY = HERE.parents[1]
FROZEN_COMMIT = "86c00449b09e2540c0d2c8f52b197055142da3e8"
PROTOCOL_PARSER_COMMIT = "f6919987b6fe3ecf9746f7d9a5d7ab1edb8db3c4"
PARSER_SOURCE_COMMIT = "87ee601846c9062ec7fc98a206b633f83b0b7d28"
CONDITIONS = ("original", "options_reversed", "segment_labels_reversed")
RULES = ("matched_seconds", "strict")
ABSTENTIONS = ("both_abstained", "answer_abstained", "event_abstained")
STATUSES = (
    "technical_missing", "format_error", "code_seconds_conflict", "invalid_field",
    *ABSTENTIONS, "complete",
)
VIDEO_CLASSES = ("event_observed", "comparable_no_event", "no_comparable_question")


def git_blob(commit, path):
    return subprocess.check_output(["git", "show", f"{commit}:{path}"], cwd=STUDY)


def check_frozen_file(path):
    relative = path.relative_to(STUDY.parent)
    current = path.read_bytes()
    if current != git_blob(FROZEN_COMMIT, relative.as_posix()):
        raise ValueError(f"{relative} differs from the prospectively frozen commit")
    return sha256(current).hexdigest()


def load_parser():
    path = "av-occurrence-study/run/infer_mlx.py"
    source = git_blob(PARSER_SOURCE_COMMIT, path)
    digest = sha256(source).hexdigest()
    if digest != "4ebbd521d030f6cfdf9f4b4bead886bd53d076ffb6893cc542202680219fc027":
        raise ValueError("Pinned parser source fingerprint differs")
    module = ModuleType("frozen_confirmation_parser")
    exec(compile(source, f"{PARSER_SOURCE_COMMIT}:{path}", "exec"), module.__dict__)
    return module, digest


def read_jsonl(source):
    return [json.loads(line) for line in source.splitlines() if line.strip()]


def row_key(row):
    return row["qid"], row["condition"]


def index_rows(rows, label):
    indexed = {}
    for row in rows:
        key = row_key(row)
        if key in indexed:
            raise ValueError(f"Duplicate {label} row {key}")
        indexed[key] = row
    return indexed


def classify(parsed):
    result_format = parsed["result_format"]
    if result_format in ("invalid_json", "invalid_keys"):
        return "format_error"
    if result_format == "code_seconds_conflict":
        return "code_seconds_conflict"
    if result_format == "invalid_value":
        return "invalid_field"
    answer, event = parsed["answer_status"], parsed["event_status"]
    return {
        ("null", "null"): "both_abstained",
        ("null", "ok"): "answer_abstained",
        ("ok", "null"): "event_abstained",
        ("ok", "ok"): "complete",
    }[answer, event]


def decode_rows(prompt_rows, prediction_rows, failure_rows, parser, interval_label):
    prompts = index_rows(prompt_rows, "prompt")
    predictions = index_rows(prediction_rows, "prediction")
    failures = index_rows(failure_rows, "technical failure")
    if predictions.keys() - prompts.keys():
        raise ValueError("The prediction ledger contains keys outside the frozen prompts")
    missing = prompts.keys() - predictions.keys()
    if missing != failures.keys():
        raise ValueError(f"Technical failure records must cover exactly the missing prompt keys: {sorted(missing)}")
    decoded = {rule: [] for rule in RULES}
    for key, prompt in prompts.items():
        for code, interval in prompt["segment_codes"].items():
            label = f"{code} {interval_label(interval)} |"
            if sum(line.startswith(label) for line in prompt["prompt"].splitlines()) != 2:
                raise ValueError(f"{key}: inconsistent displayed interval for {code}")
        identity = {name: prompt[name] for name in ("qid", "condition", "pair_id", "video_id", "ordinal")}
        if key in missing:
            failure = failures[key]
            if not all(isinstance(failure.get(name), str) and failure[name].strip() for name in ("error", "log_reference")):
                raise ValueError(f"{key}: a technical failure needs an error and log reference")
            for rule in RULES:
                decoded[rule].append(dict(identity, status="technical_missing", technical_failure=failure))
            continue
        stored = predictions[key]
        if any(stored.get(name) != value for name, value in identity.items()):
            raise ValueError(f"Prediction identity does not match the prompt: {key}")
        for rule in RULES:
            parsed = parser(stored["raw_response"], prompt["option_codes"], prompt["segment_codes"], rule=rule)
            answer = parsed["answer_id"]
            interval = parsed["interval_s"]
            physical = None if interval is None else prompt["physical_window_order"].index(interval) + 1
            action = prompt["semantic_actions"].get(answer)
            if rule == "strict":
                expected = (answer, interval, parsed["result_format"], physical, action)
                recorded = tuple(stored[name] for name in (
                    "answer_id", "interval_s", "result_format", "physical_window", "action",
                ))
                if expected != recorded:
                    raise ValueError(f"Strict re-decoding differs from the stored inference row: {key}")
            decoded[rule].append(dict(
                identity, **parsed, status=classify(parsed), physical_window=physical, action=action,
            ))
    return decoded


def reply_counts(rows):
    counts = Counter(row["status"] for row in rows)
    received = len(rows) - counts["technical_missing"]
    abstained = sum(counts[name] for name in ABSTENTIONS)
    return {
        "planned": len(rows), "received": received, "abstained": abstained,
        "abstained_per_received": abstained / received if received else None,
        "statuses": {name: counts[name] for name in STATUSES},
    }


def compare(original, intervention):
    comparable = original["status"] == intervention["status"] == "complete"
    same_action = original["answer_id"] == intervention["answer_id"] if comparable else None
    changed_window = original["physical_window"] != intervention["physical_window"] if comparable else None
    return {
        "qid": original["qid"], "video_id": original["video_id"], "ordinal": original["ordinal"],
        "original_status": original["status"], "intervention_status": intervention["status"],
        "comparable": comparable, "action_stable": same_action, "physical_window_changed": changed_window,
        "action_stable_and_window_changed": same_action and changed_window if comparable else None,
    }


def comparison_counts(questions):
    total = len(questions)
    comparable = sum(row["comparable"] for row in questions)
    joint = sum(row["action_stable_and_window_changed"] is True for row in questions)
    unavailable = Counter(
        (row["original_status"], row["intervention_status"]) for row in questions if not row["comparable"]
    )
    return {
        "planned_questions": total,
        "comparable_questions": comparable,
        "non_comparable_questions": total - comparable,
        "action_stable": sum(row["action_stable"] is True for row in questions),
        "physical_window_changed": sum(row["physical_window_changed"] is True for row in questions),
        "action_stable_and_window_changed": joint,
        "joint_per_comparable": joint / comparable if comparable else None,
        "joint_per_planned": joint / total,
        "non_comparable_status_pairs": [
            {"original_status": left, "intervention_status": right, "count": count}
            for (left, right), count in sorted(unavailable.items())
        ],
    }


def summarize(rows, videos):
    by_condition = {condition: [] for condition in CONDITIONS}
    by_question = defaultdict(dict)
    for row in rows:
        by_condition[row["condition"]].append(row)
        by_question[row["qid"]][row["condition"]] = row
    comparisons = {}
    for condition in CONDITIONS[1:]:
        questions = [compare(group["original"], group[condition]) for group in by_question.values()]
        per_video = {}
        for video in videos:
            video_questions = [row for row in questions if row["video_id"] == video]
            counts = comparison_counts(video_questions)
            if counts["action_stable_and_window_changed"]:
                group = "event_observed"
            elif counts["comparable_questions"]:
                group = "comparable_no_event"
            else:
                group = "no_comparable_question"
            per_video[video] = dict(counts, video_class=group, questions=video_questions)
        classes = Counter(row["video_class"] for row in per_video.values())
        comparisons[condition] = {
            "overall": comparison_counts(questions),
            "per_video": per_video,
            "cross_video": {
                "planned_videos": len(videos),
                "counts": {group: classes[group] for group in VIDEO_CLASSES},
                "comparable_questions_by_video": {
                    video: result["comparable_questions"] for video, result in per_video.items()
                },
            },
        }
    return {
        "all_replies": reply_counts(rows),
        "replies_by_condition": {condition: reply_counts(group) for condition, group in by_condition.items()},
        "comparisons": comparisons,
        "replies": rows,
    }


def replay(predictions_file, failures_file=None):
    protocol = HERE / "协议.md"
    hashes_file = HERE / "manifest/input_hashes.json"
    selection_file = HERE / "manifest/selected.jsonl"
    prompt_file = HERE / "manifest/prompts.jsonl"
    fingerprints = {str(path.relative_to(HERE)): check_frozen_file(path)
                    for path in (protocol, hashes_file, selection_file, prompt_file)}
    hashes = json.loads(hashes_file.read_text(encoding="utf-8"))
    if fingerprints["manifest/prompts.jsonl"] != hashes["prompts_sha256"]:
        raise ValueError("Prompt manifest does not match its frozen input fingerprint")
    prompt_rows = read_jsonl(prompt_file.read_bytes())
    selected = read_jsonl(selection_file.read_bytes())
    videos = [row["video_id"] for row in selected]
    expected = [(row["video_id"], row["pair_id"], ordinal, condition)
                for row in selected for ordinal in (1, 2) for condition in CONDITIONS]
    observed = [(row["video_id"], row["pair_id"], row["ordinal"], row["condition"]) for row in prompt_rows]
    if len(videos) != len(set(videos)) or len(videos) != 3 or expected != observed:
        raise ValueError("Expected three distinct videos, two questions each, and three fixed-order conditions per question")
    raw = predictions_file.read_bytes()
    if raw and not raw.endswith(b"\n"):
        raise ValueError("Unterminated prediction row; use the finalized run ledger")
    failure_bytes = failures_file.read_bytes() if failures_file else b""
    prediction_rows = read_jsonl(raw)
    parser, parser_sha256 = load_parser()
    decoded = decode_rows(prompt_rows, prediction_rows, read_jsonl(failure_bytes), parser.parse_reply, parser.interval_label)
    return {
        "frozen_input_commit": FROZEN_COMMIT,
        "protocol_parser_commit": PROTOCOL_PARSER_COMMIT,
        "parser_source_commit": PARSER_SOURCE_COMMIT,
        "parser_source_sha256": parser_sha256,
        "text_input_sha256": fingerprints,
        "predictions_sha256": sha256(raw).hexdigest(),
        "technical_failures_sha256": sha256(failure_bytes).hexdigest() if failures_file else None,
        "strict_rows_verified_against_inference": len(prediction_rows),
        "primary_rule": RULES[0],
        "sensitivity_rule": RULES[1],
        "rules": {name: summarize(rows, videos) for name, rows in decoded.items()},
    }


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--predictions", type=Path, default=HERE / "predictions.jsonl")
    cli.add_argument("--technical-failures", type=Path, help="Final missing rows with qid, condition, error and log_reference")
    cli.add_argument("--output", type=Path, required=True, help="Write a new JSON report")
    args = cli.parse_args()
    report = replay(args.predictions, args.technical_failures)
    with args.output.open("x", encoding="utf-8") as output:
        output.write(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(f"Verified {report['strict_rows_verified_against_inference']} strict rows. Wrote {args.output}.")


if __name__ == "__main__":
    main()
