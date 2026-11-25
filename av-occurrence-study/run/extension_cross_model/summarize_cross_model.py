#!/usr/bin/env python3
"""Pair SmolVLM2 and Qwen2.5-VL-7B first replies on the extension questions (protocol 2fac798).

Needs only Git, Python 3.11+ and the archived text. Both ledgers are decoded
with the parser source pinned by the extension plan, read from Git and checked
against its SHA-256 before execution. The Qwen ledger is the #11 archive at
3df02a8, verified by SHA-256; it is read, never rewritten.
"""
import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

import lock
import summarize as single  # participant_extension/results_1124/summarize.py

frozen = lock.frozen
MODELS = ("qwen2_5_vl_7b", "smolvlm2_2_2b")
JOINT_CLASSES = ("both", "qwen_only", "smolvlm2_only", "neither")


def decode_model(replies_path, attempts_path, calls, parser):
    reply_bytes, attempt_bytes = Path(replies_path).read_bytes(), Path(attempts_path).read_bytes()
    reply_rows = single.ledger_rows(reply_bytes, str(replies_path))
    decoded, received = single.decode_calls(calls, reply_rows, single.ledger_rows(attempt_bytes, str(attempts_path)),
                                            parser)
    ledger = {frozen.key_of(row): row for row in reply_rows}
    return {
        "replies_sha256": sha256(reply_bytes).hexdigest(),
        "attempts_sha256": sha256(attempt_bytes).hexdigest(),
        "replies_received": received,
        "decoded": decoded,
        "ledger": ledger,
    }


def reply_notes(rows, ledger, max_tokens):
    format_errors = [{"qid": row["qid"], "condition": row["condition"], "raw_response": row["raw_response"]}
                     for row in rows if row["status"] == "format_error"]
    budget = [{"qid": key[0], "condition": key[1], "status": next(
        r["status"] for r in rows if (r["qid"], r["condition"]) == key)}
              for key, stored in ledger.items() if stored.get("generated_tokens", 0) >= max_tokens]
    fields = Counter(f"answer {row['answer_status']} / event {row['event_status']}"
                     for row in rows if row["status"] not in ("technical_missing", "format_error"))
    return {"format_error_replies": format_errors,
            "token_budget_hits": sorted(budget, key=lambda r: (r["qid"], r["condition"])),
            "json_field_statuses": dict(sorted(fields.items()))}


def model_side(questions, video_of):
    counts = single.comparison_counts(questions)
    events = [row for row in questions if row["event"]]
    return {
        "comparable_questions": counts["comparable_questions"],
        "cells": counts["cells"],
        "joint_events": counts["joint_events"],
        "videos_with_event": sorted({video_of[row["pair_id"]] for row in events}),
        "non_comparable_status_pairs": counts["non_comparable_status_pairs"],
    }


def paired_contrast(groups, recordings, video_of, contrast):
    per_model = {}
    for model in MODELS:
        per_model[model] = [single.compare(groups[model][pair, ordinal]["original"],
                                           groups[model][pair, ordinal][contrast])
                            for pair in recordings for ordinal in (1, 2)]
    qwen, smol = per_model["qwen2_5_vl_7b"], per_model["smolvlm2_2_2b"]
    joint, rows = [], []
    for left, right in zip(qwen, smol):
        assert left["qid"] == right["qid"]
        both = left["comparable"] and right["comparable"]
        joint_class = None
        if both:
            joint_class = {(True, True): "both", (True, False): "qwen_only",
                           (False, True): "smolvlm2_only", (False, False): "neither"}[left["event"], right["event"]]
            joint.append((left, right, joint_class))
        rows.append({
            "qid": left["qid"], "pair_id": left["pair_id"], "video_id": video_of[left["pair_id"]],
            "ordinal": left["ordinal"], "jointly_comparable": both, "joint_class": joint_class,
            **{f"{model}_{name}": row[name] for model, row in (("qwen", left), ("smolvlm2", right))
               for name in ("original_status", "intervention_status", "comparable", "cell", "event",
                            "original_action", "intervention_action", "original_window", "intervention_window")},
        })
    classes = Counter(item[2] for item in joint)
    common = {
        "jointly_comparable_questions": len(joint),
        "qwen_joint_events": sum(item[0]["event"] for item in joint),
        "smolvlm2_joint_events": sum(item[1]["event"] for item in joint),
        "qwen_videos_with_event": sorted({video_of[item[0]["pair_id"]] for item in joint if item[0]["event"]}),
        "smolvlm2_videos_with_event": sorted({video_of[item[1]["pair_id"]] for item in joint if item[1]["event"]}),
        "classes": {name: classes[name] for name in JOINT_CLASSES},
    }
    if sum(common["classes"].values()) != len(joint):
        raise AssertionError("Joint classes must sum to the jointly comparable questions")
    return {
        "planned_questions": len(rows),
        "qwen2_5_vl_7b": model_side(qwen, video_of),
        "smolvlm2_2_2b": model_side(smol, video_of),
        "jointly_comparable": common,
        "questions": rows,
    }


def replay(smol_replies, smol_attempts):
    lock_data = lock.load_lock()
    inputs = frozen.load()
    calls, recordings = inputs["calls"], inputs["recordings"]
    video_of = {call["pair_id"]: call["video_id"] for call in calls}
    parser = single.load_parser()
    reference = lock.reference_paths()
    sides = {
        "qwen2_5_vl_7b": decode_model(reference["replies.jsonl"], reference["attempts.jsonl"], calls, parser),
        "smolvlm2_2_2b": decode_model(smol_replies, smol_attempts, calls, parser),
    }
    max_tokens = lock_data["decoding"]["max_tokens"]
    rules = {}
    for rule in single.RULES:
        groups = {}
        replies = {}
        for model, side in sides.items():
            rows = side["decoded"][rule]
            grouped = {}
            for row in rows:
                grouped.setdefault((row["pair_id"], row["ordinal"]), {})[row["condition"]] = row
            groups[model] = grouped
            replies[model] = {
                "all_calls": single.reply_counts(rows),
                "by_condition": {condition: single.reply_counts([r for r in rows if r["condition"] == condition])
                                 for condition in frozen.CONDITIONS},
                **reply_notes(rows, side["ledger"], max_tokens),
            }
        rules[rule] = {
            "replies": replies,
            "contrasts": {contrast: paired_contrast(groups, recordings, video_of, contrast)
                          for contrast in single.CONTRASTS},
        }
    return {
        "protocol_commit": lock.PROTOCOL_COMMIT,
        "protocol_sha256": lock.PROTOCOL_SHA256,
        "reference_commit": lock.REFERENCE_COMMIT,
        "reference_model": lock.REFERENCE_MODEL,
        "model": f"{lock_data['model']['repository']}@{lock_data['model']['revision'][:8]}",
        "frozen_input_sha256": inputs["fingerprints"],
        "parser_source_commit": single.PARSER_SOURCE_COMMIT,
        "parser_source_sha256": frozen.PARSER_SHA256,
        "ledgers": {model: {name: side[name] for name in ("replies_sha256", "attempts_sha256", "replies_received")}
                    for model, side in sides.items()},
        "recordings": recordings,
        "primary_rule": single.RULES[0],
        "sensitivity_rule": single.RULES[1],
        "rules": rules,
        "smolvlm2_decoded_replies": sides["smolvlm2_2_2b"]["decoded"],
    }


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--replies", type=Path, default=lock.HERE / "replies.jsonl")
    cli.add_argument("--attempts", type=Path, default=lock.HERE / "attempts.jsonl")
    cli.add_argument("--output", type=Path, required=True, help="Write a new JSON report")
    args = cli.parse_args()
    report = replay(args.replies, args.attempts)
    with args.output.open("x", encoding="utf-8") as output:
        output.write(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    counts = {model: side["replies_received"] for model, side in report["ledgers"].items()}
    print(f"Paired replies over 72 call keys: {counts}. Wrote {args.output}.")


if __name__ == "__main__":
    main()
