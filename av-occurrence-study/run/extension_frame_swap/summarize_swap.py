#!/usr/bin/env python3
"""Replay the exploratory frame-group swap from frozen inputs and first replies."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ARCHIVE = HERE.parent / "participant_extension" / "results_1124"
sys.path.insert(0, str(ARCHIVE))

import frozen
import summarize as extension

PLAN_SHA256 = "5ff907df5432b5230963e604fd5e4daad86b0edd4a122756a975e50ab545aba2"
ARCHIVE_SHA256 = "b87936a0fd0d2b0d5d1f75f93f2d31dd81ad7fbc942fe2491d12ed21adc86aaa"
CONDITION = "frame_groups_swapped"
COMPLETENESS = ("both_complete", "original_only_complete", "swapped_only_complete", "neither_complete")


def load_plan():
    plan_bytes = (HERE / "planned.jsonl").read_bytes()
    if sha256(plan_bytes).hexdigest() != PLAN_SHA256:
        raise ValueError("Frame-swap plan differs from the frozen digest")
    plan = frozen.read_jsonl(plan_bytes)
    original = [call for call in frozen.load()["calls"] if call["condition"] == "original"]
    for row, call in zip(plan, original, strict=True):
        fields = ("qid", "video_id", "ordinal", "status")
        if (any(row[name] != call[name] for name in fields)
                or row["source_run_order"] != call["run_order"] or row["condition"] != CONDITION):
            raise ValueError(f"{row['qid']}: frame-swap identity differs from the frozen original")
        if row["status"] == "technical_missing":
            if (row["stage"], row["reason"]) != (call["stage"], call["reason"]):
                raise ValueError(f"{row['qid']}: frame-swap missing-input reason differs from the original")
            continue
        frames = call["prompt"]["frames"]
        hashes = call["frames_sha256"]
        if (row["prompt_line"] != call["prompt_line"] or row["prompt_sha256"] != call["prompt_sha256"]
                or row["frames"] != frames[3:] + frames[:3] or row["frames_sha256"] != hashes[3:] + hashes[:3]):
            raise ValueError(f"{row['qid']}: expected original prompt and swapped frame groups")
    return plan, original


def index_swapped_replies(plan, replies):
    runnable = {row["qid"]: row for row in plan if row["status"] == "runnable"}
    by_question = {}
    for reply in replies:
        qid = reply["qid"]
        if reply["condition"] != CONDITION or qid not in runnable or qid in by_question:
            raise ValueError(f"{qid}: unknown, non-runnable or duplicate swapped reply")
        expected = runnable[qid]
        if any(reply[name] != expected[name] for name in ("prompt_sha256", "frames_sha256")):
            raise ValueError(f"{qid}: swapped reply input digests differ from the frozen plan")
        if not isinstance(reply["raw_response"], str):
            raise ValueError(f"{qid}: expected first reply text")
        by_question[qid] = reply
    return by_question


def check_attempts(plan, replies, attempts):
    """Match the single-attempt runner's terminal records to the reply ledger."""
    if attempts is None:
        if any(row["status"] == "runnable" and row["qid"] not in replies for row in plan):
            raise ValueError("A runnable question without a reply requires an attempts ledger")
        return {}

    if not attempts or attempts[0]["event"] != "run_started" or attempts[-1]["event"] != "run_finished":
        raise ValueError("Use the finalized frame-swap attempts ledger")
    if attempts[0]["plan_sha256"] != PLAN_SHA256:
        raise ValueError("Attempted generation used a different plan")
    by_question = defaultdict(list)
    planned = {row["qid"]: row for row in plan}
    for line, record in enumerate(attempts[1:-1], 2):
        qid = record["qid"]
        row = planned.get(qid)
        if row is None or any(record[name] != row[name] for name in (
                "video_id", "ordinal", "source_run_order", "condition")):
            raise ValueError(f"attempts.jsonl:{line}: attempted input does not match the plan")
        by_question[qid].append((line, record))

    failures = {}
    for row in plan:
        qid = row["qid"]
        records = by_question[qid]
        events = [record["event"] for _, record in records]
        if row["status"] == "technical_missing":
            expected = ["frozen_technical_missing"]
        else:
            expected = ["started", "generated" if qid in replies else "failed"]
        if events != expected:
            raise ValueError(f"{qid}: attempts ledger records {events}, expected {expected}")
        line, terminal = records[-1]
        if terminal["event"] in ("frozen_technical_missing", "failed"):
            detail = {"stage": terminal["stage"], "reason": terminal["reason"],
                      "log_reference": f"attempts.jsonl:{line}"}
            if row["status"] == "runnable":
                failures[qid] = detail
            elif any(detail[name] != row[name] for name in ("stage", "reason")):
                raise ValueError(f"{qid}: attempted missing-input reason differs from the plan")
    return failures


def decode_swapped(plan, original, by_question, failures, parser):
    decoded = {rule: [] for rule in extension.RULES}
    for row, call in zip(plan, original, strict=True):
        qid = row["qid"]
        identity = {name: call[name] for name in ("qid", "video_id", "pair_id", "ordinal", "run_order")}
        identity["condition"] = CONDITION
        if qid not in by_question:
            if row["status"] == "technical_missing":
                missing_detail = {"stage": row["stage"], "reason": row["reason"], "log_reference": "planned.jsonl"}
            else:
                missing_detail = failures[qid]
            for rule in decoded:
                decoded[rule].append(dict(identity, status="technical_missing", technical_missing=missing_detail))
            continue
        prompt = call["prompt"]
        raw = by_question[qid]["raw_response"]
        for rule in decoded:
            parsed = parser.parse_reply(raw, call["option_codes"], call["segment_codes"], rule=rule)
            interval = parsed["interval_s"]
            window = None if interval is None else prompt["physical_window_order"].index(interval) + 1
            action = prompt["semantic_actions"].get(parsed["answer_id"])
            decoded[rule].append(dict(identity, **parsed, status=extension.classify(parsed),
                                      physical_window=window, action=action, raw_response=raw))
    return decoded


def paired_row(original, swapped):
    result = extension.compare(original, swapped)
    result["video_id"] = original["video_id"]
    first_window = result["original_window"]
    swapped_window = result["intervention_window"]
    result["original_cited_image_group"] = first_window
    result["swapped_cited_image_group"] = 3 - swapped_window if swapped_window is not None else None
    result["retains_image_group_with_stable_action"] = (
        result["original_cited_image_group"] == result["swapped_cited_image_group"]
        if result["action_stable"] else None
    )
    del result["event"]
    for label, reply in (("original", original), ("swapped", swapped)):
        if reply["status"] == "technical_missing":
            result[f"{label}_technical_missing"] = reply["technical_missing"]
    return result


def count(numerator, denominator, planned):
    return {"numerator": numerator, "denominator": denominator, "planned_questions": planned,
            "proportion": numerator / denominator if denominator else None}


def comparison(questions):
    planned = len(questions)
    cells = Counter(row["cell"] for row in questions if row["comparable"])
    stable = sum(row["action_stable"] is True for row in questions)
    moved = sum(row["physical_window_changed"] is True for row in questions)
    follows = sum(row["retains_image_group_with_stable_action"] is True for row in questions)
    completeness = Counter()
    for row in questions:
        left, right = row["original_status"], row["intervention_status"]
        if "technical_missing" in (left, right):
            continue
        completeness[(left == "complete", right == "complete")] += 1
    returned_pairs = sum(completeness.values())
    comparable = sum(cells.values())
    return {
        "planned_questions": planned,
        "paired_returned_texts": returned_pairs,
        "unpaired_or_missing_texts": planned - returned_pairs,
        "paired_completeness": {
            name: completeness[key] for name, key in zip(
                COMPLETENESS, ((True, True), (True, False), (False, True), (False, False)), strict=True)
        },
        "comparable_questions": comparable,
        "cells": {name: cells[name] for name in extension.CELLS},
        "changed_displayed_interval": count(moved, comparable, planned),
        "retained_image_group_with_stable_action": count(follows, stable, planned),
    }


def compare_conditions(original, swapped):
    questions = [paired_row(first, second) for first, second in zip(original, swapped, strict=True)]
    groups = defaultdict(list)
    for row in questions:
        groups[row["video_id"]].append(row)
    return {
        "replies_by_condition": {"original": extension.reply_counts(original), CONDITION: extension.reply_counts(swapped)},
        "comparison": comparison(questions),
        "recordings": {video: comparison(rows) for video, rows in groups.items()},
        "questions": questions,
    }


def replay(replies_path, attempts_path):
    plan, original = load_plan()
    archived_bytes = (ARCHIVE / "replies.jsonl").read_bytes()
    if sha256(archived_bytes).hexdigest() != ARCHIVE_SHA256:
        raise ValueError("Archived replies differ from the published original generation")
    archived = [row for row in extension.ledger_rows(archived_bytes, "archived replies.jsonl")
                if row["condition"] == "original"]
    by_original = {row["qid"]: row for row in archived}
    replies_bytes = replies_path.read_bytes()
    attempts_bytes = attempts_path.read_bytes() if attempts_path else None
    attempts = extension.ledger_rows(attempts_bytes, "attempts.jsonl") if attempts_bytes is not None else None
    by_swapped = index_swapped_replies(plan, extension.ledger_rows(replies_bytes, "replies.jsonl"))
    failures = check_attempts(plan, by_swapped, attempts)
    parser = extension.load_parser()
    original_decoded, _ = extension.decode_calls(original, archived, [], parser)
    swapped_decoded = decode_swapped(plan, original, by_swapped, failures, parser)
    tokens = [{"qid": row["qid"], "original": by_original[row["qid"]]["prompt_tokens"],
               CONDITION: by_swapped[row["qid"]]["prompt_tokens"]} for row in plan if row["qid"] in by_swapped]
    return {
        "designation": "exploratory",
        "protocol_commit": "d1bd1a73f714d4ef4c4884134f5c64d178673b07",
        "archived_results_commit": "3df02a8e1b84d876ebdf8b54b9826d803860fcf1",
        "planned_sha256": PLAN_SHA256,
        "archived_replies_sha256": ARCHIVE_SHA256,
        "swapped_replies_sha256": sha256(replies_bytes).hexdigest(),
        "swapped_attempts_sha256": sha256(attempts_bytes).hexdigest() if attempts_bytes is not None else None,
        "parser_source_sha256": frozen.PARSER_SHA256,
        "primary_rule": extension.RULES[0],
        "sensitivity_rule": extension.RULES[1],
        "planned_input_missing_questions": [row for row in plan if row["status"] == "technical_missing"],
        "input_tokens": {"compared": len(tokens), "by_question": tokens,
                         "different": [row for row in tokens if row["original"] != row[CONDITION]]},
        "rules": {rule: compare_conditions(original_decoded[rule], swapped_decoded[rule])
                  for rule in extension.RULES},
    }


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--replies", type=Path, default=HERE / "replies.jsonl")
    cli.add_argument("--attempts", type=Path, help="Defaults to attempts.jsonl beside the supplied replies")
    cli.add_argument("--output", type=Path, required=True, help="Write a new JSON report")
    args = cli.parse_args()
    attempts = args.attempts
    if attempts is None:
        default = args.replies.with_name("attempts.jsonl")
        if default.is_file():
            attempts = default
    report = replay(args.replies, attempts)
    with args.output.open("x", encoding="utf-8") as output:
        output.write(json.dumps(report, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
