#!/usr/bin/env python3
"""Replay the gray-image control offline under the frozen primary parser (protocol 4e87d1e).

Exploratory control designed after the intact-image results of 3df02a8 were
read. Needs only Git, Python 3.11+ and the archived text: the frozen protocol
files, the extension's frozen inputs, the archived intact replies and the gray
replies.jsonl / attempts.jsonl. The parser source is read from the Git commit
pinned by the extension analysis and checked against its SHA-256 before use.
Both ledgers are re-decoded under both parsers so their stored strict fields
are verified; every statistic uses the primary matched_seconds rule only.
"""
import argparse
from collections import Counter
from hashlib import sha256
import importlib.util
import json
from pathlib import Path

import gray_inputs
from gray_inputs import frozen

spec = importlib.util.spec_from_file_location("extension_summarize", gray_inputs.EXTENSION_RESULTS / "summarize.py")
extension = importlib.util.module_from_spec(spec)
spec.loader.exec_module(extension)

PRIMARY_RULE = "matched_seconds"
CONDITIONS = ("intact", "gray")
INVALID = ("format_error", "code_seconds_conflict", "invalid_field")
PAIRED_CELLS = (
    "intact_complete_gray_complete", "intact_complete_gray_incomplete",
    "intact_incomplete_gray_complete", "both_incomplete",
)
SECONDARY_CELLS = (
    "action_retained_window_retained", "action_retained_window_changed",
    "action_changed_window_retained", "action_changed_window_changed",
)
RECORDING_GROUPS = ("at_least_one_loss", "at_least_one_gain", "loss_and_gain", "neither_change", "no_paired_reply")


def condition_counts(rows):
    statuses = Counter(row["status"] for row in rows)
    planned = len(rows)
    received = planned - statuses["technical_missing"]
    complete = statuses["complete"]
    return {
        "planned": planned,
        "received": received,
        "complete": complete,
        "complete_per_received": complete / received if received else None,
        "complete_per_planned": complete / planned,
        "both_abstained": statuses["both_abstained"],
        "partial_abstention": {"answer_only_abstained": statuses["answer_abstained"],
                               "event_only_abstained": statuses["event_abstained"]},
        "invalid": dict({name: statuses[name] for name in INVALID}, total=sum(statuses[name] for name in INVALID)),
        "technical_missing": {
            "total": statuses["technical_missing"],
            "frozen_frame_extraction": sum(row["status"] == "technical_missing"
                                           and row["technical_missing"]["stage"] != "generation" for row in rows),
            "generation": sum(row["status"] == "technical_missing"
                              and row["technical_missing"]["stage"] == "generation" for row in rows),
        },
        "statuses": {name: statuses[name] for name in extension.STATUSES},
    }


def question_row(intact, gray):
    paired = "technical_missing" not in (intact["status"], gray["status"])
    row = {
        "run_order": intact["run_order"], "pair_id": intact["pair_id"], "qid": intact["qid"],
        "ordinal": intact["ordinal"], "paired": paired,
        "intact_status": intact["status"], "gray_status": gray["status"],
        "intact_action": intact.get("action"), "gray_action": gray.get("action"),
        "intact_window": intact.get("physical_window"), "gray_window": gray.get("physical_window"),
        "completeness": None, "paired_cell": None, "secondary_cell": None,
    }
    if paired:
        before, after = intact["status"] == "complete", gray["status"] == "complete"
        row["completeness"] = {(True, True): "retained_complete", (True, False): "loss",
                               (False, True): "gain", (False, False): "retained_incomplete"}[before, after]
        row["paired_cell"] = {(True, True): "intact_complete_gray_complete",
                              (True, False): "intact_complete_gray_incomplete",
                              (False, True): "intact_incomplete_gray_complete",
                              (False, False): "both_incomplete"}[before, after]
        if before and after:
            action = "retained" if intact["answer_id"] == gray["answer_id"] else "changed"
            window = "retained" if intact["physical_window"] == gray["physical_window"] else "changed"
            row["secondary_cell"] = f"action_{action}_window_{window}"
    return row


def paired_table(questions):
    paired = [row for row in questions if row["paired"]]
    cells = Counter(row["paired_cell"] for row in paired)
    intact_complete = cells["intact_complete_gray_complete"] + cells["intact_complete_gray_incomplete"]
    gray_complete = cells["intact_complete_gray_complete"] + cells["intact_incomplete_gray_complete"]
    status_pairs = Counter((row["intact_status"], row["gray_status"]) for row in paired)
    return {
        "planned_keys": len(questions),
        "paired_keys": len(paired),
        "unpaired_keys": [{"qid": row["qid"], "intact_status": row["intact_status"], "gray_status": row["gray_status"]}
                          for row in questions if not row["paired"]],
        "cells": {name: cells[name] for name in PAIRED_CELLS},
        "intact_complete": intact_complete,
        "gray_complete": gray_complete,
        "intact_complete_proportion": intact_complete / len(paired) if paired else None,
        "gray_complete_proportion": gray_complete / len(paired) if paired else None,
        "difference_gray_minus_intact": (gray_complete - intact_complete) / len(paired) if paired else None,
        "difference_numerator": gray_complete - intact_complete,
        "status_pairs": [{"intact_status": left, "gray_status": right, "count": count}
                         for (left, right), count in sorted(status_pairs.items())],
    }


def secondary_table(questions):
    both = [row for row in questions if row["secondary_cell"]]
    cells = Counter(row["secondary_cell"] for row in both)
    return {"both_complete": len(both), "planned_keys": len(questions),
            "cells": {name: cells[name] for name in SECONDARY_CELLS}}


def recording_table(questions, recordings):
    per_recording, groups = {}, Counter()
    for pair in recordings:
        own = [row for row in questions if row["pair_id"] == pair]
        paired = [row for row in own if row["paired"]]
        loss = sum(row["completeness"] == "loss" for row in paired)
        gain = sum(row["completeness"] == "gain" for row in paired)
        flags = {
            "at_least_one_loss": bool(loss), "at_least_one_gain": bool(gain), "loss_and_gain": bool(loss and gain),
            "neither_change": bool(paired) and not loss and not gain, "no_paired_reply": not paired,
        }
        for name, flag in flags.items():
            groups[name] += flag
        per_recording[pair] = dict(paired_questions=len(paired), losses=loss, gains=gain, **flags, questions=own)
    return {
        "planned_recordings": len(recordings),
        "recordings_with_paired_reply": len(recordings) - groups["no_paired_reply"],
        "groups": {name: groups[name] for name in RECORDING_GROUPS},
        "recordings": per_recording,
    }


def token_check(keys, gray_rows, intact_rows):
    intact = {row["qid"]: row["prompt_tokens"] for row in intact_rows if row["condition"] == gray_inputs.CONDITION}
    gray = {row["qid"]: row for row in gray_rows}
    checked = []
    for call in keys:
        if call["qid"] not in gray:
            continue
        row = gray[call["qid"]]
        match = row["prompt_tokens"] == intact[call["qid"]]
        if (row["intact_prompt_tokens"], row["prompt_tokens_match"]) != (intact[call["qid"]], match):
            raise ValueError(f"{call['qid']}: stored token check disagrees with the intact ledger")
        checked.append({"qid": call["qid"], "intact_prompt_tokens": intact[call["qid"]],
                        "gray_prompt_tokens": row["prompt_tokens"], "match": match})
    mismatches = [row for row in checked if not row["match"]]
    return {"checked": len(checked), "matched": len(checked) - len(mismatches),
            "protocol_deviations": mismatches, "keys": checked}


def replay(replies_file, attempts_file):
    inputs = gray_inputs.load()
    keys = inputs["keys"]
    parser = extension.load_parser()
    reply_bytes, attempt_bytes = replies_file.read_bytes(), attempts_file.read_bytes()
    gray_rows = extension.ledger_rows(reply_bytes, "replies.jsonl")
    attempt_rows = extension.ledger_rows(attempt_bytes, "attempts.jsonl")
    for row in gray_rows:
        if row.get("image_condition") != "gray" or row.get("image_slots") != gray_inputs.SLOTS \
                or row.get("image_sha256") != gray_inputs.PROTOCOL_SHA256["gray.png"]:
            raise ValueError(f"{row['qid']}: reply is not a six-slot gray.png reply")

    all_calls = frozen.load()["calls"]
    intact_decoded, _ = extension.decode_calls(all_calls, inputs["intact_replies"], inputs["intact_attempts"], parser)
    gray_decoded, received = extension.decode_calls(keys, gray_rows, attempt_rows, parser)
    intact = [row for row in intact_decoded[PRIMARY_RULE] if row["condition"] == gray_inputs.CONDITION]
    gray = gray_decoded[PRIMARY_RULE]
    if not [row["qid"] for row in intact] == [row["qid"] for row in gray] == [call["qid"] for call in keys]:
        raise AssertionError("Intact and gray rows must follow the 24 planned keys")
    questions = [question_row(left, right) for left, right in zip(intact, gray)]
    paired = paired_table(questions)
    if sum(paired["cells"].values()) != paired["paired_keys"]:
        raise AssertionError("Paired cells must sum to the paired denominator")
    secondary = secondary_table(questions)
    if sum(secondary["cells"].values()) != secondary["both_complete"]:
        raise AssertionError("Secondary cells must sum to the both-complete subset")
    for rows, name in ((intact, "intact"), (gray, "gray")):
        for row in rows:
            row["image_condition"] = name

    return {
        "design": "exploratory control designed after the intact-image results of 3df02a8 had been read",
        "protocol_commit": gray_inputs.PROTOCOL_COMMIT,
        "protocol_plan_commit": gray_inputs.PROTOCOL_PLAN_COMMIT,
        "intact_results_commit": gray_inputs.INTACT_RESULTS_COMMIT,
        "input_sha256": inputs["fingerprints"],
        "gray_image": inputs["gray"],
        "parser_source_commit": extension.PARSER_SOURCE_COMMIT,
        "parser_source_sha256": frozen.PARSER_SHA256,
        "primary_rule": PRIMARY_RULE,
        "gray_replies_sha256": sha256(reply_bytes).hexdigest(),
        "gray_attempts_sha256": sha256(attempt_bytes).hexdigest(),
        "gray_replies_received": received,
        "strict_rows_verified_against_inference": received,
        "recordings": inputs["recordings"],
        "conditions": {"intact": condition_counts(intact), "gray": condition_counts(gray)},
        "prompt_tokens": token_check(keys, gray_rows, inputs["intact_replies"]),
        "paired": paired,
        "secondary": secondary,
        "per_recording": recording_table(questions, inputs["recordings"]),
        "technical_missing": [
            {"image_condition": row["image_condition"], "run_order": row["run_order"], "qid": row["qid"]}
            | row["technical_missing"] for row in intact + gray if row["status"] == "technical_missing"],
        "replies": {"intact": intact, "gray": gray},
    }


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--replies", type=Path, default=gray_inputs.HERE / "replies.jsonl")
    cli.add_argument("--attempts", type=Path, default=gray_inputs.HERE / "attempts.jsonl")
    cli.add_argument("--output", type=Path, required=True, help="Write a new JSON report")
    args = cli.parse_args()
    report = replay(args.replies, args.attempts)
    with args.output.open("x", encoding="utf-8") as output:
        output.write(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(f"Replayed {report['gray_replies_received']} gray replies over 24 planned keys. Wrote {args.output}.")


if __name__ == "__main__":
    main()
