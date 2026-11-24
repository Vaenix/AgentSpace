#!/usr/bin/env python3
"""Replay the extension replies under both prespecified parsers (plan f653ba6f).

Needs only Git, Python and the archived text: frozen inputs, replies.jsonl and
attempts.jsonl. The parser source is read from a Git commit and checked
against the SHA-256 pinned by the protocol before it is executed.
"""
import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
import subprocess
from types import ModuleType

import frozen

PROTOCOL_PARSER_COMMIT = "f6919987b6fe3ecf9746f7d9a5d7ab1edb8db3c4"
PARSER_SOURCE_COMMIT = "87ee601846c9062ec7fc98a206b633f83b0b7d28"
RULES = ("matched_seconds", "strict")
CONTRASTS = ("options_reversed", "segment_labels_reversed")
ABSTENTIONS = ("both_abstained", "answer_abstained", "event_abstained")
STATUSES = (
    "technical_missing", "format_error", "code_seconds_conflict", "invalid_field",
    *ABSTENTIONS, "complete",
)
CELLS = (
    "action_stable_window_stable", "action_stable_window_changed",
    "action_changed_window_stable", "action_changed_window_changed",
)
RECORDING_CLASSES = ("event_observed", "comparable_no_event", "no_comparable_question")
TRANSITIONS = ("W1->W1", "W1->W2", "W2->W1", "W2->W2")


def load_parser():
    path = "av-occurrence-study/run/infer_mlx.py"
    source = subprocess.check_output(["git", "show", f"{PARSER_SOURCE_COMMIT}:{path}"], cwd=frozen.HERE)
    if sha256(source).hexdigest() != frozen.PARSER_SHA256:
        raise ValueError("Pinned parser source fingerprint differs")
    module = ModuleType("pinned_extension_parser")
    exec(compile(source, f"{PARSER_SOURCE_COMMIT}:{path}", "exec"), module.__dict__)
    return module


def classify(parsed):
    result_format = parsed["result_format"]
    if result_format in ("invalid_json", "invalid_keys"):
        return "format_error"
    if result_format == "code_seconds_conflict":
        return "code_seconds_conflict"
    if result_format == "invalid_value":
        return "invalid_field"
    return {
        ("null", "null"): "both_abstained",
        ("null", "ok"): "answer_abstained",
        ("ok", "null"): "event_abstained",
        ("ok", "ok"): "complete",
    }[parsed["answer_status"], parsed["event_status"]]


def ledger_rows(data, name):
    if data and not data.endswith(b"\n"):
        raise ValueError(f"Unterminated row in {name}; use the finalized ledger")
    return frozen.read_jsonl(data)


def unrecovered_failures(attempt_rows, missing_keys):
    """For each runnable key without a reply, the attempts that tried it and why they failed."""
    started = defaultdict(list)
    outcome = {}
    for line, row in enumerate(attempt_rows, 1):
        if row["event"] not in ("started", "failed", "generated"):
            continue
        key = frozen.key_of(row)
        if row["event"] == "started":
            started[key].append(line)
        else:
            outcome[key, row["run_id"]] = (line, row)
    failures = {}
    for key in missing_keys:
        records = []
        for line in started.get(key, []):
            run_id = attempt_rows[line - 1]["run_id"]
            done = outcome.get((key, run_id))
            if done and done[1]["event"] == "generated":
                raise ValueError(f"{key}: attempts.jsonl logs a generated reply that replies.jsonl lacks")
            if done:
                records.append({"reason": done[1]["error"], "log_reference": f"attempts.jsonl:{done[0]}"})
            else:
                records.append({"reason": "process ended during generation without a reply",
                                "log_reference": f"attempts.jsonl:{line}"})
        if not records:
            raise ValueError(f"{key}: no reply and no logged attempt")
        failures[key] = records
    return failures


def decode_calls(calls, reply_rows, attempt_rows, parser):
    by_key = {frozen.key_of(call): call for call in calls}
    replies = {}
    for row in reply_rows:
        key = frozen.key_of(row)
        if key in replies:
            raise ValueError(f"Duplicate reply for {key}")
        call = by_key.get(key)
        if call is None or call["status"] != "runnable":
            raise ValueError(f"Reply outside the runnable frozen keys: {key}")
        if any(row[name] != call[name] for name in ("run_order", "video_id", "pair_id", "ordinal")):
            raise ValueError(f"Reply identity differs from the frozen call: {key}")
        if row["prompt_sha256"] != call["prompt_sha256"]:
            raise ValueError(f"Reply was generated from another prompt: {key}")
        replies[key] = row
    missing = [frozen.key_of(c) for c in calls if c["status"] == "runnable" and frozen.key_of(c) not in replies]
    failures = unrecovered_failures(attempt_rows, missing)

    decoded = {rule: [] for rule in RULES}
    for call in calls:
        key = frozen.key_of(call)
        identity = {name: call[name] for name in ("run_order", "pair_id", "video_id", "qid", "ordinal", "condition")}
        if key not in replies:
            if call["status"] == "technical_missing":
                missing_record = {"stage": call["stage"], "reason": call["reason"],
                                  "log_reference": "frozen_inputs/calls.jsonl"}
            else:
                missing_record = {"stage": "generation", "reason": failures[key][-1]["reason"],
                                  "attempts": failures[key]}
            for rule in RULES:
                decoded[rule].append(dict(identity, status="technical_missing", technical_missing=missing_record))
            continue
        prompt = call["prompt"]
        stored = replies[key]
        for rule in RULES:
            parsed = parser.parse_reply(stored["raw_response"], call["option_codes"], call["segment_codes"], rule=rule)
            interval = parsed["interval_s"]
            physical = None if interval is None else prompt["physical_window_order"].index(interval) + 1
            action = prompt["semantic_actions"].get(parsed["answer_id"])
            if rule == "strict":
                expected = (parsed["answer_id"], interval, parsed["result_format"], physical, action)
                recorded = tuple(stored[name] for name in
                                 ("answer_id", "interval_s", "result_format", "physical_window", "action"))
                if expected != recorded:
                    raise ValueError(f"Strict re-decoding differs from the stored inference row: {key}")
            decoded[rule].append(dict(identity, **parsed, status=classify(parsed),
                                      physical_window=physical, action=action,
                                      raw_response=stored["raw_response"]))
    return decoded, len(replies)


def reply_counts(rows):
    counts = Counter(row["status"] for row in rows)
    received = len(rows) - counts["technical_missing"]
    abstained = sum(counts[name] for name in ABSTENTIONS)
    return {
        "planned": len(rows), "received": received, "abstained": abstained,
        "abstained_per_received": abstained / received if received else None,
        "statuses": {name: counts[name] for name in STATUSES},
    }


def compare(original, test):
    comparable = original["status"] == test["status"] == "complete"
    result = {
        "qid": original["qid"], "pair_id": original["pair_id"], "ordinal": original["ordinal"],
        "original_status": original["status"], "intervention_status": test["status"],
        "comparable": comparable,
        "original_action": original.get("action"), "intervention_action": test.get("action"),
        "original_window": original.get("physical_window"), "intervention_window": test.get("physical_window"),
        "action_stable": None, "physical_window_changed": None, "cell": None, "event": None,
    }
    if comparable:
        same = original["answer_id"] == test["answer_id"]
        changed = original["physical_window"] != test["physical_window"]
        result.update(action_stable=same, physical_window_changed=changed,
                      cell=f"action_{'stable' if same else 'changed'}_window_{'changed' if changed else 'stable'}",
                      event=same and changed)
    return result


def comparison_counts(questions):
    planned = len(questions)
    comparable = [row for row in questions if row["comparable"]]
    cells = Counter(row["cell"] for row in comparable)
    joint = cells["action_stable_window_changed"]
    pairs = Counter((row["original_status"], row["intervention_status"]) for row in questions if not row["comparable"])
    return {
        "planned_questions": planned,
        "comparable_questions": len(comparable),
        "non_comparable_questions": planned - len(comparable),
        "cells": {name: cells[name] for name in CELLS},
        "joint_events": joint,
        "joint_per_comparable": joint / len(comparable) if comparable else None,
        "joint_per_planned": joint / planned,
        "non_comparable_status_pairs": [
            {"original_status": left, "intervention_status": right, "count": count}
            for (left, right), count in sorted(pairs.items())
        ],
    }


def transitions(groups, recordings):
    """Original condition: first-sound question to second-sound question within each recording."""
    per_recording = []
    table = {name: {"same_action": 0, "changed_action": 0} for name in TRANSITIONS}
    for pair in recordings:
        first, second = groups[pair, 1]["original"], groups[pair, 2]["original"]
        complete = first["status"] == second["status"] == "complete"
        row = {
            "pair_id": pair, "first_status": first["status"], "second_status": second["status"],
            "first_action": first.get("action"), "second_action": second.get("action"),
            "first_window": first.get("physical_window"), "second_window": second.get("physical_window"),
            "complete": complete, "transition": None, "action": None,
        }
        if complete:
            row["transition"] = f"W{first['physical_window']}->W{second['physical_window']}"
            row["action"] = "same_action" if first["answer_id"] == second["answer_id"] else "changed_action"
            table[row["transition"]][row["action"]] += 1
        per_recording.append(row)
    complete_pairs = sum(row["complete"] for row in per_recording)
    if sum(sum(cell.values()) for cell in table.values()) != complete_pairs:
        raise AssertionError("Transition cells must sum to the complete pairs")
    return {"planned_pairs": len(recordings), "complete_pairs": complete_pairs,
            "table": table, "recordings": per_recording}


def summarize(rows, recordings):
    groups = defaultdict(dict)
    for row in rows:
        groups[row["pair_id"], row["ordinal"]][row["condition"]] = row
    replies_by_condition = {}
    for condition in frozen.CONDITIONS:
        replies_by_condition[condition] = reply_counts([row for row in rows if row["condition"] == condition])
    comparisons = {}
    for contrast in CONTRASTS:
        questions = [compare(groups[pair, ordinal]["original"], groups[pair, ordinal][contrast])
                     for pair in recordings for ordinal in (1, 2)]
        per_recording, classes = {}, Counter()
        for pair in recordings:
            own = [row for row in questions if row["pair_id"] == pair]
            counts = comparison_counts(own)
            if counts["joint_events"]:
                group = "event_observed"
            elif counts["comparable_questions"]:
                group = "comparable_no_event"
            else:
                group = "no_comparable_question"
            classes[group] += 1
            per_recording[pair] = {
                "comparable_questions": counts["comparable_questions"], "joint_events": counts["joint_events"],
                "joint_per_comparable": counts["joint_per_comparable"], "recording_class": group,
                "questions": [{name: row[name] for name in (
                    "qid", "ordinal", "original_status", "intervention_status", "comparable", "cell", "event",
                    "original_action", "intervention_action", "original_window", "intervention_window")}
                    for row in own],
            }
        comparisons[contrast] = {
            "questions": comparison_counts(questions),
            "recordings": per_recording,
            "across_recordings": {
                "planned_recordings": len(recordings),
                "classes": {name: classes[name] for name in RECORDING_CLASSES},
                "exactly_one_comparable_question": sum(
                    result["comparable_questions"] == 1 for result in per_recording.values()),
            },
        }
    missing = [{name: row[name] for name in ("run_order", "qid", "condition")} | row["technical_missing"]
               for row in rows if row["status"] == "technical_missing"]
    return {
        "all_calls": reply_counts(rows),
        "replies_by_condition": replies_by_condition,
        "technical_missing": {"planned_calls": len(rows), "missing_calls": len(missing), "keys": missing},
        "comparisons": comparisons,
        "original_first_to_second": transitions(groups, recordings),
        "replies": rows,
    }


def replay(replies_file, attempts_file):
    inputs = frozen.load()
    reply_bytes, attempt_bytes = replies_file.read_bytes(), attempts_file.read_bytes()
    parser = load_parser()
    decoded, received = decode_calls(inputs["calls"], ledger_rows(reply_bytes, "replies.jsonl"),
                                     ledger_rows(attempt_bytes, "attempts.jsonl"), parser)
    return {
        "plan_commit": frozen.PLAN_COMMIT,
        "approved_freeze_commit": frozen.APPROVED_FREEZE_COMMIT,
        "rebased_freeze_commit": frozen.REBASED_FREEZE_COMMIT,
        "frozen_input_sha256": inputs["fingerprints"],
        "protocol_parser_commit": PROTOCOL_PARSER_COMMIT,
        "parser_source_commit": PARSER_SOURCE_COMMIT,
        "parser_source_sha256": frozen.PARSER_SHA256,
        "replies_sha256": sha256(reply_bytes).hexdigest(),
        "attempts_sha256": sha256(attempt_bytes).hexdigest(),
        "replies_received": received,
        "strict_rows_verified_against_inference": received,
        "recordings": inputs["recordings"],
        "primary_rule": RULES[0],
        "sensitivity_rule": RULES[1],
        "rules": {rule: summarize(rows, inputs["recordings"]) for rule, rows in decoded.items()},
    }


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--replies", type=Path, default=frozen.HERE / "replies.jsonl")
    cli.add_argument("--attempts", type=Path, default=frozen.HERE / "attempts.jsonl")
    cli.add_argument("--output", type=Path, required=True, help="Write a new JSON report")
    args = cli.parse_args()
    report = replay(args.replies, args.attempts)
    with args.output.open("x", encoding="utf-8") as output:
        output.write(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(f"Replayed {report['replies_received']} replies over 72 call keys. Wrote {args.output}.")


if __name__ == "__main__":
    main()
