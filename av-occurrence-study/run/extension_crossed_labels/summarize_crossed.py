#!/usr/bin/env python3
"""Replay the four-condition paired analysis frozen in af01c3d from archived text.

Needs only Git, Python 3.11+ and the committed text: the frozen extension
inputs, the archived three-condition ledgers of 3df02a8 and this directory's
joint-reversal ledgers. The parser source is read from Git and checked against
its pinned SHA-256 before it runs. Every reply is decoded with its own prompt's
option and evidence maps, under the primary rule (matched_seconds) and the
frozen strict sensitivity rule.

  python3 summarize_crossed.py --output /tmp/crossed-replay.json
  cmp statistics.json /tmp/crossed-replay.json
"""
import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import sys

import crossed_inputs
from crossed_inputs import frozen

sys.path.insert(0, str(crossed_inputs.EXTENSION_RESULTS))
import summarize as archived  # noqa: E402  (results_1124/summarize.py at 3df02a8, fingerprint-checked)

CONDITIONS = crossed_inputs.CONDITIONS
RULES = archived.RULES
EDGES = (
    # (name, factor changed, label held fixed, from-condition, to-condition)
    ("answer_order_with_original_codes", "answer_order", "original_evidence_codes", "original", "options_reversed"),
    ("answer_order_with_reversed_codes", "answer_order", "reversed_evidence_codes",
     "segment_labels_reversed", "both_reversed"),
    ("evidence_codes_with_original_answers", "evidence_codes", "original_answer_order",
     "original", "segment_labels_reversed"),
    ("evidence_codes_with_reversed_answers", "evidence_codes", "reversed_answer_order",
     "options_reversed", "both_reversed"),
)
PRIMARY_EDGES = {"original_labels": EDGES[0], "reversed_labels": EDGES[1]}
PRIMARY_CLASSES = ("neither", "original_labels_only", "reversed_labels_only", "both")
RECORDING_CLASSES = ("discordant_question", "comparable_none_discordant", "no_all_four_complete_question")
DECODED_FIELDS = ("status", "answer_id", "action", "interval_s", "physical_window", "result_format")


def compare(left, right):
    """One single-factor edge for one question: semantic action stable/changed x physical window stable/changed."""
    comparable = left["status"] == right["status"] == "complete"
    result = {
        "from_status": left["status"], "to_status": right["status"], "comparable": comparable,
        "from_action": left.get("action"), "to_action": right.get("action"),
        "from_window": left.get("physical_window"), "to_window": right.get("physical_window"),
        "cell": None, "option_swap_event": None,
    }
    if comparable:
        same = left["answer_id"] == right["answer_id"]
        changed = left["physical_window"] != right["physical_window"]
        result.update(cell=f"action_{'stable' if same else 'changed'}_window_{'changed' if changed else 'stable'}",
                      option_swap_event=same and changed)
    return result


def edge_summary(questions, name):
    rows = [dict(qid=question["qid"], pair_id=question["pair_id"], ordinal=question["ordinal"],
                 **question["edges"][name]) for question in questions]
    comparable = [row for row in rows if row["comparable"]]
    cells = Counter(row["cell"] for row in comparable)
    reasons = Counter((row["from_status"], row["to_status"]) for row in rows if not row["comparable"])
    events = cells["action_stable_window_changed"]
    _, factor, held, left, right = next(edge for edge in EDGES if edge[0] == name)
    return {
        "factor_changed": factor, "held_fixed": held, "from_condition": left, "to_condition": right,
        "planned_questions": len(rows),
        "pairwise_complete": len(comparable),
        "non_comparable": len(rows) - len(comparable),
        "cells": {cell: cells[cell] for cell in archived.CELLS},
        "option_swap_events": events,
        "option_swap_events_per_pairwise_complete": events / len(comparable) if comparable else None,
        "non_comparable_status_pairs": [
            {"from_status": a, "to_status": b, "count": count} for (a, b), count in sorted(reasons.items())],
        "questions": rows,
    }


def analyse(rows, recordings):
    """rows: decoded rows of all four conditions (24 each) under one rule."""
    groups = {}
    for row in rows:
        groups.setdefault((row["pair_id"], row["ordinal"]), {})[row["condition"]] = row
    questions = []
    for pair in recordings:
        for ordinal in (1, 2):
            slots = groups[pair, ordinal]
            if sorted(slots) != sorted(CONDITIONS):
                raise ValueError(f"{pair} question {ordinal} lacks a condition")
            statuses = [slots[condition]["status"] for condition in CONDITIONS]
            complete = all(status == "complete" for status in statuses)
            edges = {name: compare(slots[left], slots[right]) for name, _, _, left, right in EDGES}
            label_class = None
            if complete:
                at_original = edges[PRIMARY_EDGES["original_labels"][0]]["option_swap_event"]
                at_reversed = edges[PRIMARY_EDGES["reversed_labels"][0]]["option_swap_event"]
                label_class = {(False, False): "neither", (True, False): "original_labels_only",
                               (False, True): "reversed_labels_only", (True, True): "both"}[at_original, at_reversed]
            questions.append({
                "qid": slots["original"]["qid"], "pair_id": pair, "ordinal": ordinal,
                "statuses": dict(zip(CONDITIONS, statuses)),
                "all_four_complete": complete,
                "event_at_original_labels": edges[EDGES[0][0]]["option_swap_event"] if complete else None,
                "event_at_reversed_labels": edges[EDGES[1][0]]["option_swap_event"] if complete else None,
                "primary_class": label_class,
                "conditions": {condition: {name: slots[condition].get(name) for name in DECODED_FIELDS}
                               for condition in CONDITIONS},
                "edges": edges,
            })

    complete = [question for question in questions if question["all_four_complete"]]
    classes = Counter(question["primary_class"] for question in complete)
    discordant = classes["original_labels_only"] + classes["reversed_labels_only"]
    patterns = Counter(tuple(question["statuses"].values()) for question in questions
                       if not question["all_four_complete"])
    per_recording, recording_classes = [], Counter()
    for pair in recordings:
        own = [question for question in questions if question["pair_id"] == pair]
        own_complete = [question for question in own if question["all_four_complete"]]
        if any(question["primary_class"] in ("original_labels_only", "reversed_labels_only") for question in own):
            group = "discordant_question"
        elif own_complete:
            group = "comparable_none_discordant"
        else:
            group = "no_all_four_complete_question"
        recording_classes[group] += 1
        per_recording.append({
            "pair_id": pair, "all_four_complete_questions": len(own_complete), "recording_class": group,
            "questions": [{name: question[name] for name in (
                "qid", "ordinal", "statuses", "all_four_complete", "event_at_original_labels",
                "event_at_reversed_labels", "primary_class")} for question in own],
        })
    missing = [{name: row[name] for name in ("run_order", "qid", "condition")} | row["technical_missing"]
               for row in rows if row["status"] == "technical_missing"]
    return {
        "conditions": {condition: archived.reply_counts([row for row in rows if row["condition"] == condition])
                       for condition in CONDITIONS},
        "technical_missing": {
            "planned_slots": len(rows), "missing_slots": len(missing),
            "by_condition": {condition: sum(row["condition"] == condition for row in missing)
                             for condition in CONDITIONS},
            "slots": missing,
        },
        "coverage": {
            "planned_questions": len(questions),
            "all_four_complete": len(complete),
            "all_four_complete_per_planned": len(complete) / len(questions),
        },
        "primary": {
            "definition": "option-swap event = same semantic action, other physical window; "
                          "original labels: original vs options_reversed; "
                          "reversed labels: segment_labels_reversed vs both_reversed",
            "common_denominator": len(complete),
            "planned_denominator": len(questions),
            "cells": {name: classes[name] for name in PRIMARY_CLASSES},
            "different_event_status": discordant,
            "different_event_status_per_common": discordant / len(complete) if complete else None,
            "events_at_original_labels": classes["original_labels_only"] + classes["both"],
            "events_at_reversed_labels": classes["reversed_labels_only"] + classes["both"],
        },
        "non_comparable_questions": {
            "count": len(questions) - len(complete),
            "status_patterns": [{"statuses": dict(zip(CONDITIONS, pattern)), "count": count}
                                for pattern, count in sorted(patterns.items())],
            "questions": [{"qid": question["qid"], "statuses": question["statuses"]}
                          for question in questions if not question["all_four_complete"]],
        },
        "edges": {name: edge_summary(questions, name) for name, *_ in EDGES},
        "recordings": {
            "planned_recordings": len(recordings),
            "classes": {name: recording_classes[name] for name in RECORDING_CLASSES},
            "per_recording": per_recording,
        },
        "questions": questions,
    }


def prompt_tokens(reply_rows):
    rows = [{"qid": row["qid"], "prompt_tokens": row["prompt_tokens"], "generated_tokens": row["generated_tokens"],
             "archived_prompt_tokens": row["archived_prompt_tokens"]} for row in reply_rows]
    return {
        "prompt_tokens_range": [min(row["prompt_tokens"] for row in rows), max(row["prompt_tokens"] for row in rows)]
        if rows else None,
        "generated_tokens_range": [min(row["generated_tokens"] for row in rows),
                                   max(row["generated_tokens"] for row in rows)] if rows else None,
        "reached_token_cap": sum(row["generated_tokens"] >= frozen.MAX_TOKENS for row in rows),
        "keys": rows,
    }


def replay(replies_file, attempts_file):
    inputs = crossed_inputs.load()
    if frozen.file_sha256(Path(archived.__file__)) != crossed_inputs.ARCHIVE_SHA256["summarize.py"]:
        raise ValueError("results_1124/summarize.py differs from the archive")
    reply_bytes, attempt_bytes = replies_file.read_bytes(), attempts_file.read_bytes()
    parser = archived.load_parser()
    archive_dir = crossed_inputs.EXTENSION_RESULTS
    old_decoded, old_received = archived.decode_calls(
        inputs["archived_calls"], archived.ledger_rows((archive_dir / "replies.jsonl").read_bytes(), "replies.jsonl"),
        archived.ledger_rows((archive_dir / "attempts.jsonl").read_bytes(), "attempts.jsonl"), parser)
    reply_rows = archived.ledger_rows(reply_bytes, "replies.jsonl")
    for row in reply_rows:
        key = next((key for key in inputs["keys"] if frozen.key_of(key) == frozen.key_of(row)), None)
        if key is not None and row.get("frames_sha256") != key["frames_sha256"]:
            raise ValueError(f"Reply used other images: {frozen.key_of(row)}")
    new_decoded, received = archived.decode_calls(
        inputs["keys"], reply_rows, archived.ledger_rows(attempt_bytes, "attempts.jsonl"), parser)
    rules = {}
    for rule in RULES:
        rows = old_decoded[rule] + new_decoded[rule]
        rules[rule] = analyse(rows, inputs["recordings"])
        rules[rule]["both_reversed_replies"] = new_decoded[rule]
    return {
        "protocol_commit": crossed_inputs.PROTOCOL_COMMIT,
        "archive_commit": crossed_inputs.ARCHIVE_COMMIT,
        "fingerprints": inputs["fingerprints"],
        "parser_source_commit": archived.PARSER_SOURCE_COMMIT,
        "parser_source_sha256": frozen.PARSER_SHA256,
        "replies_sha256": sha256(reply_bytes).hexdigest(),
        "attempts_sha256": sha256(attempt_bytes).hexdigest(),
        "archived_replies_received": old_received,
        "both_reversed_replies_received": received,
        "strict_rows_verified_against_inference": old_received + received,
        "recordings": inputs["recordings"],
        "conditions": list(CONDITIONS),
        "primary_rule": RULES[0],
        "sensitivity_rule": RULES[1],
        "both_reversed_tokens": prompt_tokens(reply_rows),
        "rules": rules,
    }


def main():
    cli = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    cli.add_argument("--replies", type=Path, default=crossed_inputs.HERE / "replies.jsonl")
    cli.add_argument("--attempts", type=Path, default=crossed_inputs.HERE / "attempts.jsonl")
    cli.add_argument("--output", type=Path, required=True, help="Write a new JSON report")
    args = cli.parse_args()
    report = replay(args.replies, args.attempts)
    with args.output.open("x", encoding="utf-8") as output:
        output.write(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(f"Replayed {report['archived_replies_received']} archived and {report['both_reversed_replies_received']} "
          f"joint-reversal replies over 96 planned slots. Wrote {args.output}.")


if __name__ == "__main__":
    main()
