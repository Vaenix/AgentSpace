#!/usr/bin/env python3
"""Replay the same-host original-input repeat and rebase the frame-swap comparison."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
SWAP = HERE.parent / "extension_frame_swap"
sys.path.insert(0, str(SWAP))

import summarize_swap as swap
from summarize_swap import extension, frozen

PROTOCOL_COMMIT = "fd4d3afbfa8e05bc42993b2bd3d29f57bd59a4d8"
PROTOCOL_SHA256 = "fcb057e785e99dbc492c1a7488e5edece671068083787564ed4081062e580f79"
SWAP_COMMIT = "cb2b9ab601d6dacc148ca60640939f895fb06acf"
SWAP_SHA256 = {
    "replies.jsonl": "44f4f5dc529e7343a414eaa2ef46b8b809599104c672f5ca64db267768893aea",
    "attempts.jsonl": "97286c9b7f0b91fb9f347a26ab9ad47fcab6452d51fb97065ecb68a1e63c532a",
}
PARSED_FIELDS = ("status", "answer_id", "physical_window")
PARSED_FIELDS_WITH_DETAILS = (*PARSED_FIELDS, "action", "interval_s", "result_format")
CALL_EVENTS = ("frozen_technical_missing", "reused", "started", "generated", "failed")


def read_pinned(path, digest):
    data = path.read_bytes()
    if sha256(data).hexdigest() != digest:
        raise ValueError(f"{path.name} differs from the pinned study version")
    return data


def validate_attempts(calls, repeats, attempts):
    """Verify all 72 runner slots, including the 44 archived skip records."""
    if not attempts or attempts[0]["event"] != "run_started" or attempts[-1]["event"] != "run_finished":
        raise ValueError("Use the finalized original repeat attempts ledger")
    run_id = attempts[0]["run_id"]
    if any(row["run_id"] != run_id for row in [*attempts, *repeats]):
        raise ValueError("The repeat ledger must contain a single invocation")
    if attempts[0]["runner_sha256"] != frozen.file_sha256(swap.ARCHIVE / "run_generation.py"):
        raise ValueError("Runner fingerprint differs from the original runner")
    if attempts[0]["input_sha256"] != frozen.FROZEN_SHA256:
        raise ValueError("Repeat ledger reports different frozen inputs")

    repeated_qids = {reply["qid"] for reply in repeats}
    if any(row["condition"] != "original" for row in repeats) or len(repeated_qids) != len(repeats):
        raise ValueError("Repeat reply ledger must contain only one new original reply per question")
    observed = [row for row in attempts if row["event"] in CALL_EVENTS]
    expected = []
    for call in calls:
        if call["status"] == "technical_missing":
            events = ("frozen_technical_missing",)
        elif call["condition"] != "original":
            events = ("reused",)
        else:
            events = ("started", "generated" if call["qid"] in repeated_qids else "failed")
        expected.extend((call, event) for event in events)
    for (call, event), record in zip(expected, observed, strict=True):
        identity = ("run_order", "video_id", "pair_id", "qid", "ordinal", "condition")
        if record["event"] != event or any(call[name] != record[name] for name in identity):
            raise ValueError(f"{call['qid']}, {call['condition']}: unexpected repeat ledger event")
        if event == "frozen_technical_missing" and any(record[name] != call[name] for name in ("stage", "reason")):
            raise ValueError(f"{call['qid']}: frozen input failure differs from the original")
    totals = Counter(row["event"] for row in observed)
    for event in ("reused", "generated", "failed", "frozen_technical_missing"):
        if totals[event] != attempts[-1][event]:
            raise ValueError(f"Repeat ledger totals disagree for {event}")
    return {event: totals[event] for event in CALL_EVENTS}


def describe_pair(first, second, first_decoded, second_decoded):
    qid = first_decoded[extension.RULES[0]]["qid"]
    rule_rows = {}
    for rule in extension.RULES:
        original, repeat = first_decoded[rule], second_decoded[rule]
        rule_rows[rule] = {
            "first": {name: original.get(name) for name in PARSED_FIELDS_WITH_DETAILS},
            "repeat": {name: repeat.get(name) for name in PARSED_FIELDS_WITH_DETAILS},
            "equal": ({name: original.get(name) == repeat.get(name) for name in PARSED_FIELDS}
                      if first is not None and second is not None else None),
        }
    result = {
        "qid": qid,
        "video_id": first_decoded[extension.RULES[0]]["video_id"],
        "condition": "original",
        "first_raw": first["raw_response"] if first is not None else None,
        "repeat_raw": second["raw_response"] if second is not None else None,
        "first_input_tokens": first["prompt_tokens"] if first is not None else None,
        "repeat_input_tokens": second["prompt_tokens"] if second is not None else None,
        "rules": rule_rows,
    }
    paired = first is not None and second is not None
    result["exact_raw_match"] = result["first_raw"] == result["repeat_raw"] if paired else None
    result["input_tokens_match"] = result["first_input_tokens"] == result["repeat_input_tokens"] if paired else None
    for label, decoded in (("first", first_decoded), ("repeat", second_decoded)):
        detail = decoded[extension.RULES[0]].get("technical_missing")
        if detail is not None:
            result[f"{label}_technical_missing"] = detail
    return result


def replay(replies_path, attempts_path):
    read_pinned(HERE / "protocol.md", PROTOCOL_SHA256)
    plan, originals = swap.load_plan()
    original_bytes = read_pinned(swap.ARCHIVE / "replies.jsonl", swap.ARCHIVE_SHA256)
    archived = [row for row in extension.ledger_rows(original_bytes, "archived replies.jsonl")
                if row["condition"] == "original"]
    replies_bytes, attempts_bytes = replies_path.read_bytes(), attempts_path.read_bytes()
    repeats = extension.ledger_rows(replies_bytes, "repeated replies.jsonl")
    attempts = extension.ledger_rows(attempts_bytes, "repeated attempts.jsonl")
    counts = validate_attempts(frozen.load()["calls"], repeats, attempts)
    parser = extension.load_parser()
    archived_decoded, _ = extension.decode_calls(originals, archived, [], parser)
    repeat_decoded, _ = extension.decode_calls(originals, repeats, attempts, parser)
    swapped = swap.index_swapped_replies(
        plan, extension.ledger_rows(read_pinned(SWAP / "replies.jsonl", SWAP_SHA256["replies.jsonl"]), "swap replies"))
    swap_attempts = extension.ledger_rows(
        read_pinned(SWAP / "attempts.jsonl", SWAP_SHA256["attempts.jsonl"]), "swap attempts")
    failed = swap.check_attempts(plan, swapped, swap_attempts)
    swapped_decoded = swap.decode_swapped(plan, originals, swapped, failed, parser)

    first_by_qid = {row["qid"]: row for row in archived}
    repeat_by_qid = {row["qid"]: row for row in repeats}
    questions = []
    for index, call in enumerate(originals):
        qid = call["qid"]
        questions.append(describe_pair(first_by_qid.get(qid), repeat_by_qid.get(qid),
                                       {rule: rows[index] for rule, rows in archived_decoded.items()},
                                       {rule: rows[index] for rule, rows in repeat_decoded.items()}))
    planned = len(questions)
    paired = [row for row in questions if row["exact_raw_match"] is not None]
    return {
        "designation": "exploratory_same_host_repeatability",
        "protocol_commit": PROTOCOL_COMMIT,
        "protocol_sha256": PROTOCOL_SHA256,
        "archived_replies_sha256": swap.ARCHIVE_SHA256,
        "swapped_results_commit": SWAP_COMMIT,
        "swapped_ledger_sha256": SWAP_SHA256,
        "repeat_replies_sha256": sha256(replies_bytes).hexdigest(),
        "repeat_attempts_sha256": sha256(attempts_bytes).hexdigest(),
        "parser_source_sha256": frozen.PARSER_SHA256,
        "runner_event_counts_across_all_conditions": counts,
        "exact_raw_agreement": swap.count(sum(q["exact_raw_match"] for q in paired), len(paired), planned),
        "input_token_agreement": swap.count(sum(q["input_tokens_match"] for q in paired), len(paired), planned),
        "missing": [row for row in questions if row["exact_raw_match"] is None],
        "mismatches": [row for row in paired if not (row["exact_raw_match"] and row["input_tokens_match"]
                       and all(all(rule["equal"].values()) for rule in row["rules"].values()))],
        "rules": {
            rule: {
                "agreement": {
                    field: swap.count(sum(q["rules"][rule]["equal"][field] for q in paired), len(paired), planned)
                    for field in PARSED_FIELDS
                } | {"all_three": swap.count(sum(all(q["rules"][rule]["equal"].values()) for q in paired),
                                             len(paired), planned)},
                "frame_swap_with_repeat_baseline": swap.compare_conditions(repeat_decoded[rule], swapped_decoded[rule]),
            } for rule in extension.RULES
        },
        "questions": questions,
    }


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--replies", type=Path, default=HERE / "replies.jsonl")
    cli.add_argument("--attempts", type=Path, help="Defaults to attempts.jsonl beside the supplied replies")
    cli.add_argument("--output", type=Path, required=True, help="Write a new JSON report")
    args = cli.parse_args()
    report = replay(args.replies, args.attempts or args.replies.with_name("attempts.jsonl"))
    with args.output.open("x", encoding="utf-8") as output:
        output.write(json.dumps(report, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
