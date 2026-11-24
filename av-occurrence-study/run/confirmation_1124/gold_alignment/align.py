#!/usr/bin/env python3
"""Align frozen human gold with the confirmation_1124 two-choice replies offline.

Replies are decoded with each condition's own A/B and E01/E02 codes by the
batch's frozen decoder (``summarize.decode_rows``), questions are aligned by
pair_id, video_id and ordinal, actions only through normalized-text identity
or an explicit human mapping, and every aligned pair is scored by
``eval/score.py`` against the adjudicated original-video intervals.
"""

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
BATCH = HERE.parent
STUDY = BATCH.parents[1]
sys.path[:0] = [str(BATCH), str(STUDY / "eval")]

import summarize  # noqa: E402  frozen decoder of this batch
import score  # noqa: E402
from freeze_reviews import action_id as gold_action_id  # noqa: E402


CONDITIONS = summarize.CONDITIONS
RULES = summarize.RULES
STATUSES = summarize.STATUSES
FROZEN_PREDICTIONS_SHA256 = "3e465d57ad01f4a09ed31beb0c85711c79872cb3dfaa78c1b3431a712a2d6aa9"
FIELD_STATUSES = ("valid", "explicit_null", "invalid_field", "code_seconds_conflict", "format_error", "technical_missing")
FIELD_STATUS = {"ok": "valid", "null": "explicit_null", "invalid": "invalid_field", "conflict": "code_seconds_conflict"}
UNUSABLE_ANSWER = "unusable_answer:"
PAIR_STATUSES = (
    "no_gold", "excluded", "undecided", "mapping_pending", "answer_not_in_original_options", "aligned",
)
DECISIONS = {"纳入": "included", "排除": "excluded", "待定": "undecided"}
MAPPING_COLUMNS = [
    "题对", "视频", "金标指纹", "原选项ID", "原选项文本", "原选项来源序数",
    "金标动作1", "金标动作2", "自动判定", "映射", "映射人", "理由",
]
IDENTITY_COLUMNS = MAPPING_COLUMNS[:9]
MAPPING_VALUES = {"金标1": 1, "金标2": 2, "无对应": "none"}
NOT_IN_GOLD = "original_option_not_in_gold:"


def sha256_bytes(raw):
    return hashlib.sha256(raw).hexdigest()


def semantic_id(epic_action_id):
    """Opaque option ID used by prepare_epic_smoke.py for an EPIC narration ID."""
    return hashlib.blake2s(epic_action_id.encode(), digest_size=8).hexdigest()


def comparable_action(text):
    """Normalized action identity shared by gold IDs and original option texts.

    The prompt builder appended one full stop to each source hint; removing it
    and applying freeze_reviews.action_id is the only automatic equivalence.
    """
    return gold_action_id(text.strip().removesuffix("."))


# ---------------------------------------------------------------- batch inputs

def check_batch(selected, prompts, predictions):
    """Verify keys and the option/segment code maps; return the original options per pair."""
    pair_ids = [row["pair_id"] for row in selected]
    videos = [row["video_id"] for row in selected]
    if len(set(pair_ids)) != len(pair_ids) or len(set(videos)) != len(videos):
        raise ValueError("Batch selection needs distinct pair_id and video_id values")
    expected = [(row["video_id"], row["pair_id"], ordinal, condition)
                for row in selected for ordinal in (1, 2) for condition in CONDITIONS]
    observed = [(row["video_id"], row["pair_id"], row["ordinal"], row["condition"]) for row in prompts]
    if expected != observed:
        raise ValueError("Prompt rows must cover every pair, both ordinals and the three conditions in order")
    options, windows, qids = {}, {}, {}
    for pair in selected:
        occurrences = sorted(pair["occurrences"], key=lambda item: item["ordinal"])
        if [item["ordinal"] for item in occurrences] != [1, 2]:
            raise ValueError(f"{pair['pair_id']}: expected source ordinals 1 and 2")
        options[pair["pair_id"]] = {
            semantic_id(item["action_id"]): {
                "text": item["action_hint"].strip().removesuffix(".") + ".",
                "source_ordinal": item["ordinal"],
                "epic_action_id": item["action_id"],
            }
            for item in occurrences
        }
        windows[pair["pair_id"]] = [item["sound_s"] for item in occurrences]
    by_question = {}
    for row in prompts:
        pair_id, ordinal = row["pair_id"], row["ordinal"]
        texts = {key: value["text"] for key, value in options[pair_id].items()}
        if row["semantic_actions"] != texts:
            raise ValueError(f"{row['qid']} {row['condition']}: options differ from the selected source actions")
        if sorted(row["option_codes"]) != ["A", "B"] or set(row["option_codes"].values()) != set(texts):
            raise ValueError(f"{row['qid']} {row['condition']}: A/B must cover both source actions")
        if row["physical_window_order"] != windows[pair_id]:
            raise ValueError(f"{row['qid']} {row['condition']}: physical windows differ from the source intervals")
        if sorted(row["segment_codes"]) != ["E01", "E02"] or sorted(row["segment_codes"].values()) != sorted(windows[pair_id]):
            raise ValueError(f"{row['qid']} {row['condition']}: E01/E02 must cover both physical windows")
        if qids.setdefault((pair_id, ordinal), row["qid"]) != row["qid"]:
            raise ValueError(f"{pair_id} ordinal {ordinal}: conditions disagree on the question ID")
        by_question.setdefault((pair_id, ordinal), {})[row["condition"]] = row
    for key, group in by_question.items():
        original = group["original"]
        if original["segment_codes"] != {"E01": windows[key[0]][0], "E02": windows[key[0]][1]}:
            raise ValueError(f"{original['qid']}: original condition must label the earlier window E01")
        swapped_options = {"A": original["option_codes"]["B"], "B": original["option_codes"]["A"]}
        if group["options_reversed"]["option_codes"] != swapped_options:
            raise ValueError(f"{original['qid']}: options_reversed must swap only A and B")
        if group["options_reversed"]["segment_codes"] != original["segment_codes"]:
            raise ValueError(f"{original['qid']}: options_reversed changed the evidence codes")
        labels = group["segment_labels_reversed"]
        if labels["option_codes"] != original["option_codes"] or labels["segment_codes"] != {
            "E01": original["segment_codes"]["E02"], "E02": original["segment_codes"]["E01"],
        }:
            raise ValueError(f"{original['qid']}: segment_labels_reversed must swap only E01 and E02")
    prompt_keys = {summarize.row_key(row) for row in prompts}
    prediction_keys = [summarize.row_key(row) for row in predictions]
    if len(set(prediction_keys)) != len(prediction_keys) or not set(prediction_keys) <= prompt_keys:
        raise ValueError("Prediction keys must be unique and belong to the frozen prompts")
    precheck = {
        "videos": len(selected),
        "pairs": len(selected),
        "questions": len(by_question),
        "prompt_keys": len(prompt_keys),
        "prediction_keys": len(prediction_keys),
        "keys_without_prediction": len(prompt_keys) - len(prediction_keys),
        "conditions": list(CONDITIONS),
        "checked": [
            "pair_id, video_id, ordinal and condition order of every prompt row",
            "option IDs equal blake2s-8 of the selected EPIC action IDs and texts equal the source hints",
            "options_reversed swaps only A/B; segment_labels_reversed swaps only E01/E02",
            "physical windows equal the selected source sound intervals",
            "prediction keys are unique and frozen prompt keys",
        ],
    }
    return options, qids, precheck


def load_frozen_batch(predictions_file, failures_file=None, diagnostic=False):
    """Load the real batch after the frozen-input checks used by summarize.py.

    Only the archived reply ledger (pinned SHA-256, no technical-failure file)
    yields a confirmation_1124 result; any other ledger needs diagnostic=True
    and is labelled as a diagnostic in the report.
    """
    protocol, hashes_file = BATCH / "协议.md", BATCH / "manifest/input_hashes.json"
    selection_file, prompt_file = BATCH / "manifest/selected.jsonl", BATCH / "manifest/prompts.jsonl"
    fingerprints = {str(path.relative_to(BATCH)): summarize.check_frozen_file(path)
                    for path in (protocol, hashes_file, selection_file, prompt_file)}
    if fingerprints["manifest/prompts.jsonl"] != json.loads(hashes_file.read_text(encoding="utf-8"))["prompts_sha256"]:
        raise ValueError("Prompt manifest does not match its frozen input fingerprint")
    raw = predictions_file.read_bytes()
    if raw and not raw.endswith(b"\n"):
        raise ValueError("Unterminated prediction row; use the finalized run ledger")
    failure_bytes = failures_file.read_bytes() if failures_file else b""
    recorded = json.loads((BATCH / "statistics.json").read_text(encoding="utf-8"))
    if recorded["predictions_sha256"] != FROZEN_PREDICTIONS_SHA256:
        raise ValueError("statistics.json no longer records the archived reply ledger fingerprint")
    frozen = sha256_bytes(raw) == FROZEN_PREDICTIONS_SHA256 and failures_file is None
    if not frozen and not diagnostic:
        raise ValueError(
            "The reply ledger differs from the archived confirmation_1124 ledger "
            f"(SHA-256 {FROZEN_PREDICTIONS_SHA256}) or adds technical failures; "
            "pass --diagnostic-ledger to inspect it as a diagnostic, not as this batch's result"
        )
    provenance = {
        "frozen_input_commit": summarize.FROZEN_COMMIT,
        "text_input_sha256": fingerprints,
        "ledger": "frozen_confirmation_1124" if frozen else "diagnostic_custom_ledger",
        "frozen_predictions_sha256": FROZEN_PREDICTIONS_SHA256,
        "predictions_sha256": sha256_bytes(raw),
        "technical_failures_sha256": sha256_bytes(failure_bytes) if failures_file else None,
    }
    return {
        "selected": summarize.read_jsonl(selection_file.read_bytes()),
        "prompts": summarize.read_jsonl(prompt_file.read_bytes()),
        "predictions": summarize.read_jsonl(raw),
        "failures": summarize.read_jsonl(failure_bytes),
        "provenance": provenance,
    }


def field_statuses(row):
    """Return the (answer, event) field status of one decoded key.

    A received reply with a well-formed JSON object keeps each field's own
    status, so a valid action survives an invalid or conflicting event and
    vice versa. Format errors and technical failures affect both fields.
    """
    if row["status"] in ("technical_missing", "format_error"):
        return row["status"], row["status"]
    return FIELD_STATUS[row["answer_status"]], FIELD_STATUS[row["event_status"]]


def field_counts(rows):
    counts = {"answer": Counter(), "event": Counter()}
    for row in rows:
        answer, event = field_statuses(row)
        counts["answer"][answer] += 1
        counts["event"][event] += 1
    return {field: {name: counter[name] for name in FIELD_STATUSES} for field, counter in counts.items()}


def decode_batch(batch):
    options, qids, precheck = check_batch(batch["selected"], batch["prompts"], batch["predictions"])
    parser, parser_sha = summarize.load_parser()
    decoded = summarize.decode_rows(
        batch["prompts"], batch["predictions"], batch["failures"], parser.parse_reply, parser.interval_label,
    )
    precheck["technical_failures"] = len(batch["failures"])
    precheck["strict_rows_verified_against_inference"] = len(batch["predictions"])
    precheck["reply_statuses_all_keys"] = {
        rule: {name: sum(row["status"] == name for row in rows) for name in STATUSES} for rule, rows in decoded.items()
    }
    precheck["field_statuses_all_keys"] = {
        rule: {condition: field_counts(row for row in rows if row["condition"] == condition) for condition in CONDITIONS}
        for rule, rows in decoded.items()
    }
    indexed = {rule: {summarize.row_key(row): row for row in rows} for rule, rows in decoded.items()}
    parser_info = {
        "protocol_parser_commit": summarize.PROTOCOL_PARSER_COMMIT,
        "parser_source_commit": summarize.PARSER_SOURCE_COMMIT,
        "parser_source_sha256": parser_sha,
        "primary_rule": RULES[0],
        "sensitivity_rule": RULES[1],
        "reused_code_sha256": {
            path: sha256_bytes((STUDY / path).read_bytes())
            for path in ("eval/score.py", "eval/freeze_reviews.py", "run/confirmation_1124/summarize.py")
        },
    }
    return options, qids, precheck, indexed, parser_info


# ------------------------------------------------------------------ human gold

def load_gold_bundle(gold_dir):
    """Read gold.jsonl, questions.jsonl and freeze.json written by eval/freeze_reviews.py."""
    files = {name: (gold_dir / name).read_bytes() for name in ("gold.jsonl", "questions.jsonl", "freeze.json")}
    freeze = json.loads(files["freeze.json"])
    if freeze.get("gold_sha256") != sha256_bytes(files["gold.jsonl"]):
        raise ValueError("gold.jsonl differs from the fingerprint in freeze.json")
    if freeze.get("questions_sha256") != sha256_bytes(files["questions.jsonl"]):
        raise ValueError("questions.jsonl differs from the fingerprint in freeze.json")
    references = score.load_gold(gold_dir / "gold.jsonl")
    rows = {}
    for row in summarize.read_jsonl(files["gold.jsonl"]):
        if row["answer_id"] != gold_action_id(row["answer_text"]):
            raise ValueError(f"{row['qid']}: answer_id does not match its canonical action text")
        rows[row["pair_id"], row["occurrence"]] = row
    decisions = {}
    for item in freeze["decisions"]:
        if item["pair_id"] in decisions or item["decision"] not in DECISIONS:
            raise ValueError(f"freeze.json has a duplicate or unknown decision for {item['pair_id']}")
        decisions[item["pair_id"]] = item
    gold_pairs = {pair_id for pair_id, _ in rows}
    included = {pair_id for pair_id, item in decisions.items() if item["decision"] == "纳入"}
    if gold_pairs != included:
        raise ValueError("gold.jsonl must contain exactly the pairs adjudicated as 纳入")
    fingerprint = sha256_bytes(b"".join(files[name] for name in ("freeze.json", "gold.jsonl", "questions.jsonl")))
    return {
        "references": {pair[0].pair_id: pair for pair in references},
        "rows": rows,
        "decisions": decisions,
        "fingerprint": fingerprint,
        "summary": {
            "files_sha256": {name: sha256_bytes(raw) for name, raw in files.items()},
            "fingerprint": fingerprint,
            "adjudicator": freeze.get("adjudicator"),
            "frozen_at": freeze.get("frozen_at"),
            "queue_sha256": freeze.get("queue_sha256"),
            "pairs_included": freeze.get("pairs_included"),
        },
    }


def included_pairs(selected, gold):
    for pair in selected:
        decision = gold["decisions"].get(pair["pair_id"])
        if decision and decision["decision"] == "纳入":
            yield pair


def gold_answers(gold, pair):
    answers = []
    for occurrence in (1, 2):
        row = gold["rows"][pair["pair_id"], occurrence]
        if row["video_id"] != pair["video_id"]:
            raise ValueError(f"{pair['pair_id']}: gold video {row['video_id']} differs from {pair['video_id']}")
        answers.append(row)
    return answers


def automatic_target(text, answers):
    matches = [index for index, row in enumerate(answers, 1) if comparable_action(row["answer_text"]) == comparable_action(text)]
    return matches[0] if len(matches) == 1 else None


def template_rows(selected, options, gold):
    rows = []
    for pair in included_pairs(selected, gold):
        answers = gold_answers(gold, pair)
        for option_id, option in sorted(options[pair["pair_id"]].items(), key=lambda item: item[1]["source_ordinal"]):
            target = automatic_target(option["text"], answers)
            rows.append({
                "题对": pair["pair_id"], "视频": pair["video_id"], "金标指纹": gold["fingerprint"],
                "原选项ID": option_id, "原选项文本": option["text"], "原选项来源序数": str(option["source_ordinal"]),
                "金标动作1": answers[0]["answer_text"], "金标动作2": answers[1]["answer_text"],
                "自动判定": f"规范化文字同金标{target}" if target else "待人工映射",
                "映射": "", "映射人": "", "理由": "",
            })
    return rows


def write_template(path, rows):
    with path.open("x", encoding="utf-8-sig", newline="") as stream:
        table = csv.DictWriter(stream, MAPPING_COLUMNS, dialect="excel-tab")
        table.writeheader()
        table.writerows(rows)


def load_mapping(path, expected_rows):
    raw = path.read_bytes()
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, dialect="excel-tab")
        if reader.fieldnames != MAPPING_COLUMNS:
            raise ValueError("Keep the mapping table's original columns and tab separators")
        rows = list(reader)
    expected = {(row["题对"], row["原选项ID"]): row for row in expected_rows}
    keys = [(row["题对"], row["原选项ID"]) for row in rows]
    duplicates = sorted(key for key, count in Counter(keys).items() if count > 1)
    if duplicates or set(keys) != set(expected):
        raise ValueError(
            "The mapping table must list every original option of every included pair exactly once; "
            f"duplicated {duplicates}, missing {sorted(set(expected) - set(keys))}, "
            f"unexpected {sorted(set(keys) - set(expected))}"
        )
    mapping = {}
    for row in rows:
        key = (row["题对"], row["原选项ID"])
        if None in row or None in row.values():
            raise ValueError(f"Mapping row {key} is malformed")
        if any(row[name] != expected[key][name] for name in IDENTITY_COLUMNS):
            raise ValueError(f"Mapping row {key} does not match the current gold and original options")
        value = row["映射"].strip()
        if not value:
            continue
        if value not in MAPPING_VALUES:
            raise ValueError(f"Mapping row {key}: 映射 must be 金标1, 金标2, 无对应 or empty")
        reviewer, reason = row["映射人"].strip(), row["理由"].strip()
        if not reviewer or not reason:
            raise ValueError(f"Mapping row {key}: a human mapping needs 映射人 and 理由")
        mapping[key] = {"target": MAPPING_VALUES[value], "reviewer": reviewer, "reason": reason}
    return mapping, sha256_bytes(raw)


# ------------------------------------------------------------------- alignment

def resolve_pair(pair, options, gold, mapping):
    """Return the alignment record of one batch pair."""
    pair_id = pair["pair_id"]
    record = {"pair_id": pair_id, "video_id": pair["video_id"]}
    decision = gold["decisions"].get(pair_id) if gold else None
    if decision is None:
        return dict(record, status="no_gold", adjudication=None)
    record["adjudication"] = {"decision": decision["decision"], "reason": decision["reason"]}
    if decision["decision"] != "纳入":
        return dict(record, status=DECISIONS[decision["decision"]])
    answers = gold_answers(gold, pair)
    resolved = []
    for option_id, option in sorted(options[pair_id].items(), key=lambda item: item[1]["source_ordinal"]):
        automatic = automatic_target(option["text"], answers)
        explicit = mapping.get((pair_id, option_id))
        if explicit and automatic and explicit["target"] != automatic:
            raise ValueError(f"{pair_id} {option_id}: human mapping contradicts identical normalized text")
        if explicit:
            target, basis = explicit["target"], dict(explicit, kind="human_mapping")
        elif automatic:
            target, basis = automatic, {"kind": "normalized_text_identical"}
        else:
            target, basis = None, {"kind": "pending_human_mapping"}
        resolved.append(dict(option, original_option_id=option_id, gold_occurrence=target, basis=basis))
    targets = [item["gold_occurrence"] for item in resolved if item["gold_occurrence"] in (1, 2)]
    if len(targets) != len(set(targets)):
        raise ValueError(f"{pair_id}: two original options cannot map to the same gold action")
    pending = any(item["gold_occurrence"] is None for item in resolved)
    windows = pair_windows(pair)
    questions = []
    for row in answers:
        occurrence = row["occurrence"]
        if occurrence in targets:
            mapping_status = "aligned"
        elif pending:
            mapping_status = "mapping_pending"
        else:
            mapping_status = "answer_not_in_original_options"
        questions.append({
            "gold_qid": row["qid"], "original_qid": None, "ordinal": occurrence,
            "gold_answer_id": row["answer_id"], "gold_answer_text": row["answer_text"],
            "adjudicated_interval_s": row["interval_s"], "adjudicated_action_interval_s": row["action_interval_s"],
            "adjudicated_interval_iou_with_windows": [score.iou(row["interval_s"], window) for window in windows],
            "answer_mapping": mapping_status,
        })
    if pending:
        status = "mapping_pending"
    elif any(item["answer_mapping"] != "aligned" for item in questions):
        status = "answer_not_in_original_options"
    else:
        status = "aligned"
    return dict(record, status=status, original_options=resolved, questions=questions)


def pair_windows(pair):
    return [item["sound_s"] for item in sorted(pair["occurrences"], key=lambda item: item["ordinal"])]


def gold_answer_for(option, answers):
    target = option["gold_occurrence"]
    return answers[target - 1]["answer_id"] if target in (1, 2) else NOT_IN_GOLD + option["original_option_id"]


def score_rule(records, gold, decoded, qids):
    aligned = [record for record in records if record["status"] == "aligned"]
    references = [gold["references"][record["pair_id"]] for record in aligned]
    conditions = {}
    for condition in CONDITIONS:
        predictions, statuses, rows = {}, Counter(), []
        for record in aligned:
            answers = {option["original_option_id"]: option for option in record["original_options"]}
            gold_rows = [gold["rows"][record["pair_id"], occurrence] for occurrence in (1, 2)]
            for question in record["questions"]:
                row = decoded[qids[record["pair_id"], question["ordinal"]], condition]
                rows.append(row)
                statuses[row["status"]] += 1
                answer_field, event_field = field_statuses(row)
                if answer_field == "technical_missing":
                    continue  # the only case score.py sees as a missing prediction
                if answer_field == "valid":
                    answer = gold_answer_for(answers[row["answer_id"]], gold_rows)
                elif answer_field == "explicit_null":
                    answer = None  # score.py counts this as an abstention
                else:
                    answer = UNUSABLE_ANSWER + answer_field  # received but not a valid choice: wrong, not abstained
                interval = tuple(row["interval_s"]) if event_field == "valid" else None
                predictions[record["pair_id"], question["gold_qid"]] = score.Prediction(answer, interval)
        result = score.score_group(references, predictions)
        conditions[condition] = dict(result, reply_statuses={
            "planned_questions": 2 * len(aligned),
            "received_replies": len(rows) - statuses["technical_missing"],
            "technical_missing": statuses["technical_missing"],
            "statuses": {name: statuses[name] for name in STATUSES},
            "fields": field_counts(rows),
        })
    if not aligned:
        status = "未跑"
    else:
        status = "已评分" if all(item["status"] == "已评分" for item in conditions.values()) else "部分完成"
    return {"status": status, "conditions": conditions, "comparisons": score.compare_groups(conditions)}


def build_report(batch, gold=None, mapping_file=None):
    options, qids, precheck, decoded, parser_info = decode_batch(batch)
    selected = batch["selected"]
    mapping, mapping_sha = {}, None
    if mapping_file is not None:
        if gold is None:
            raise ValueError("A mapping table needs the frozen gold it was generated from")
        mapping, mapping_sha = load_mapping(mapping_file, template_rows(selected, options, gold))
    records = [resolve_pair(pair, options, gold, mapping) for pair in selected]
    for record in records:
        for question in record.get("questions", []):
            question["original_qid"] = qids[record["pair_id"], question["ordinal"]]
    counts = Counter(record["status"] for record in records)
    question_counts, aligned_outside = Counter(), Counter()
    for record in records:
        if "questions" not in record:
            question_counts[record["status"]] += 2
            continue
        for question in record["questions"]:
            question_counts[question["answer_mapping"]] += 1
            if question["answer_mapping"] == "aligned" and record["status"] != "aligned":
                aligned_outside[record["status"]] += 1
    gold_summary = None
    if gold is not None:
        batch_pairs = {pair["pair_id"] for pair in selected}
        selection_sha = batch.get("provenance", {}).get("text_input_sha256", {}).get("manifest/selected.jsonl")
        gold_summary = dict(
            gold["summary"],
            queue_is_this_batch_selection=gold["summary"]["queue_sha256"] == selection_sha,
            pairs_outside_batch=sorted(set(gold["decisions"]) - batch_pairs),
        )
    ledger = batch.get("provenance", {}).get("ledger")
    report = {
        "batch": "confirmation_1124",
        "result_scope": (
            "frozen confirmation_1124 ledger" if ledger == "frozen_confirmation_1124"
            else "diagnostic only: not a frozen confirmation_1124 result"
        ),
        "status": "未跑" if counts["aligned"] == 0 else None,
        "inputs": dict(batch.get("provenance", {}), **parser_info),
        "key_precheck": precheck,
        "gold": gold_summary,
        "mapping_sha256": mapping_sha,
        "denominators": {
            "planned_pairs": len(records),
            "planned_questions": 2 * len(records),
            "pairs": {name: counts[name] for name in PAIR_STATUSES},
            "questions": {name: question_counts[name] for name in PAIR_STATUSES},
            "scored_pairs": counts["aligned"],
            "scored_questions": 2 * counts["aligned"],
            "aligned_questions_in_unscored_pairs": {
                name: aligned_outside[name] for name in ("mapping_pending", "answer_not_in_original_options")
            },
        },
        "answer_not_in_original_options": [
            dict(question, pair_id=record["pair_id"], original_options=[
                {key: option[key] for key in ("original_option_id", "text", "gold_occurrence", "basis")}
                for option in record["original_options"]
            ])
            for record in records for question in record.get("questions", [])
            if question["answer_mapping"] == "answer_not_in_original_options"
        ],
        "pairs": records,
        "scoring": {
            "scorer": "eval/score.py score_group and compare_groups",
            "iou_threshold": score.IOU_THRESHOLD,
            "rules": {rule: score_rule(records, gold, decoded[rule], qids) for rule in RULES},
        },
    }
    if report["status"] is None:
        report["status"] = report["scoring"]["rules"][RULES[0]]["status"]
    return report


# ------------------------------------------------------------------------- CLI

def main():
    cli = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = cli.add_subparsers(dest="command", required=True)
    template = commands.add_parser("template", help="Write the action-mapping table for the included pairs")
    template.add_argument("--gold-dir", type=Path, required=True, help="Output directory of freeze_reviews.py freeze")
    template.add_argument("--output", type=Path, required=True, help="New UTF-8 tab-separated file")
    align = commands.add_parser("align", help="Write the alignment and score report; without gold only keys and inputs are checked")
    align.add_argument("--gold-dir", type=Path, help="Output directory of freeze_reviews.py freeze")
    align.add_argument("--mapping", type=Path, help="Filled mapping table from the template command")
    align.add_argument("--output", type=Path, required=True, help="New JSON report")
    for command in (template, align):
        command.add_argument("--predictions", type=Path, default=BATCH / "predictions.jsonl")
        command.add_argument("--technical-failures", type=Path, help="Rows with qid, condition, error and log_reference")
        command.add_argument(
            "--diagnostic-ledger", action="store_true",
            help="Accept a ledger other than the archived one; the report is labelled diagnostic only",
        )
    args = cli.parse_args()
    try:
        batch = load_frozen_batch(args.predictions, args.technical_failures, args.diagnostic_ledger)
        gold = load_gold_bundle(args.gold_dir) if args.gold_dir else None
        if args.command == "template":
            options, _, _ = check_batch(batch["selected"], batch["prompts"], batch["predictions"])
            rows = template_rows(batch["selected"], options, gold)
            write_template(args.output, rows)
            pending = sum(row["自动判定"] == "待人工映射" for row in rows)
            print(f"Wrote {len(rows)} original options ({pending} pending human mapping) to {args.output}")
            return
        report = build_report(batch, gold, args.mapping)
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        pairs = report["denominators"]["pairs"]
        print(f"{report['status']} ({report['result_scope']}): {pairs['aligned']} aligned of "
              f"{report['denominators']['planned_pairs']} pairs. Wrote {args.output}")
    except (ValueError, OSError, KeyError, TypeError) as exc:
        cli.error(str(exc))


if __name__ == "__main__":
    main()
