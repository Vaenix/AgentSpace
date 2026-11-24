"""Synthetic swapped replies against the archived original extension inputs."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import summarize_swap as swap


class FrameSwapReplay(unittest.TestCase):
    def test_paired_denominators_failure_and_strict_sensitivity(self):
        plan, _ = swap.load_plan()
        archived = {row["qid"]: row for row in swap.frozen.read_jsonl(
            (swap.ARCHIVE / "replies.jsonl").read_bytes()) if row["condition"] == "original"}
        generated = {
            "P06_102-5-smoke-1": {"answer": "B", "event": "E02 333.326–334.514s"},
            "P07_04-7-smoke-1": {"answer": "B", "event": "E02"},
            "P13_08-2-smoke-1": {"answer": "A", "event": "E01"},
        }
        failed = "P06_102-5-smoke-2"
        replies = []
        attempts = [{"event": "run_started", "plan_sha256": swap.PLAN_SHA256}]
        for row in plan:
            key = {name: row[name] for name in ("qid", "video_id", "ordinal", "source_run_order", "condition")}
            if row["status"] == "technical_missing":
                attempts.append(dict(key, event="frozen_technical_missing", stage=row["stage"], reason=row["reason"]))
                continue
            attempts.append(dict(key, event="started"))
            if row["qid"] == failed:
                attempts.append(dict(key, event="failed", stage="inference", reason="synthetic failure"))
                continue
            qid = row["qid"]
            raw = generated.get(qid, {"answer": None, "event": None})
            replies.append(dict(key, raw_response=json.dumps(raw), prompt_sha256=row["prompt_sha256"],
                                frames_sha256=row["frames_sha256"], generated_tokens=9,
                                prompt_tokens=archived[qid]["prompt_tokens"] + int(qid == "P06_102-5-smoke-1")))
            attempts.append(dict(key, event="generated"))
        attempts.append({"event": "run_finished"})
        with TemporaryDirectory() as folder:
            rp = Path(folder) / "replies.jsonl"
            ap = Path(folder) / "attempts.jsonl"
            rp.write_text("".join(json.dumps(row) + "\n" for row in replies))
            ap.write_text("".join(json.dumps(row) + "\n" for row in attempts))
            report = swap.replay(rp, ap)
            with self.assertRaisesRegex(ValueError, "requires an attempts ledger"):
                swap.replay(rp, None)
        rules = report["rules"]
        primary = rules["matched_seconds"]
        paired = primary["comparison"]
        self.assertEqual(len(report["planned_input_missing_questions"]), 2)
        self.assertEqual(len(report["input_tokens"]["different"]), 1)
        self.assertEqual((paired["planned_questions"], paired["paired_returned_texts"]), (24, 21))
        self.assertEqual(paired["paired_completeness"], {
            "both_complete": 2, "original_only_complete": 12, "swapped_only_complete": 1, "neither_complete": 6,
        })
        self.assertEqual(paired["changed_displayed_interval"], swap.count(1, 2, 24))
        self.assertEqual(paired["retained_image_group_with_stable_action"], swap.count(1, 1, 24))
        strict = rules["strict"]["comparison"]
        self.assertEqual(strict["changed_displayed_interval"], swap.count(0, 1, 24))
        self.assertIsNone(strict["retained_image_group_with_stable_action"]["proportion"])
        missing = next(q for q in primary["questions"] if q["qid"] == failed)
        self.assertEqual(missing["swapped_technical_missing"]["reason"], "synthetic failure")
        self.assertEqual(primary["recordings"]["P11_104"]["comparable_questions"], 0)


if __name__ == "__main__":
    unittest.main()
