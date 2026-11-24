"""Offline checks for the joint-reversal condition; no media, GPU or model needed."""
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import crossed_inputs  # noqa: E402
from crossed_inputs import frozen  # noqa: E402
import run_crossed  # noqa: E402
import summarize_crossed  # noqa: E402

HERE = crossed_inputs.HERE
FRAMES_ROOT = Path(os.environ.get("CROSSED_FRAMES_ROOT", "/private/tmp/av-extension-freeze-314da9/frozen-rebased"))
PARSER = summarize_crossed.archived.load_parser()
INPUTS = crossed_inputs.load()
KEYS = INPUTS["keys"]
RUNNABLE = [key for key in KEYS if key["status"] == "runnable"]
ARCHIVED = {frozen.key_of(call): call for call in INPUTS["archived_calls"]}


class InputTest(unittest.TestCase):
    def test_planned_keys_follow_the_frozen_queue(self):
        self.assertEqual(len(KEYS), 24)
        self.assertEqual(len(RUNNABLE), 22)
        self.assertEqual({key["condition"] for key in KEYS}, {"both_reversed"})
        self.assertEqual([key["run_order"] for key in KEYS], list(range(1, 25)))
        self.assertEqual([key["qid"] for key in KEYS if key["status"] == "technical_missing"],
                         ["P11_104-7-smoke-1", "P11_104-7-smoke-2"])
        queue = [call["qid"] for call in INPUTS["archived_calls"] if call["condition"] == "original"]
        self.assertEqual([key["qid"] for key in KEYS], queue)

    def test_joint_prompt_takes_answer_choices_and_evidence_codes_from_the_single_reversals(self):
        for key in RUNNABLE:
            answers = ARCHIVED[key["qid"], "options_reversed"]
            codes = ARCHIVED[key["qid"], "segment_labels_reversed"]
            self.assertEqual(key["option_codes"], answers["option_codes"])
            self.assertEqual(key["segment_codes"], codes["segment_codes"])
            self.assertEqual(key["frames_sha256"], answers["frames_sha256"])
            original = ARCHIVED[key["qid"], "original"]
            self.assertNotEqual(key["option_codes"], original["option_codes"])
            self.assertNotEqual(key["segment_codes"], original["segment_codes"])

    def test_input_check_report_is_reproduced(self):
        text = json.dumps(crossed_inputs.report(INPUTS), indent=2, ensure_ascii=False) + "\n"
        self.assertEqual(text, (HERE / "input_check.json").read_text(encoding="utf-8"))

    def test_construction_paths_that_disagree_are_refused(self):
        key = RUNNABLE[0]
        answers = copy.deepcopy(ARCHIVED[key["qid"], "options_reversed"]["prompt"])
        codes = copy.deepcopy(ARCHIVED[key["qid"], "segment_labels_reversed"]["prompt"])
        self.assertEqual(crossed_inputs.construct(answers, codes)["prompt"], key["prompt"]["prompt"])
        moved = copy.deepcopy(codes)
        moved["physical_window_order"] = moved["physical_window_order"][::-1]
        with self.assertRaises(ValueError):
            crossed_inputs.construct(answers, moved)
        reordered = copy.deepcopy(codes)
        reordered["frames"] = reordered["frames"][::-1]
        with self.assertRaises(ValueError):
            crossed_inputs.construct(answers, reordered)
        restemmed = copy.deepcopy(codes)
        restemmed["prompt"] = restemmed["prompt"].replace("first sound", "second sound")
        with self.assertRaises(ValueError):
            crossed_inputs.construct(answers, restemmed)

    def test_maps_must_match_the_prompt_text(self):
        prompt = RUNNABLE[0]["prompt"]
        crossed_inputs.check_maps(prompt["prompt"], prompt["option_codes"], prompt["segment_codes"],
                                  prompt["semantic_actions"])
        swapped_options = dict(zip(prompt["option_codes"], reversed(list(prompt["option_codes"].values()))))
        with self.assertRaises(ValueError):
            crossed_inputs.check_maps(prompt["prompt"], swapped_options, prompt["segment_codes"],
                                      prompt["semantic_actions"])
        swapped_codes = dict(zip(prompt["segment_codes"], reversed(list(prompt["segment_codes"].values()))))
        with self.assertRaises(ValueError):
            crossed_inputs.check_maps(prompt["prompt"], prompt["option_codes"], swapped_codes,
                                      prompt["semantic_actions"])

    def test_frames_must_match_hash_and_size(self):
        with tempfile.TemporaryDirectory() as folder, self.assertRaises(FileNotFoundError):
            crossed_inputs.check_frames(Path(folder), RUNNABLE[:1])


def joint_reply(key, raw):
    prompt = key["prompt"]
    answer, interval, result_format = PARSER.decode(raw, key["option_codes"], key["segment_codes"])
    return {
        **{name: key[name] for name in run_crossed.LEDGER_IDENTITY},
        "prompt_sha256": key["prompt_sha256"], "frames_sha256": key["frames_sha256"], "answer_id": answer,
        "action": prompt["semantic_actions"].get(answer), "interval_s": interval,
        "physical_window": None if interval is None else prompt["physical_window_order"].index(interval) + 1,
        "result_format": result_format, "raw_response": raw, "prompt_tokens": 1830, "generated_tokens": 19,
        "archived_prompt_tokens": {}, "run_id": "test",
    }


def null_reply(key):
    return joint_reply(key, json.dumps({"answer": None, "event": None}))


def write_ledgers(folder, replies, attempts=()):
    replies_file, attempts_file = Path(folder) / "replies.jsonl", Path(folder) / "attempts.jsonl"
    replies_file.write_text("".join(json.dumps(row) + "\n" for row in replies), encoding="utf-8")
    attempts_file.write_text("".join(json.dumps(row) + "\n" for row in attempts), encoding="utf-8")
    return replies_file, attempts_file


REPORT = summarize_crossed.replay(HERE / "replies.jsonl", HERE / "attempts.jsonl")


class ReplayTest(unittest.TestCase):
    def replay(self, replies, attempts=()):
        with tempfile.TemporaryDirectory() as folder:
            return summarize_crossed.replay(*write_ledgers(folder, replies, attempts))

    def test_archived_statistics_replay_exactly(self):
        text = json.dumps(REPORT, indent=2, ensure_ascii=False) + "\n"
        self.assertEqual(text, (HERE / "statistics.json").read_text(encoding="utf-8"))

    def test_tables_sum_to_their_denominators(self):
        for analysis in REPORT["rules"].values():
            for counts in analysis["conditions"].values():
                self.assertEqual(sum(counts["statuses"].values()), 24)
            primary = analysis["primary"]
            self.assertEqual(sum(primary["cells"].values()), primary["common_denominator"])
            self.assertEqual(primary["common_denominator"], analysis["coverage"]["all_four_complete"])
            self.assertEqual(primary["common_denominator"] + analysis["non_comparable_questions"]["count"], 24)
            self.assertEqual(sum(row["count"] for row in analysis["non_comparable_questions"]["status_patterns"]),
                             analysis["non_comparable_questions"]["count"])
            for edge in analysis["edges"].values():
                self.assertEqual(sum(edge["cells"].values()), edge["pairwise_complete"])
                self.assertEqual(edge["pairwise_complete"] + sum(row["count"] for row in
                                                                  edge["non_comparable_status_pairs"]), 24)
            self.assertEqual(sum(analysis["recordings"]["classes"].values()), 12)

    def test_original_edges_reproduce_the_archived_single_factor_tables(self):
        archived = json.loads((crossed_inputs.EXTENSION_RESULTS / "statistics.json").read_text(encoding="utf-8"))
        pairs = {"answer_order_with_original_codes": "options_reversed",
                 "evidence_codes_with_original_answers": "segment_labels_reversed"}
        for rule, analysis in REPORT["rules"].items():
            for condition in crossed_inputs.CONDITIONS[:3]:
                self.assertEqual(analysis["conditions"][condition],
                                 archived["rules"][rule]["replies_by_condition"][condition])
            for edge, contrast in pairs.items():
                old = archived["rules"][rule]["comparisons"][contrast]["questions"]
                self.assertEqual(analysis["edges"][edge]["cells"], old["cells"])
                self.assertEqual(analysis["edges"][edge]["pairwise_complete"], old["comparable_questions"])

    def test_primary_classes_follow_the_two_answer_order_edges(self):
        for question in REPORT["rules"]["matched_seconds"]["questions"]:
            if not question["all_four_complete"]:
                self.assertIsNone(question["primary_class"])
                continue
            original = question["edges"]["answer_order_with_original_codes"]["option_swap_event"]
            reversed_ = question["edges"]["answer_order_with_reversed_codes"]["option_swap_event"]
            expected = {(False, False): "neither", (True, False): "original_labels_only",
                        (False, True): "reversed_labels_only", (True, True): "both"}[original, reversed_]
            self.assertEqual(question["primary_class"], expected)

    def test_all_null_joint_replies_leave_no_common_question(self):
        report = self.replay([null_reply(key) for key in RUNNABLE])
        primary = report["rules"]["matched_seconds"]["primary"]
        self.assertEqual(primary["common_denominator"], 0)
        self.assertIsNone(primary["different_event_status_per_common"])
        self.assertEqual(report["rules"]["matched_seconds"]["conditions"]["both_reversed"]["statuses"]
                         ["both_abstained"], 22)

    def test_failed_key_is_technical_missing_not_abstention(self):
        dropped = RUNNABLE[0]
        identity = {name: dropped[name] for name in run_crossed.LEDGER_IDENTITY}
        attempts = [{"run_id": "r", "event": "started", **identity},
                    {"run_id": "r", "event": "failed", "error": "RuntimeError: test", **identity}]
        report = self.replay([null_reply(key) for key in RUNNABLE[1:]], attempts)
        analysis = report["rules"]["matched_seconds"]
        self.assertEqual(analysis["conditions"]["both_reversed"]["received"], 21)
        self.assertEqual(analysis["technical_missing"]["by_condition"]["both_reversed"], 3)
        stages = [row["stage"] for row in analysis["technical_missing"]["slots"] if row["condition"] == "both_reversed"]
        self.assertEqual(stages, ["frame_extraction", "frame_extraction", "generation"])

    def test_missing_reply_without_logged_attempt_is_refused(self):
        with self.assertRaises(ValueError):
            self.replay([null_reply(key) for key in RUNNABLE[1:]])

    def test_tampered_duplicate_or_reimaged_rows_are_refused(self):
        rows = [null_reply(key) for key in RUNNABLE]
        with self.assertRaises(ValueError):
            self.replay(rows + rows[:1])
        tampered = [dict(row) for row in rows]
        tampered[0]["result_format"] = "invalid_json"
        with self.assertRaises(ValueError):
            self.replay(tampered)
        reimaged = [dict(row) for row in rows]
        reimaged[0]["frames_sha256"] = reimaged[0]["frames_sha256"][::-1]
        with self.assertRaises(ValueError):
            self.replay(reimaged)


@unittest.skipUnless(FRAMES_ROOT.is_dir(), "set CROSSED_FRAMES_ROOT to the extension's frozen frames directory")
class RunnerTest(unittest.TestCase):
    def test_stored_replies_are_reused_without_loading_the_model(self):
        with tempfile.TemporaryDirectory() as folder:
            shutil.copy(HERE / "replies.jsonl", folder)
            shutil.copy(HERE / "attempts.jsonl", folder)
            before = (Path(folder) / "replies.jsonl").read_bytes()
            result = subprocess.run(
                [sys.executable, str(HERE / "run_crossed.py"), "--model", folder, "--frames-root", str(FRAMES_ROOT),
                 "--results-dir", folder], capture_output=True, text=True, check=True)
            counts = json.loads(result.stdout.strip().splitlines()[-1])
            self.assertEqual((counts["generated"], counts["reused"], counts["frozen_technical_missing"]), (0, 22, 2))
            self.assertEqual((Path(folder) / "replies.jsonl").read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
