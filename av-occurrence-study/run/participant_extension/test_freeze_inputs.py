"""Offline consistency checks of the committed frozen inputs; no media, frames or model needed."""
import csv
import hashlib
import json
import re
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE)]
from infer_mlx import parse_reply, seconds_label  # noqa: E402
import freeze_inputs  # noqa: E402

FROZEN = HERE / "frozen_inputs"
SWAP = {"E01": "E02", "E02": "E01"}


def read(name):
    return [json.loads(line) for line in (FROZEN / name).read_text(encoding="utf-8").splitlines() if line.strip()]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class FrozenInputs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.calls = read("calls.jsonl")
        cls.prompts = read("prompts.jsonl")
        cls.hashes = json.loads((FROZEN / "input_hashes.json").read_text())
        cls.provenance = json.loads((FROZEN / "provenance.json").read_text())
        cls.windows = freeze_inputs.replicate.read(freeze_inputs.WINDOWS)

    def test_call_keys_follow_planned_calls(self):
        self.assertEqual(sha(freeze_inputs.PLANNED_CALLS), freeze_inputs.PLANNED_CALLS_SHA)
        with freeze_inputs.PLANNED_CALLS.open(newline="", encoding="utf-8") as handle:
            plan = list(csv.DictReader(handle))
        keys = ("run_order", "extension_rank", "participant_id", "video_id", "pair_id", "qid", "ordinal", "condition")
        self.assertEqual([{key: str(call[key]) for key in keys} for call in self.calls], plan)

    def test_counts_and_missing_reason(self):
        runnable = [call for call in self.calls if call["status"] == "runnable"]
        missing = [call for call in self.calls if call["status"] == "technical_missing"]
        self.assertEqual((len(self.calls), len(runnable), len(missing)), (72, 66, 6))
        self.assertEqual({call["pair_id"] for call in missing}, {"P11_104-7"})
        self.assertEqual([call["run_order"] for call in missing], list(range(1, 7)))
        for call in missing:
            self.assertIsNone(call["prompt_line"])
            self.assertIn("action_75pct at clip 9.525500 s", call["reason"])
            self.assertIn("last video frame starts at 9.400 s", call["reason"])
        unavailable = self.provenance["unavailable"]
        self.assertEqual([item["pair_id"] for item in unavailable], ["P11_104-7"])
        self.assertEqual(unavailable[0]["available_frames"], 5)
        self.assertEqual(self.provenance["counts"]["runnable"], 66)

    def test_manifest_fingerprints(self):
        self.assertEqual(sha(FROZEN / "prompts.jsonl"), self.hashes["prompts_sha256"])
        for name, digest in self.provenance["manifest_sha256"].items():
            self.assertEqual(sha(FROZEN / name), digest, name)
        for name, digest in self.provenance["frame_manifest_sha256"].items():
            self.assertEqual(sha(FROZEN / name), digest, name)
        self.assertEqual(sha(freeze_inputs.WINDOWS), self.provenance["media_prep_windows_sha256"])
        self.assertEqual(self.hashes["runner_sha256"], sha(HERE.parent / "replication_1124/replicate.py"))
        self.assertEqual(sha(HERE.parent / "infer_mlx.py"), self.provenance["parser"]["sha256"])

    def test_frame_hashes_match_media_preparation(self):
        recorded = {
            f"frames/{row['pair_id']}/{frame['file']}": frame["sha256"]
            for row in self.windows for frame in row.get("frames", [])
            if frame["status"] == "extracted" and row["smoke_input_sha256"]
        }
        self.assertEqual(self.hashes["frame_sha256"], recorded)
        self.assertEqual(len(recorded), 66)

    def test_calls_point_at_their_prompts(self):
        for call in self.calls:
            if call["status"] != "runnable":
                continue
            row = self.prompts[call["prompt_line"] - 1]
            self.assertEqual((row["qid"], row["condition"]), (call["qid"], call["condition"]))
            self.assertEqual(hashlib.sha256(row["prompt"].encode()).hexdigest(), call["prompt_sha256"])
            self.assertEqual([self.hashes["frame_sha256"][path] for path in row["frames"]], call["frames_sha256"])
            self.assertEqual((row["option_codes"], row["segment_codes"]), (call["option_codes"], call["segment_codes"]))

    def test_conditions_change_only_their_mapping(self):
        for index in range(0, len(self.prompts), 3):
            original, options, labels = self.prompts[index:index + 3]
            self.assertEqual([row["condition"] for row in (original, options, labels)],
                             list(freeze_inputs.replicate.CONDITIONS))
            for row in (options, labels):
                self.assertEqual(row["frames"], original["frames"])
                self.assertEqual(row["physical_window_order"], original["physical_window_order"])
            self.assertEqual(options["option_codes"], {"A": original["option_codes"]["B"], "B": original["option_codes"]["A"]})
            self.assertEqual(options["segment_codes"], original["segment_codes"])
            self.assertEqual(labels["option_codes"], original["option_codes"])
            self.assertEqual(labels["segment_codes"], {SWAP[code]: interval for code, interval in original["segment_codes"].items()})
            self.assertEqual(labels["prompt"], re.sub(r"(?m)^E0[12](?= )", lambda m: SWAP[m[0]], original["prompt"]))

    def test_pinned_parsers_decode_every_mapping(self):
        for row in self.prompts:
            for letter, action in row["option_codes"].items():
                for code, interval in row["segment_codes"].items():
                    shown = f"{code} {seconds_label(interval[0])}–{seconds_label(interval[1])}s"
                    self.assertIn(shown + " |", row["prompt"])
                    with_seconds = json.dumps({"answer": letter, "event": shown})
                    primary = parse_reply(with_seconds, row["option_codes"], row["segment_codes"], "matched_seconds")
                    self.assertEqual((primary["result_format"], primary["answer_id"], primary["interval_s"]),
                                     ("ok", action, interval))
                    self.assertIn(primary["interval_s"], row["physical_window_order"])
                    bare = json.dumps({"answer": letter, "event": code})
                    strict = parse_reply(bare, row["option_codes"], row["segment_codes"], "strict")
                    self.assertEqual((strict["result_format"], strict["interval_s"]), ("ok", interval))
                    self.assertEqual(parse_reply(with_seconds, row["option_codes"], row["segment_codes"],
                                                 "strict")["result_format"], "invalid_value")


if __name__ == "__main__":
    unittest.main()
