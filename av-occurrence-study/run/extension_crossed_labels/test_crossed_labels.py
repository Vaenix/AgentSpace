"""Offline checks for the joint-reversal condition; no media, GPU or model needed."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import crossed_inputs  # noqa: E402
from crossed_inputs import frozen  # noqa: E402

HERE = crossed_inputs.HERE
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


if __name__ == "__main__":
    unittest.main()
