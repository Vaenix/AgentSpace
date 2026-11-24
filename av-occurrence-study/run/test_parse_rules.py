"""Text-only tests for the reply parse rules and the offline 1124 replay.

Run: python3 -m unittest discover -s av-occurrence-study/run -p 'test_*.py' -v
"""
import json
from pathlib import Path
import sys
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "replication_1124"))
from infer_mlx import decode, parse_reply, prompt_for  # noqa: E402
import replay_parse  # noqa: E402

OPTIONS = {"A": "action-a", "B": "action-b"}
FIRST, SECOND = [530.566, 533.272], [629.763, 636.539]
MAPPING = {"E01": FIRST, "E02": SECOND}
REVERSED = {"E02": FIRST, "E01": SECOND}


def reply(answer, event):
    return "```json\n" + json.dumps({"answer": answer, "event": event}, ensure_ascii=False) + "\n```"


def parse(answer, event, mapping=MAPPING, rule="matched_seconds"):
    return parse_reply(reply(answer, event), OPTIONS, mapping, rule)


class LegalCode(unittest.TestCase):
    def test_bare_code_is_accepted_by_both_rules(self):
        for rule in ("strict", "matched_seconds"):
            parsed = parse("A", "E02", rule=rule)
            self.assertEqual((parsed["answer_id"], parsed["interval_s"], parsed["result_format"]),
                             ("action-a", SECOND, "ok"))

    def test_unknown_code_is_an_invalid_field(self):
        parsed = parse("A", "E03")
        self.assertEqual((parsed["interval_s"], parsed["event_status"], parsed["result_format"]),
                         (None, "invalid", "invalid_value"))

    def test_unknown_answer_letter_is_an_invalid_field(self):
        parsed = parse("C", "E01")
        self.assertEqual((parsed["answer_id"], parsed["answer_status"], parsed["result_format"]),
                         (None, "invalid", "invalid_value"))
        self.assertEqual(parsed["interval_s"], FIRST)


class MatchedSeconds(unittest.TestCase):
    def test_code_with_displayed_seconds_restores_window(self):
        for event in ("E01 530.566–533.272s", "E01 530.566-533.272", "E01  530.566 – 533.272 s", "E01 530.5660–533.272s"):
            with self.subTest(event=event):
                parsed = parse("A", event)
                self.assertEqual((parsed["interval_s"], parsed["event_status"], parsed["result_format"]),
                                 (FIRST, "ok", "ok"))

    def test_strict_default_still_rejects_code_with_seconds(self):
        response = reply("A", "E01 530.566–533.272s")
        self.assertEqual(decode(response, OPTIONS, MAPPING), ("action-a", None, "invalid_value"))
        self.assertEqual(parse("A", "E01 530.566–533.272s", rule="strict")["event_status"], "invalid")

    def test_matches_seconds_as_printed_in_the_prompt(self):
        mapping = {"E01": [6.917, 7.945], "E02": [94.69200000000001, 95.53]}
        self.assertEqual(parse("A", "E02 94.692–95.53s", mapping)["interval_s"], [94.69200000000001, 95.53])

    def test_unknown_code_with_seconds_is_invalid(self):
        parsed = parse("A", "E03 530.566–533.272s")
        self.assertEqual((parsed["event_status"], parsed["result_format"]), ("invalid", "invalid_value"))

    def test_malformed_seconds_are_invalid(self):
        for event in ("E01 530.566s", "530.566–533.272s", "E01 at 530.566–533.272s"):
            with self.subTest(event=event):
                self.assertEqual(parse("A", event)["event_status"], "invalid")


class ConflictingSeconds(unittest.TestCase):
    def test_seconds_of_another_window_conflict_with_the_code(self):
        parsed = parse("A", "E01 629.763–636.539s")
        self.assertEqual((parsed["interval_s"], parsed["event_status"], parsed["result_format"]),
                         (None, "conflict", "code_seconds_conflict"))
        self.assertEqual(parsed["answer_id"], "action-a")

    def test_partially_matching_seconds_conflict(self):
        for event in ("E01 530.566–636.539s", "E01 530.566–533.27s", "E02 1–2s"):
            with self.subTest(event=event):
                self.assertEqual(parse("A", event)["result_format"], "code_seconds_conflict")


class ReversedMapping(unittest.TestCase):
    def test_bare_code_uses_this_conditions_mapping(self):
        self.assertEqual(parse("A", "E02", REVERSED)["interval_s"], FIRST)
        self.assertEqual(parse("A", "E01", REVERSED)["interval_s"], SECOND)

    def test_code_with_seconds_uses_this_conditions_mapping(self):
        parsed = parse("A", "E02 530.566–533.272s", REVERSED)
        self.assertEqual((parsed["interval_s"], parsed["result_format"]), (FIRST, "ok"))

    def test_original_code_with_reversed_seconds_conflicts(self):
        parsed = parse("A", "E01 530.566–533.272s", REVERSED)
        self.assertEqual((parsed["interval_s"], parsed["result_format"]), (None, "code_seconds_conflict"))


class Abstention(unittest.TestCase):
    def test_explicit_null_is_an_abstention(self):
        parsed = parse(None, None)
        self.assertEqual((parsed["answer_id"], parsed["interval_s"], parsed["result_format"]), (None, None, "abstained"))
        self.assertEqual((parsed["answer_status"], parsed["event_status"]), ("null", "null"))

    def test_strict_keeps_reporting_explicit_null_as_ok(self):
        self.assertEqual(decode(reply(None, None), OPTIONS, MAPPING), (None, None, "ok"))

    def test_single_null_is_a_partial_abstention(self):
        self.assertEqual(parse("A", None)["result_format"], "partially_abstained")
        self.assertEqual(parse(None, "E01 530.566–533.272s")["result_format"], "partially_abstained")

    def test_null_is_not_hidden_by_other_problems(self):
        self.assertEqual(parse(None, "E01 629.763–636.539s")["result_format"], "code_seconds_conflict")
        self.assertEqual(parse(None, 3)["result_format"], "invalid_value")


class Envelope(unittest.TestCase):
    def test_unparseable_reply_and_wrong_keys(self):
        for rule in ("strict", "matched_seconds"):
            self.assertEqual(parse_reply("E01", OPTIONS, MAPPING, rule)["result_format"], "invalid_json")
            self.assertEqual(parse_reply('{"answer": "A"}', OPTIONS, MAPPING, rule)["result_format"], "invalid_keys")

    def test_unknown_rule_is_rejected(self):
        with self.assertRaises(ValueError):
            parse_reply(reply("A", "E01"), OPTIONS, MAPPING, "lenient")


class FrozenReplication(unittest.TestCase):
    """Replays the frozen replication_1124 text files; nothing is written."""

    @classmethod
    def setUpClass(cls):
        cls.strict = replay_parse.frozen_replay("strict")
        cls.matched = replay_parse.frozen_replay("matched_seconds")

    def test_prompt_generation_is_unchanged(self):
        manifest = HERE / "replication_1124" / "manifest"
        frozen = {(row["qid"], row["condition"]): row for row in replay_parse.read(manifest / "prompts.jsonl")}
        rows = [json.loads(line) for path in sorted(manifest.glob("frames/*/smoke-input.jsonl"))
                for line in path.read_text().splitlines() if line.strip()]
        self.assertEqual(len(rows), 6)
        for row in rows:
            prompt, _, options, segments = prompt_for(row, "flat")
            expected = frozen[row["qid"], "original"]
            self.assertEqual((prompt, options, segments),
                             (expected["prompt"], expected["option_codes"], expected["segment_codes"]))

    def test_strict_replay_reproduces_stored_statistics(self):
        summary = json.loads((HERE / "replication_1124" / "summary.json").read_text())
        self.assertEqual(self.strict["strict_recheck"]["replies_rechecked"], 18)
        self.assertEqual(self.strict["metrics"], summary)

    def test_matched_seconds_replay_counts(self):
        self.assertEqual(len(self.matched["replies"]), 18)
        for condition in ("original", "options_reversed", "segment_labels_reversed"):
            self.assertEqual(self.matched["reply_outcomes_by_condition"][condition], {"total": 6, "abstained": 2, "ok": 4})
        for condition, changed in (("options_reversed", 2), ("segment_labels_reversed", 0)):
            metrics = self.matched["metrics"][condition]
            self.assertEqual((metrics["all_questions"], metrics["comparable_questions"],
                              metrics["action_stable_and_window_changed"]), (6, 4, changed))
            self.assertEqual(self.matched["not_comparable_reasons"][condition],
                             {"original=abstained, test=abstained": 2})
        self.assertTrue(self.matched["matches_post_hoc_format_sensitivity"])

    def test_reversed_labels_restore_the_same_physical_window(self):
        windows = {(row["qid"], row["condition"]): row["physical_window"] for row in self.matched["replies"]}
        self.assertEqual(windows["P03_117-2-smoke-1", "original"], 1)
        self.assertEqual(windows["P03_117-2-smoke-1", "segment_labels_reversed"], 1)
        self.assertEqual(windows["P03_117-2-smoke-1", "options_reversed"], 2)


if __name__ == "__main__":
    unittest.main()
