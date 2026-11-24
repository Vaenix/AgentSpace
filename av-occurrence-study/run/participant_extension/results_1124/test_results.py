"""Offline checks for the extension ledger and replay; no media, GPU or model needed."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import frozen
import run_generation
import summarize

INPUTS = frozen.load()
CALLS = INPUTS["calls"]
PARSER = summarize.load_parser()
RUNNABLE = [call for call in CALLS if call["status"] == "runnable"]


def reply(call, raw):
    prompt = call["prompt"]
    answer, interval, result_format = PARSER.decode(raw, call["option_codes"], call["segment_codes"])
    return {
        "run_order": call["run_order"], "video_id": call["video_id"], "pair_id": call["pair_id"],
        "qid": call["qid"], "ordinal": call["ordinal"], "condition": call["condition"],
        "prompt_sha256": call["prompt_sha256"], "answer_id": answer,
        "action": prompt["semantic_actions"].get(answer), "interval_s": interval,
        "physical_window": None if interval is None else prompt["physical_window_order"].index(interval) + 1,
        "result_format": result_format, "raw_response": raw, "run_id": "test",
    }


def code_for_window(call, window):
    interval = call["prompt"]["physical_window_order"][window - 1]
    return next(code for code, value in call["segment_codes"].items() if value == interval)


def letter_for(call, answer_id):
    return next(letter for letter, value in call["option_codes"].items() if value == answer_id)


def first_action(call):
    return next(iter(call["prompt"]["semantic_actions"]))


def stable_replies():
    """Every reply picks the same semantic action and physical window 1."""
    return [reply(call, json.dumps({"answer": letter_for(call, first_action(call)),
                                     "event": code_for_window(call, 1)})) for call in RUNNABLE]


def run_replay(rows, attempts=()):
    decoded, received = summarize.decode_calls(CALLS, rows, list(attempts), PARSER)
    return {rule: summarize.summarize(found, INPUTS["recordings"]) for rule, found in decoded.items()}, received


def find(pair, ordinal, condition):
    return next(call for call in CALLS if (call["pair_id"], call["ordinal"], call["condition"]) == (pair, ordinal, condition))


class ReplayTest(unittest.TestCase):
    def test_denominators_sum_to_plan(self):
        report, received = run_replay(stable_replies())
        self.assertEqual(received, 66)
        for rule in summarize.RULES:
            result = report[rule]
            self.assertEqual(result["technical_missing"]["missing_calls"], 6)
            self.assertEqual({row["qid"][:9] for row in result["technical_missing"]["keys"]}, {"P11_104-7"})
            for counts in result["replies_by_condition"].values():
                self.assertEqual(counts["planned"], 24)
                self.assertEqual(sum(counts["statuses"].values()), 24)
                self.assertEqual(counts["statuses"]["technical_missing"], 2)
            for contrast in result["comparisons"].values():
                questions = contrast["questions"]
                self.assertEqual(questions["planned_questions"], 24)
                self.assertEqual(sum(questions["cells"].values()), questions["comparable_questions"])
                self.assertEqual(questions["comparable_questions"], 22)
                self.assertEqual(questions["joint_events"], 0)
                self.assertEqual(contrast["across_recordings"]["classes"],
                                 {"event_observed": 0, "comparable_no_event": 11, "no_comparable_question": 1})
            moves = result["original_first_to_second"]
            self.assertEqual(moves["complete_pairs"], 11)
            self.assertEqual(moves["table"]["W1->W1"]["same_action"], 11)

    def test_statuses_events_and_transitions(self):
        rows = {(row["qid"], row["condition"]): row for row in stable_replies()}
        pair = "P06_102-5"
        swap = find(pair, 1, "options_reversed")
        rows[swap["qid"], swap["condition"]] = reply(swap, json.dumps(
            {"answer": letter_for(swap, first_action(swap)), "event": code_for_window(swap, 2)}))
        for condition in frozen.CONDITIONS:
            second = find(pair, 2, condition)
            rows[second["qid"], second["condition"]] = reply(second, json.dumps(
                {"answer": letter_for(second, first_action(second)), "event": code_for_window(second, 2)}))
        labels = find(pair, 1, "segment_labels_reversed")
        code = code_for_window(labels, 1)
        start, end = labels["segment_codes"][code]
        rows[labels["qid"], labels["condition"]] = reply(labels, json.dumps(
            {"answer": letter_for(labels, first_action(labels)), "event": f"{code} {start:g}–{end:g}s"}))
        other = find("P01_05-4", 2, "options_reversed")
        rows[other["qid"], other["condition"]] = reply(other, '{"answer": null, "event": null}')
        third = find("P01_05-4", 1, "segment_labels_reversed")
        rows[third["qid"], third["condition"]] = reply(third, '{"answer": "A", "event": "E01 1-2s"}')
        report, _ = run_replay(list(rows.values()))

        primary, strict = report["matched_seconds"], report["strict"]
        options = primary["comparisons"]["options_reversed"]
        self.assertEqual(options["questions"]["cells"]["action_stable_window_changed"], 1)
        self.assertEqual(options["questions"]["joint_events"], 1)
        self.assertEqual(options["recordings"][pair]["recording_class"], "event_observed")
        self.assertEqual(options["questions"]["comparable_questions"], 21)
        self.assertIn({"original_status": "complete", "intervention_status": "both_abstained", "count": 1},
                      options["questions"]["non_comparable_status_pairs"])
        self.assertEqual(options["across_recordings"]["exactly_one_comparable_question"], 1)
        self.assertEqual(primary["replies_by_condition"]["options_reversed"]["abstained"], 1)
        self.assertEqual(primary["replies_by_condition"]["segment_labels_reversed"]["statuses"]["code_seconds_conflict"], 1)
        self.assertEqual(strict["replies_by_condition"]["segment_labels_reversed"]["statuses"]["invalid_field"], 2)
        self.assertEqual(primary["comparisons"]["segment_labels_reversed"]["questions"]["comparable_questions"], 21)
        self.assertEqual(strict["comparisons"]["segment_labels_reversed"]["questions"]["comparable_questions"], 20)
        moves = primary["original_first_to_second"]
        self.assertEqual(moves["table"]["W1->W2"]["same_action"], 1)
        self.assertEqual(moves["table"]["W1->W1"]["same_action"], 10)

    def test_missing_key_requires_logged_attempt(self):
        rows = stable_replies()
        dropped = rows.pop(0)
        with self.assertRaisesRegex(ValueError, "no reply and no logged attempt"):
            run_replay(rows)
        identity = {name: dropped[name] for name in run_generation.LEDGER_IDENTITY}
        attempts = [dict(identity, run_id="r1", event="started"),
                    dict(identity, run_id="r1", event="failed", error="RuntimeError: metal"),
                    dict(identity, run_id="r2", event="started")]
        report, received = run_replay(rows, attempts)
        self.assertEqual(received, 65)
        missing = report["matched_seconds"]["technical_missing"]
        self.assertEqual(missing["missing_calls"], 7)
        record = next(row for row in missing["keys"] if row["qid"] == dropped["qid"] and row["condition"] == dropped["condition"])
        self.assertEqual(record["stage"], "generation")
        self.assertEqual([item["log_reference"] for item in record["attempts"]], ["attempts.jsonl:2", "attempts.jsonl:3"])
        self.assertEqual(record["reason"], "process ended during generation without a reply")

    def test_ledger_mismatches_are_rejected(self):
        rows = stable_replies()
        with self.assertRaisesRegex(ValueError, "Duplicate reply"):
            run_replay(rows + rows[:1])
        tampered = [dict(rows[0], physical_window=2)] + rows[1:]
        with self.assertRaisesRegex(ValueError, "Strict re-decoding differs"):
            run_replay(tampered)
        other_prompt = [dict(rows[0], prompt_sha256="0" * 64)] + rows[1:]
        with self.assertRaisesRegex(ValueError, "another prompt"):
            run_replay(other_prompt)

    def test_runner_reuses_stored_replies_and_refuses_partial_rows(self):
        rows = stable_replies()[:3]
        with tempfile.TemporaryDirectory() as directory:
            ledger = Path(directory) / "replies.jsonl"
            ledger.write_text("".join(json.dumps(row) + "\n" for row in rows))
            stored = run_generation.stored_replies(ledger, CALLS)
            self.assertEqual(list(stored), [(row["qid"], row["condition"]) for row in rows])
            ledger.write_text("".join(json.dumps(row) + "\n" for row in rows + rows[:1]))
            with self.assertRaisesRegex(ValueError, "duplicate reply"):
                run_generation.stored_replies(ledger, CALLS)
            ledger.write_text(json.dumps(rows[0]))
            with self.assertRaisesRegex(ValueError, "unterminated"):
                run_generation.stored_replies(ledger, CALLS)
            missing = next(call for call in CALLS if call["status"] == "technical_missing")
            ledger.write_text(json.dumps(dict(rows[0], qid=missing["qid"], condition=missing["condition"])) + "\n")
            with self.assertRaisesRegex(ValueError, "not a runnable frozen key"):
                run_generation.stored_replies(ledger, CALLS)


if __name__ == "__main__":
    unittest.main()
