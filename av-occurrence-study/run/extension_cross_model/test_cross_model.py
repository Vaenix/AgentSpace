"""Offline checks for the cross-model ledger and paired replay; no media, GPU or model needed."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lock
import run_cross_model
import summarize_cross_model

frozen = lock.frozen
single = summarize_cross_model.single
INPUTS = frozen.load()
CALLS = INPUTS["calls"]
PARSER = single.load_parser()
RUNNABLE = [call for call in CALLS if call["status"] == "runnable"]
REFERENCE = lock.reference_paths()


def reply(call, raw, generated_tokens=20):
    prompt = call["prompt"]
    answer, interval, result_format = PARSER.decode(raw, call["option_codes"], call["segment_codes"])
    return {
        "run_order": call["run_order"], "video_id": call["video_id"], "pair_id": call["pair_id"],
        "qid": call["qid"], "ordinal": call["ordinal"], "condition": call["condition"],
        "prompt_sha256": call["prompt_sha256"], "answer_id": answer,
        "action": prompt["semantic_actions"].get(answer), "interval_s": interval,
        "physical_window": None if interval is None else prompt["physical_window_order"].index(interval) + 1,
        "result_format": result_format, "raw_response": raw, "generated_tokens": generated_tokens, "run_id": "test",
    }


def code_for_window(call, window):
    interval = call["prompt"]["physical_window_order"][window - 1]
    return next(code for code, value in call["segment_codes"].items() if value == interval)


def letter_for(call, answer_id):
    return next(letter for letter, value in call["option_codes"].items() if value == answer_id)


def first_action(call):
    return next(iter(call["prompt"]["semantic_actions"]))


def find(pair, ordinal, condition):
    return next(c for c in CALLS if (c["pair_id"], c["ordinal"], c["condition"]) == (pair, ordinal, condition))


def stable_rows():
    return {(call["qid"], call["condition"]): reply(call, json.dumps(
        {"answer": letter_for(call, first_action(call)), "event": code_for_window(call, 1)})) for call in RUNNABLE}


def replay_rows(rows, attempts=()):
    with tempfile.TemporaryDirectory() as directory:
        replies, log = Path(directory) / "replies.jsonl", Path(directory) / "attempts.jsonl"
        replies.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
        log.write_text("".join(json.dumps(row) + "\n" for row in attempts), encoding="utf-8")
        return summarize_cross_model.replay(replies, log)


class PairedReplayTest(unittest.TestCase):
    def test_reference_against_itself_is_symmetric(self):
        report = summarize_cross_model.replay(REFERENCE["replies.jsonl"], REFERENCE["attempts.jsonl"])
        options = report["rules"]["matched_seconds"]["contrasts"]["options_reversed"]
        self.assertEqual(options["planned_questions"], 24)
        self.assertEqual(len(options["questions"]), 24)
        for model in summarize_cross_model.MODELS:
            self.assertEqual(options[model]["comparable_questions"], 14)
            self.assertEqual(options[model]["joint_events"], 4)
            self.assertEqual(options[model]["videos_with_event"], ["P21_01", "P35_105"])
        self.assertEqual(options["jointly_comparable"]["classes"], {"both": 4, "qwen_only": 0, "smolvlm2_only": 0,
                                                                     "neither": 10})
        labels = report["rules"]["matched_seconds"]["contrasts"]["segment_labels_reversed"]
        self.assertEqual(labels["smolvlm2_2_2b"]["videos_with_event"], ["P06_102", "P07_04"])
        self.assertEqual(report["rules"]["strict"]["contrasts"]["segment_labels_reversed"]
                         ["jointly_comparable"]["jointly_comparable_questions"], 0)

    def test_joint_classes_videos_and_reply_notes(self):
        rows = stable_rows()
        # SmolVLM2 shows the event where Qwen also does (P35_105 first question, options reversed).
        both = find("P35_105-5", 1, "options_reversed")
        rows[both["qid"], both["condition"]] = reply(both, json.dumps(
            {"answer": letter_for(both, first_action(both)), "event": code_for_window(both, 2)}))
        # ... and where Qwen does not (P28_104 first question, options reversed).
        only = find("P28_104-7", 1, "options_reversed")
        rows[only["qid"], only["condition"]] = reply(only, json.dumps(
            {"answer": letter_for(only, first_action(only)), "event": code_for_window(only, 2)}))
        # A free-text refusal is a format error, listed verbatim and not counted as an abstention.
        refusal = find("P21_01-7", 1, "options_reversed")
        rows[refusal["qid"], refusal["condition"]] = reply(refusal, "I cannot tell from these images.", 64)
        abstained = find("P09_103-5", 2, "options_reversed")
        rows[abstained["qid"], abstained["condition"]] = reply(abstained, '{"answer": null, "event": null}')
        report = replay_rows(list(rows.values()))
        rule = report["rules"]["matched_seconds"]
        options = rule["contrasts"]["options_reversed"]
        smol = options["smolvlm2_2_2b"]
        self.assertEqual(smol["comparable_questions"], 20)
        self.assertEqual(smol["joint_events"], 2)
        self.assertEqual(smol["videos_with_event"], ["P28_104", "P35_105"])
        self.assertEqual(sum(smol["cells"].values()), smol["comparable_questions"])
        common = options["jointly_comparable"]
        # Qwen comparable 14; SmolVLM2 loses P21_01 q1 and P09_103 q2, both Qwen-comparable.
        self.assertEqual(common["jointly_comparable_questions"], 12)
        self.assertEqual(common["classes"], {"both": 1, "qwen_only": 2, "smolvlm2_only": 1, "neither": 8})
        self.assertEqual(common["qwen_videos_with_event"], ["P21_01", "P35_105"])
        self.assertEqual(common["smolvlm2_videos_with_event"], ["P28_104", "P35_105"])
        smol_replies = rule["replies"]["smolvlm2_2_2b"]
        self.assertEqual(smol_replies["by_condition"]["options_reversed"]["statuses"]["format_error"], 1)
        self.assertEqual(smol_replies["by_condition"]["options_reversed"]["abstained"], 1)
        self.assertEqual(smol_replies["format_error_replies"],
                         [{"qid": refusal["qid"], "condition": "options_reversed",
                           "raw_response": "I cannot tell from these images."}])
        self.assertEqual(smol_replies["token_budget_hits"],
                         [{"qid": refusal["qid"], "condition": "options_reversed", "status": "format_error"}])
        row = next(r for r in options["questions"] if r["qid"] == only["qid"])
        self.assertEqual((row["joint_class"], row["smolvlm2_cell"], row["qwen_cell"]),
                         ("smolvlm2_only", "action_stable_window_changed", "action_stable_window_stable"))

    def test_generation_failure_is_missing_not_abstention(self):
        rows = stable_rows()
        dropped = rows.pop(next(iter(rows)))
        identity = {name: dropped[name] for name in run_cross_model.LEDGER_IDENTITY}
        with self.assertRaisesRegex(ValueError, "no reply and no logged attempt"):
            replay_rows(list(rows.values()))
        attempts = [dict(identity, run_id="r1", event="started"),
                    dict(identity, run_id="r1", event="failed", error="RuntimeError: metal")]
        report = replay_rows(list(rows.values()), attempts)
        counts = report["rules"]["matched_seconds"]["replies"]["smolvlm2_2_2b"]["by_condition"][dropped["condition"]]
        self.assertEqual(counts["statuses"]["technical_missing"], 3)
        self.assertEqual(counts["abstained"], 0)

    def test_archived_statistics_replay_byte_identical(self):
        here = Path(__file__).resolve().parent
        report = summarize_cross_model.replay(here / "replies.jsonl", here / "attempts.jsonl")
        text = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
        self.assertEqual(text, (here / "statistics.json").read_text(encoding="utf-8"))

    def test_reference_ledger_is_pinned(self):
        self.assertEqual(frozen.file_sha256(REFERENCE["replies.jsonl"]), lock.REFERENCE_SHA256["replies.jsonl"])
        self.assertEqual(lock.load_lock()["model"]["revision"], "be92141c8d0ceb98596a923c308ac2bbb4a7c466")

    def test_runner_reuses_stored_replies_and_refuses_bad_rows(self):
        rows = list(stable_rows().values())[:3]
        with tempfile.TemporaryDirectory() as directory:
            ledger = Path(directory) / "replies.jsonl"
            ledger.write_text("".join(json.dumps(row) + "\n" for row in rows))
            stored = run_cross_model.stored_replies(ledger, CALLS)
            self.assertEqual(list(stored), [(row["qid"], row["condition"]) for row in rows])
            ledger.write_text("".join(json.dumps(row) + "\n" for row in rows + rows[:1]))
            with self.assertRaisesRegex(ValueError, "duplicate reply"):
                run_cross_model.stored_replies(ledger, CALLS)
            ledger.write_text(json.dumps(rows[0]))
            with self.assertRaisesRegex(ValueError, "unterminated"):
                run_cross_model.stored_replies(ledger, CALLS)

    def test_native_template_rendering(self):
        try:
            import jinja2  # noqa: F401
        except ImportError:
            self.skipTest("jinja2 not installed")
        template = ("<|im_start|>{% for message in messages %}{{message['role'] | capitalize}}"
                    "{% if message['content'] is string %}{{': ' + message['content']}}{% else %}"
                    "{% if message['content'][0]['type'] == 'image' %}{{':'}}{% else %}{{': '}}{% endif %}"
                    "{% for line in message['content'] %}{% if line['type'] == 'text' %}{{line['text']}}"
                    "{% elif line['type'] == 'image' %}{{ '<image>' }}{% endif %}{% endfor %}{% endif %}"
                    "<end_of_utterance>\n{% endfor %}{% if add_generation_prompt %}{{ 'Assistant:' }}{% endif %}")
        from hashlib import sha256
        self.assertEqual(sha256(template.encode()).hexdigest(),
                         lock.load_lock()["chat_template"]["sha256_of_template_string"])
        self.assertEqual(lock.render_native(template, "Q?"),
                         "<|im_start|>User:" + "<image>" * 6 + "Q?<end_of_utterance>\nAssistant:")


if __name__ == "__main__":
    unittest.main()
