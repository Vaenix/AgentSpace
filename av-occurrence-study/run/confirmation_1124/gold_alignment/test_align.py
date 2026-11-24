"""Tests for the gold/reply alignment entry.

Every fixture below is SYNTHETIC: pair, video, action and reviewer IDs start
with SYNTHETIC, texts are invented, and gold is produced by running the real
eval/freeze_reviews.py on synthetic review records. Only the last test reads
the real confirmation_1124 batch, and it checks keys and inputs without gold.

Run: python3 -m unittest discover -s av-occurrence-study/run/confirmation_1124/gold_alignment -v
"""

import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import align  # noqa: E402

FREEZE = align.STUDY / "eval/freeze_reviews.py"
PARSER, _ = align.summarize.load_parser()
SWAP_CODES = {"E01": "E02", "E02": "E01"}


def synthetic_pair(index, hints, windows=((100.0, 101.0), (400.0, 401.5))):
    video = f"SYNTHETIC_P9{index}_01"
    return {
        "pair_id": f"{video}-1", "video_id": video, "participant_id": f"SYNTHETIC_P9{index}",
        "duration_s": 900.0, "sound_class": "synthetic-click", "media_url": f"synthetic://{video}.mp4",
        "synthetic": True,
        "occurrences": [
            {"ordinal": ordinal, "sound_id": f"{video}_s{ordinal}", "sound_s": list(window),
             "action_id": f"{video}_a{ordinal}", "action_s": [window[1] + 1.0, window[1] + 3.0],
             "action_hint": hint}
            for ordinal, (hint, window) in enumerate(zip(hints, windows), 1)
        ],
    }


def synthetic_prompts(pair):
    """Mirror replication_1124/replicate.py freeze() for one synthetic pair."""
    options = dict(sorted(
        (align.semantic_id(item["action_id"]), item["action_hint"].strip().removesuffix(".") + ".")
        for item in pair["occurrences"]
    ))
    segments = [{
        "interval_s": item["sound_s"], "sound_description": "Sound label from the source annotation: synthetic-click.",
        "frames": [{"path": f"w{item['ordinal']}f{frame}.png", "time_s": item["sound_s"][0] + frame}
                   for frame in (1, 2, 3)],
    } for item in pair["occurrences"]]
    rows = []
    for ordinal, order in ((1, "first"), (2, "second")):
        question = {"question": f"SYNTHETIC: what happens after the {order} click?",
                    "options": options, "segments": segments}
        prompt, images, codes, intervals = PARSER.prompt_for(question, "flat")
        common = {"qid": f"{pair['pair_id']}-smoke-{ordinal}", "pair_id": pair["pair_id"],
                  "video_id": pair["video_id"], "ordinal": ordinal, "frames": images,
                  "physical_window_order": list(intervals.values()), "semantic_actions": options}
        rows.append(dict(common, condition="original", prompt=prompt, option_codes=codes, segment_codes=intervals))
        reversed_codes = {"A": codes["B"], "B": codes["A"]}
        reversed_prompt = PARSER.prompt_for(dict(question, options=dict(reversed(list(options.items())))), "flat")[0]
        rows.append(dict(common, condition="options_reversed", prompt=reversed_prompt,
                         option_codes=reversed_codes, segment_codes=intervals))
        swapped = "\n".join(SWAP_CODES[line[:3]] + line[3:] if line[:3] in SWAP_CODES else line
                            for line in prompt.splitlines())
        rows.append(dict(common, condition="segment_labels_reversed", prompt=swapped, option_codes=codes,
                         segment_codes={SWAP_CODES[key]: value for key, value in intervals.items()}))
    return rows


def letter_for(prompt, source_ordinal, pair):
    target = align.semantic_id(pair["occurrences"][source_ordinal - 1]["action_id"])
    return next(letter for letter, value in prompt["option_codes"].items() if value == target)


def code_for(prompt, window):
    target = prompt["physical_window_order"][window - 1]
    return next(code for code, value in prompt["segment_codes"].items() if value == target)


def reply(answer, event):
    return "```json\n" + json.dumps({"answer": answer, "event": event}) + "\n```"


def correct(prompt, pair):
    """Choose the source action and window of the asked ordinal under this condition's codes."""
    return reply(letter_for(prompt, prompt["ordinal"], pair), code_for(prompt, prompt["ordinal"]))


def prediction_row(prompt, raw):
    parsed = PARSER.parse_reply(raw, prompt["option_codes"], prompt["segment_codes"], rule="strict")
    interval = parsed["interval_s"]
    return {
        **{key: prompt[key] for key in ("video_id", "pair_id", "qid", "ordinal", "condition")},
        "answer_id": parsed["answer_id"], "action": prompt["semantic_actions"].get(parsed["answer_id"]),
        "interval_s": interval,
        "physical_window": None if interval is None else prompt["physical_window_order"].index(interval) + 1,
        "result_format": parsed["result_format"], "raw_response": raw,
    }


def synthetic_batch(pairs, respond=correct):
    """respond(prompt, pair) returns a raw reply, or None for a traceable technical failure."""
    prompts = [row for pair in pairs for row in synthetic_prompts(pair)]
    by_pair = {pair["pair_id"]: pair for pair in pairs}
    predictions, failures = [], []
    for prompt in prompts:
        raw = respond(prompt, by_pair[prompt["pair_id"]])
        if raw is None:
            failures.append({"qid": prompt["qid"], "condition": prompt["condition"],
                             "error": "SYNTHETIC timeout", "log_reference": "synthetic.log:1"})
        else:
            predictions.append(prediction_row(prompt, raw))
    selection = "".join(json.dumps(pair) + "\n" for pair in pairs).encode()
    provenance = {"synthetic": True,
                  "text_input_sha256": {"manifest/selected.jsonl": hashlib.sha256(selection).hexdigest()}}
    return {"selected": pairs, "prompts": prompts, "predictions": predictions, "failures": failures,
            "provenance": provenance, "selection_bytes": selection}


def review(slot, queue_sha, pairs):
    return {
        "record_type": "independent_review", "schema_version": 1, "queue_revision": queue_sha,
        "reviewer": {"slot": slot, "code": f"SYNTHETIC_R{slot}"},
        "records": [{
            "pair_id": pair["pair_id"], "video_id": pair["video_id"],
            "judgments": {"disposition": "propose-include", "notes": "SYNTHETIC review note"},
            "clips": {"1": {"sound": "synthetic click"}},
        } for pair in pairs],
    }


def freeze_gold(directory, queue_pairs, adjudications):
    """Run the real freeze_reviews.py on synthetic reviews; adjudications maps pair_id to a row update."""
    queue = directory / "queue.jsonl"
    queue.write_bytes("".join(json.dumps(pair) + "\n" for pair in queue_pairs).encode())
    queue_sha = hashlib.sha256(queue.read_bytes()).hexdigest()
    for slot in "AB":
        (directory / f"review-{slot}.json").write_text(json.dumps(review(slot, queue_sha, queue_pairs)))
    common = ["--queue", str(queue), "--review-a", str(directory / "review-A.json"),
              "--review-b", str(directory / "review-B.json")]
    table = directory / "adjudication.tsv"
    subprocess.run([sys.executable, str(FREEZE), "prepare", *common, "--output", str(table)],
                   check=True, capture_output=True)
    with table.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, dialect="excel-tab")
        columns, rows = reader.fieldnames, list(reader)
    for row in rows:
        row.update(adjudications[row["题对"]])
    with table.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, columns, dialect="excel-tab")
        writer.writeheader()
        writer.writerows(rows)
    output = directory / "frozen"
    subprocess.run([sys.executable, str(FREEZE), "freeze", *common, "--adjudication", str(table),
                    "--adjudicator", "SYNTHETIC_R03", "--output", str(output)], check=True, capture_output=True)
    return output


def include(answers, sounds=("100.1~101.0", "400.0~401.4"), actions=("102.0~104.0", "402.0~404.0")):
    return {
        "决策": "纳入", "仲裁理由": "SYNTHETIC adjudication", "声音类别": "合成提示音",
        "发声1秒数": sounds[0], "动作1秒数": actions[0], "动作1": answers[0],
        "发声2秒数": sounds[1], "动作2秒数": actions[1], "动作2": answers[1],
        "干扰动作1": "synthetic distractor one", "干扰动作2": "synthetic distractor two",
    }


def decline(decision):
    return {"决策": decision, "仲裁理由": f"SYNTHETIC {decision}"}


class SyntheticCase(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="synthetic-gold-")
        self.directory = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def gold(self, queue_pairs, adjudications):
        return align.load_gold_bundle(freeze_gold(self.directory, queue_pairs, adjudications))

    def template(self, batch, gold):
        options, _, _ = align.check_batch(batch["selected"], batch["prompts"], batch["predictions"])
        return align.template_rows(batch["selected"], options, gold)

    def write_mapping(self, rows, choices):
        path = self.directory / "mapping.tsv"
        for row in rows:
            if row["原选项文本"] in choices:
                row.update({"映射": choices[row["原选项文本"]], "映射人": "SYNTHETIC_M1", "理由": "SYNTHETIC check"})
        path.unlink(missing_ok=True)
        align.write_template(path, rows)
        return path


def metric(report, condition, flag, rule="matched_seconds"):
    item = report["scoring"]["rules"][rule]["conditions"][condition]["metrics"][flag]
    return item["count"], item["total"]


class NormalAlignment(SyntheticCase):
    def test_identical_wording_aligns_to_new_gold_ids_and_scores_all_conditions(self):
        pair = synthetic_pair(1, ("open drawer", "close drawer"))
        batch = synthetic_batch([pair])
        gold = self.gold([pair], {pair["pair_id"]: include(("Open  Drawer", "close drawer"))})
        report = align.build_report(batch, gold)
        record = report["pairs"][0]
        self.assertEqual(record["status"], "aligned")
        self.assertEqual({item["basis"]["kind"] for item in record["original_options"]}, {"normalized_text_identical"})
        self.assertEqual([q["gold_qid"] for q in record["questions"]], [f"{pair['pair_id']}-1", f"{pair['pair_id']}-2"])
        self.assertEqual([q["original_qid"] for q in record["questions"]],
                         [f"{pair['pair_id']}-smoke-1", f"{pair['pair_id']}-smoke-2"])
        self.assertEqual(record["questions"][0]["adjudicated_interval_s"], [100.1, 101.0])
        self.assertEqual(report["status"], "已评分")
        for condition in align.CONDITIONS:
            for flag in ("answer_correct", "localization_correct"):
                self.assertEqual(metric(report, condition, flag), (2, 2), (condition, flag))
            self.assertEqual(metric(report, condition, "joint_pair_success"), (1, 1))
        scored = report["scoring"]["rules"]["matched_seconds"]["conditions"]["original"]["pairs"][0]["questions"]
        self.assertTrue(all(item["answer_id"].startswith("act_") for item in scored))
        self.assertEqual([list(item["interval_s"]) for item in scored], [[100.0, 101.0], [400.0, 401.5]])


class Reversal(SyntheticCase):
    def test_letters_and_codes_are_decoded_with_each_conditions_own_mapping(self):
        pair = synthetic_pair(2, ("pick up cup", "put down cup"))

        def sticky(prompt, item):
            # Keep the original condition's letter and code in every condition.
            original = next(row for row in synthetic_prompts(item)
                            if row["ordinal"] == prompt["ordinal"] and row["condition"] == "original")
            return correct(original, item)

        report = align.build_report(synthetic_batch([pair], sticky),
                                    self.gold([pair], {pair["pair_id"]: include(("pick up cup", "put down cup"))}))
        self.assertEqual(metric(report, "original", "joint_pair_success"), (1, 1))
        self.assertEqual(metric(report, "options_reversed", "answer_is_other_occurrence"), (2, 2))
        self.assertEqual(metric(report, "options_reversed", "answer_swap"), (1, 1))
        self.assertEqual(metric(report, "options_reversed", "localization_correct"), (2, 2))
        self.assertEqual(metric(report, "segment_labels_reversed", "localization_swap"), (1, 1))
        self.assertEqual(metric(report, "segment_labels_reversed", "answer_correct"), (2, 2))
        comparison = report["scoring"]["rules"]["matched_seconds"]["comparisons"][0]
        self.assertEqual(comparison["joint_pair_comparison"]["left_only_success"], 1)

    def test_code_with_matching_seconds_counts_only_under_the_primary_rule(self):
        pair = synthetic_pair(3, ("wipe table", "rinse sponge"))

        def with_seconds(prompt, item):
            if prompt["condition"] != "segment_labels_reversed":
                return correct(prompt, item)
            code = code_for(prompt, prompt["ordinal"])
            label = PARSER.interval_label(prompt["segment_codes"][code])
            return reply(letter_for(prompt, prompt["ordinal"], item), f"{code} {label}")

        report = align.build_report(synthetic_batch([pair], with_seconds),
                                    self.gold([pair], {pair["pair_id"]: include(("wipe table", "rinse sponge"))}))
        self.assertEqual(metric(report, "segment_labels_reversed", "localization_correct"), (2, 2))
        strict = report["scoring"]["rules"]["strict"]["conditions"]["segment_labels_reversed"]
        self.assertEqual((strict["status"], strict["metrics"]), ("未跑", None))
        self.assertEqual(strict["reply_statuses"]["unusable_reply"], 2)
        self.assertEqual(report["scoring"]["rules"]["strict"]["status"], "部分完成")


class Mapping(SyntheticCase):
    def test_unresolved_wording_is_pending_until_a_human_maps_it(self):
        pair = synthetic_pair(4, ("take lid", "stir pot"))
        batch = synthetic_batch([pair])
        gold = self.gold([pair], {pair["pair_id"]: include(("remove pan lid", "stir pot"))})
        rows = self.template(batch, gold)
        self.assertEqual([row["自动判定"] for row in rows], ["待人工映射", "规范化文字同金标2"])
        pending = align.build_report(batch, gold)
        self.assertEqual(pending["pairs"][0]["status"], "mapping_pending")
        self.assertEqual(pending["denominators"]["pairs"]["mapping_pending"], 1)
        self.assertEqual(pending["status"], "未跑")
        self.assertIsNone(pending["scoring"]["rules"]["matched_seconds"]["conditions"]["original"]["metrics"])

        mapped = align.build_report(batch, gold, self.write_mapping(rows, {"take lid.": "金标1"}))
        self.assertEqual(mapped["pairs"][0]["status"], "aligned")
        self.assertEqual(mapped["pairs"][0]["original_options"][0]["basis"]["kind"], "human_mapping")
        self.assertEqual(metric(mapped, "original", "answer_correct"), (2, 2))

    def test_crossed_human_mapping_follows_the_gold_not_the_source_ordinal(self):
        pair = synthetic_pair(5, ("grab towel", "dry hands"))
        batch = synthetic_batch([pair])
        gold = self.gold([pair], {pair["pair_id"]: include(("dry both hands", "grab a towel"))})
        path = self.write_mapping(self.template(batch, gold), {"grab towel.": "金标2", "dry hands.": "金标1"})
        report = align.build_report(batch, gold, path)
        self.assertEqual(metric(report, "original", "answer_correct"), (0, 2))
        self.assertEqual(metric(report, "original", "answer_swap"), (1, 1))

    def test_human_confirmed_absent_action_is_listed_separately(self):
        pair = synthetic_pair(6, ("open tap", "fill kettle"))
        batch = synthetic_batch([pair])
        gold = self.gold([pair], {pair["pair_id"]: include(("open tap", "put kettle on base"))})
        report = align.build_report(batch, gold, self.write_mapping(self.template(batch, gold), {"fill kettle.": "无对应"}))
        self.assertEqual(report["pairs"][0]["status"], "answer_not_in_original_options")
        self.assertEqual(report["denominators"]["pairs"]["answer_not_in_original_options"], 1)
        listed = report["answer_not_in_original_options"]
        self.assertEqual([(item["gold_qid"], item["gold_answer_text"]) for item in listed],
                         [(f"{pair['pair_id']}-2", "put kettle on base")])
        self.assertEqual(report["status"], "未跑")

    def test_contradicting_or_stale_mapping_is_rejected(self):
        pair = synthetic_pair(7, ("cut bread", "butter toast"))
        batch = synthetic_batch([pair])
        gold = self.gold([pair], {pair["pair_id"]: include(("cut bread", "spread butter"))})
        rows = self.template(batch, gold)
        with self.assertRaisesRegex(ValueError, "contradicts"):
            align.build_report(batch, gold, self.write_mapping(rows, {"cut bread.": "金标2"}))
        rows = self.template(batch, gold)
        rows[0]["金标指纹"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "does not match"):
            align.build_report(batch, gold, self.write_mapping(rows, {}))


class Denominators(SyntheticCase):
    def test_excluded_undecided_no_gold_missing_and_abstained_stay_in_the_counts(self):
        pairs = [synthetic_pair(index, (f"synthetic act {index}a", f"synthetic act {index}b")) for index in range(1, 5)]

        def respond(prompt, pair):
            if pair is pairs[0] and prompt["condition"] == "original":
                if prompt["ordinal"] == 1:
                    return None
                return reply(None, code_for(prompt, 2))
            return correct(prompt, pair)

        batch = synthetic_batch(pairs, respond)
        gold = self.gold(pairs[:3], {
            pairs[0]["pair_id"]: include(("synthetic act 1a", "synthetic act 1b")),
            pairs[1]["pair_id"]: decline("排除"),
            pairs[2]["pair_id"]: decline("待定"),
        })
        report = align.build_report(batch, gold)
        self.assertEqual(report["denominators"]["pairs"], {
            "no_gold": 1, "excluded": 1, "undecided": 1, "mapping_pending": 0,
            "answer_not_in_original_options": 0, "aligned": 1,
        })
        self.assertEqual(report["denominators"]["planned_questions"], 8)
        self.assertFalse(report["gold"]["queue_is_this_batch_selection"])
        original = report["scoring"]["rules"]["matched_seconds"]["conditions"]["original"]
        self.assertEqual(original["status"], "部分完成")
        self.assertEqual((original["reply_statuses"]["missing_prediction"], original["reply_statuses"]["abstained"]),
                         (1, 1))
        self.assertEqual(metric(report, "original", "prediction_present"), (1, 2))
        self.assertEqual(metric(report, "original", "abstained"), (1, 2))
        self.assertEqual(metric(report, "original", "localization_correct"), (1, 2))
        self.assertEqual(report["key_precheck"]["technical_failures"], 1)

    def test_gold_from_this_batch_selection_is_recognized(self):
        pair = synthetic_pair(8, ("lift pan", "lower pan"))
        batch = synthetic_batch([pair])
        directory = freeze_gold(self.directory, [pair], {pair["pair_id"]: include(("lift pan", "lower pan"))})
        self.assertEqual((self.directory / "queue.jsonl").read_bytes(), batch["selection_bytes"])
        report = align.build_report(batch, align.load_gold_bundle(directory))
        self.assertTrue(report["gold"]["queue_is_this_batch_selection"])


class RealBatchPrecheck(unittest.TestCase):
    def test_real_three_clips_get_key_and_input_checks_only(self):
        batch = align.load_frozen_batch(align.BATCH / "predictions.jsonl")
        report = align.build_report(batch)
        self.assertEqual(report["status"], "未跑")
        self.assertIsNone(report["gold"])
        self.assertTrue(report["inputs"]["predictions_match_statistics_json"])
        self.assertEqual({key: report["key_precheck"][key] for key in ("videos", "questions", "prompt_keys", "prediction_keys")},
                         {"videos": 3, "questions": 6, "prompt_keys": 18, "prediction_keys": 18})
        self.assertEqual(report["denominators"]["pairs"]["no_gold"], 3)
        for rule in report["scoring"]["rules"].values():
            self.assertEqual(rule["status"], "未跑")
            self.assertTrue(all(group["metrics"] is None for group in rule["conditions"].values()))
        committed = (HERE / "precheck.json").read_text(encoding="utf-8")
        self.assertEqual(committed, json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    unittest.main()
