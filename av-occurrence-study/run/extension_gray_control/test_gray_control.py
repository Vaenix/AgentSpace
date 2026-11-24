"""Offline checks for the gray-image control; no media, GPU or model needed."""
import json
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
import zlib

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gray_inputs  # noqa: E402
from gray_inputs import frozen  # noqa: E402
import run_gray  # noqa: E402
import summarize_gray  # noqa: E402

HERE = gray_inputs.HERE
INPUTS = gray_inputs.load()
KEYS = INPUTS["keys"]
RUNNABLE = [call for call in KEYS if call["status"] == "runnable"]
PARSER = summarize_gray.extension.load_parser()
INTACT_TOKENS = run_gray.intact_prompt_tokens(INPUTS["intact_replies"])


def png(width, height, pixel, filter_type=0):
    row = bytes(pixel) * width
    if filter_type == 2:
        body = b"".join(bytes([0 if y == 0 else 2]) + (row if y == 0 else bytes(len(row))) for y in range(height))
    else:
        body = b"".join(b"\x00" + row for _ in range(height))

    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(body)) + chunk(b"IEND", b"")


def gray_reply(call, raw):
    prompt = call["prompt"]
    answer, interval, result_format = PARSER.decode(raw, call["option_codes"], call["segment_codes"])
    return {
        "run_order": call["run_order"], "video_id": call["video_id"], "pair_id": call["pair_id"],
        "qid": call["qid"], "ordinal": call["ordinal"], "condition": call["condition"],
        "image_condition": "gray", "image_slots": 6, "image_sha256": gray_inputs.PROTOCOL_SHA256["gray.png"],
        "prompt_sha256": call["prompt_sha256"], "answer_id": answer,
        "action": prompt["semantic_actions"].get(answer), "interval_s": interval,
        "physical_window": None if interval is None else prompt["physical_window_order"].index(interval) + 1,
        "result_format": result_format, "raw_response": raw, "prompt_tokens": INTACT_TOKENS[call["qid"]],
        "intact_prompt_tokens": INTACT_TOKENS[call["qid"]], "prompt_tokens_match": True, "run_id": "test",
    }


def null_reply(call):
    return gray_reply(call, json.dumps({"answer": None, "event": None}))


def write_ledgers(folder, replies, attempts=()):
    replies_file, attempts_file = Path(folder) / "replies.jsonl", Path(folder) / "attempts.jsonl"
    replies_file.write_text("".join(json.dumps(row) + "\n" for row in replies), encoding="utf-8")
    attempts_file.write_text("".join(json.dumps(row) + "\n" for row in attempts), encoding="utf-8")
    return replies_file, attempts_file


class GrayImageTest(unittest.TestCase):
    def test_reference_image_is_uniform_gray(self):
        self.assertEqual(INPUTS["gray"]["pixels_checked"], 448 * 448)
        self.assertEqual(gray_inputs.check_gray(png(448, 448, (128, 128, 128), filter_type=2))["pixel"],
                         [128, 128, 128])

    def test_other_size_or_value_is_rejected(self):
        with self.assertRaises(ValueError):
            gray_inputs.check_gray(png(447, 448, (128, 128, 128)))
        with self.assertRaises(ValueError):
            gray_inputs.check_gray(png(448, 448, (128, 128, 127)))

    def test_planned_keys_follow_the_frozen_original_order(self):
        self.assertEqual(len(KEYS), 24)
        self.assertEqual(len(RUNNABLE), 22)
        self.assertEqual({call["condition"] for call in KEYS}, {"original"})
        self.assertEqual([call["qid"] for call in KEYS if call["status"] == "technical_missing"],
                         ["P11_104-7-smoke-1", "P11_104-7-smoke-2"])


class ReplayTest(unittest.TestCase):
    def replay(self, replies, attempts=()):
        with tempfile.TemporaryDirectory() as folder:
            return summarize_gray.replay(*write_ledgers(folder, replies, attempts))

    def test_archived_statistics_replay_exactly(self):
        report = summarize_gray.replay(HERE / "replies.jsonl", HERE / "attempts.jsonl")
        text = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
        self.assertEqual(text, (HERE / "statistics.json").read_text(encoding="utf-8"))

    def test_tables_sum_to_their_denominators(self):
        report = summarize_gray.replay(HERE / "replies.jsonl", HERE / "attempts.jsonl")
        paired = report["paired"]
        self.assertEqual(sum(paired["cells"].values()), paired["paired_keys"])
        self.assertEqual(sum(report["secondary"]["cells"].values()), report["secondary"]["both_complete"])
        self.assertEqual(paired["cells"]["intact_complete_gray_complete"], report["secondary"]["both_complete"])
        for counts in report["conditions"].values():
            self.assertEqual(sum(counts["statuses"].values()), 24)
        self.assertEqual(report["per_recording"]["groups"]["no_paired_reply"], 1)

    def test_all_null_gray_replies_lose_every_complete_intact_key(self):
        report = self.replay([null_reply(call) for call in RUNNABLE])
        paired, intact = report["paired"], report["conditions"]["intact"]
        self.assertEqual(paired["paired_keys"], 22)
        self.assertEqual(paired["cells"]["intact_complete_gray_incomplete"], intact["complete"])
        self.assertEqual(paired["gray_complete"], 0)
        self.assertEqual(paired["difference_gray_minus_intact"], -intact["complete"] / 22)
        self.assertEqual(report["conditions"]["gray"]["both_abstained"], 22)
        self.assertEqual(report["secondary"]["both_complete"], 0)

    def test_failed_key_is_technical_missing_not_abstention(self):
        dropped = RUNNABLE[0]
        attempts = [
            {"run_id": "r", "event": "started", **{name: dropped[name] for name in run_gray.LEDGER_IDENTITY}},
            {"run_id": "r", "event": "failed", "error": "RuntimeError: test",
             **{name: dropped[name] for name in run_gray.LEDGER_IDENTITY}},
        ]
        report = self.replay([null_reply(call) for call in RUNNABLE[1:]], attempts)
        gray = report["conditions"]["gray"]
        self.assertEqual((gray["received"], gray["both_abstained"]), (21, 21))
        self.assertEqual(gray["technical_missing"], {"total": 3, "frozen_frame_extraction": 2, "generation": 1})
        self.assertEqual(report["paired"]["paired_keys"], 21)

    def test_missing_reply_without_logged_attempt_is_refused(self):
        with self.assertRaises(ValueError):
            self.replay([null_reply(call) for call in RUNNABLE[1:]])

    def test_token_mismatch_is_reported_as_deviation(self):
        rows = [null_reply(call) for call in RUNNABLE]
        rows[0].update(prompt_tokens=rows[0]["prompt_tokens"] + 1, prompt_tokens_match=False)
        deviations = self.replay(rows)["prompt_tokens"]["protocol_deviations"]
        self.assertEqual([row["qid"] for row in deviations], [RUNNABLE[0]["qid"]])

    def test_tampered_or_duplicate_rows_are_refused(self):
        rows = [null_reply(call) for call in RUNNABLE]
        with self.assertRaises(ValueError):
            self.replay(rows + rows[:1])
        tampered = [dict(row) for row in rows]
        tampered[0]["result_format"] = "invalid_json"
        with self.assertRaises(ValueError):
            self.replay(tampered)
        intact_image = [dict(row) for row in rows]
        intact_image[0]["image_condition"] = "intact"
        with self.assertRaises(ValueError):
            self.replay(intact_image)


class RunnerTest(unittest.TestCase):
    def test_stored_replies_are_reused_without_loading_the_model(self):
        with tempfile.TemporaryDirectory() as folder:
            shutil.copy(HERE / "replies.jsonl", folder)
            shutil.copy(HERE / "attempts.jsonl", folder)
            before = (Path(folder) / "replies.jsonl").read_bytes()
            result = subprocess.run(
                [sys.executable, str(HERE / "run_gray.py"), "--model", folder, "--results-dir", folder],
                capture_output=True, text=True, check=True)
            counts = json.loads(result.stdout.strip().splitlines()[-1])
            self.assertEqual(counts["generated"], 0)
            self.assertEqual(counts["reused"], 22)
            self.assertEqual((Path(folder) / "replies.jsonl").read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
