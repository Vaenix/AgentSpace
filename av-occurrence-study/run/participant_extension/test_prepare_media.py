"""Offline regression tests for prepare_media.py; they need ffmpeg but no network."""
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prepare_media  # noqa: E402

PAIR = "P11_104-7"


def synthetic_clip(path, seconds):
    """Write a dual-track clip shaped like recover_review_media.py output (5 fps H.264, 24 kHz AAC)."""
    subprocess.run([
        "ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i", f"testsrc2=size=480x270:rate=5:duration={seconds}",
        "-f", "lavfi", "-i", f"sine=frequency=440:sample_rate=24000:duration={seconds}",
        "-map", "0:v:0", "-map", "1:a:0", "-c:v", "libx264", "-preset", "veryfast",
        "-ac", "1", "-ar", "24000", "-c:a", "aac", "-movflags", "+faststart", str(path),
    ], check=True)


def empty_and_valid(work):
    """Window 1 exists but is empty; window 2 is valid. Both exist, so nothing is downloaded."""
    media = work / "review_media"
    media.mkdir()
    (media / f"{PAIR}_1.mp4").write_bytes(b"")
    synthetic_clip(media / f"{PAIR}_2.mp4", 9.695)


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "needs ffmpeg and ffprobe")
class UnprobeableClip(unittest.TestCase):
    def test_empty_clip_is_recorded_and_other_slots_kept(self):
        queue, _, _ = prepare_media.verify_queue()
        with tempfile.TemporaryDirectory() as temp:
            work = Path(temp)
            empty_and_valid(work)
            records = prepare_media.prepare(queue, {PAIR}, work)

        self.assertEqual(len(records), 24)
        first, second = records[0], records[1]
        self.assertEqual((first["pair_id"], first["ordinal"]), (PAIR, 1))
        self.assertEqual(first["status"], "decode_failed")
        self.assertIn("could not be probed", first["failure_reason"])
        self.assertIn("ffprobe exited", first["failure_reason"])
        self.assertEqual(first["bytes"], 0)
        self.assertEqual(second["status"], "decoded")
        self.assertEqual(set(second["streams"]) >= {"video", "audio"}, True)
        for record in (first, second):
            self.assertTrue(record["frames_status"].startswith("skipped"))
        self.assertTrue(all(r["status"] == "not_attempted" for r in records[2:]))

        summary = prepare_media.summarize(records, "q", "c")
        self.assertEqual(summary["windows_decoded"], 1)
        self.assertEqual(summary["windows_failed"], {"decode_failed": 1})
        self.assertEqual(summary["pairs_complete"], 0)

    def test_failure_manifests_replay_byte_for_byte(self):
        script = Path(prepare_media.__file__)
        with tempfile.TemporaryDirectory() as temp:
            work = Path(temp)
            empty_and_valid(work)
            runs = []
            for name, check in (("first", None), ("second", "first")):
                command = [sys.executable, str(script), "--work-dir", str(work), "--output", str(work / name),
                           "--pairs", PAIR]
                if check:
                    command += ["--check", str(work / check)]
                runs.append(subprocess.run(command, capture_output=True, text=True))
            self.assertEqual([run.returncode for run in runs], [0, 0], runs[-1].stderr)
            self.assertIn("identical to", runs[-1].stdout)
            for name in prepare_media.OUTPUTS:
                self.assertEqual((work / "first" / name).read_bytes(), (work / "second" / name).read_bytes(), name)
            reason = json.loads((work / "first" / "windows.jsonl").read_text().splitlines()[0])["failure_reason"]
        self.assertNotRegex(reason, r"@ 0x[0-9a-fA-F]+")
        self.assertNotIn(temp, reason)
        for kept in ("ffprobe exited with status 1", f"{PAIR}_1.mp4", "moov atom not found"):
            self.assertIn(kept, reason)

    def test_stable_text(self):
        text = "/w/review_media/a.mp4: [mov,mp4 @ 0x74c7054000] moov atom not found [h264 @ 0xABC]"
        self.assertEqual(prepare_media.stable_text(text, Path("/w/review_media/a.mp4")),
                         "a.mp4: [mov,mp4] moov atom not found [h264]")


if __name__ == "__main__":
    unittest.main()
