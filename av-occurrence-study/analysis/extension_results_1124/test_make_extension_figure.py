import importlib.util
from pathlib import Path
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("make_extension_figure", HERE / "make_extension_figure.py")
fig = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fig)


def reply(status, answer_id=None, window=None):
    return {"status": status, "answer_id": answer_id, "physical_window": window}


class ClassifyTest(unittest.TestCase):
    def test_comparable_cells(self):
        self.assertEqual(fig.classify(reply("complete", "a", 1), reply("complete", "a", 2)),
                         ("comparable_target_event", "action_stable_window_changed"))
        self.assertEqual(fig.classify(reply("complete", "a", 1), reply("complete", "a", 1)),
                         ("comparable_no_event", "action_stable_window_stable"))
        self.assertEqual(fig.classify(reply("complete", "a", 2), reply("complete", "b", 1)),
                         ("comparable_no_event", "action_changed_window_changed"))

    def test_non_comparable_reasons(self):
        missing = reply("technical_missing")
        self.assertEqual(fig.classify(missing, missing), ("technical_missing", None))
        self.assertEqual(fig.classify(reply("complete", "a", 1), reply("both_abstained")),
                         ("not_comparable_abstention", None))
        self.assertEqual(fig.classify(reply("answer_abstained", None, 1), reply("answer_abstained", None, 1)),
                         ("not_comparable_abstention", None))

    def test_unlegended_state_is_refused(self):
        with self.assertRaises(SystemExit):
            fig.classify(reply("invalid_field"), reply("complete", "a", 1))


class OutputTest(unittest.TestCase):
    def test_data_outputs_match_committed(self):
        with tempfile.TemporaryDirectory() as tmp:
            fig.generate(Path(tmp), figures=False)
            for name in fig.DATA_OUTPUTS:
                self.assertTrue(fig.same_output(name, HERE, Path(tmp)), name)


if __name__ == "__main__":
    unittest.main()
