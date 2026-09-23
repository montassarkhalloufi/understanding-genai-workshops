"""Context variants and fresh datasets, without networking or real models."""
from contextlib import ExitStack, redirect_stderr, redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import runpy
import socket
import sys
import tempfile
import unittest
from unittest.mock import patch
import horizon_rag as rag


class RagVariantsTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.work = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.stack.enter_context(redirect_stdout(io.StringIO()))
        self.stack.enter_context(patch.object(socket.socket, "connect",
            side_effect=AssertionError("Network forbidden")))
        self.question = {"id": "new-1", "question": "Fresh case",
                         "sources": ["d4"], "correction": "Do not send"}
        self.docs = [{"id": f"d{i}", "title": "Record", "text": "Fresh case"}
                     for i in range(6)]
        self.questions_path = self.work / "questions.json"
        self.questions_path.write_text(json.dumps({"held_out": [self.question]}))

    def evaluate(self, *extra):
        case = {}
        rag.evaluate_case(self.question, case, self.docs, [[1, 0]] * 6, [1, 0], *extra)
        return case

    def test_three_and_five_change_context_and_recall_together(self):
        for top_k in (3, 5):
            with self.subTest(top_k=top_k):
                case = self.evaluate(top_k)
                expected = [f"d{i}" for i in range(top_k)]
                self.assertEqual(case["context_ids"], expected)
                for metrics in case["retrieval"].values():
                    self.assertEqual(metrics[f"top{top_k}"], expected)
                    self.assertEqual(metrics["all_required"], top_k == 5)
                    self.assertEqual(metrics["required_retrieved"],
                                     ["d4"] if top_k == 5 else [])

    def test_default_and_short_corpus(self):
        self.assertEqual(rag.positive_integer("3"), 3)
        self.assertEqual(rag.positive_integer("5"), 5)
        self.assertEqual(self.evaluate(), self.evaluate(3))
        self.assertEqual(len(self.evaluate(20)["context_ids"]), 6)

    def test_invalid_top_k_fails_before_provider_and_report(self):
        for value in (0, -1, 1.5, True, "5", None):
            with self.subTest(value=value), patch.object(rag, "api") as api:
                with self.assertRaisesRegex(ValueError, "strictly positive integer"):
                    rag.experiment("held_out", self.work / "bad.json", top_k=value)
                with self.assertRaises(ValueError):
                    self.evaluate(value)
                api.assert_not_called()
                self.assertFalse((self.work / "bad.json").exists())

    def test_report_generation_and_custom_questions(self):
        def fake_api(path, payload=None):
            if path == "/api/embed":
                count = len(payload["input"]) if isinstance(payload["input"], list) else 1
                return {"embeddings": [[1, 0]] * count}
            return {}
        for top_k in (3, 5):
            output = self.work / f"result-{top_k}.json"
            with patch.object(rag, "api", side_effect=fake_api):
                with patch.object(rag, "answer", return_value={"status": "complete"}) as answer:
                    report = rag.experiment("held_out", output, True, top_k, self.questions_path)
            self.assertEqual(report["status"], "complete")
            self.assertEqual(report["top_k"], top_k)
            self.assertEqual(len(report["cases"]), 1)
            self.assertEqual(report["fingerprints"][self.questions_path.name],
                hashlib.sha256(self.questions_path.read_bytes()).hexdigest())
            question, context = answer.call_args.args
            self.assertEqual(question, self.question["question"])
            self.assertEqual(len(context), top_k)
            self.assertEqual([d["id"] for d in context], report["cases"][0]["context_ids"])
            self.assertNotIn("Do not send", json.dumps(context))
            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), json.loads(json.dumps(report)))

    def test_bad_question_files_fail_before_provider(self):
        values = ([], {}, {"held_out": []}, {"held_out": [{}]},
                  {"held_out": [dict(self.question, sources="d4")]},
                  {"held_out": [self.question, self.question]})
        for value in values:
            with self.subTest(value=value), patch.object(rag, "api") as api:
                self.questions_path.write_text(json.dumps(value))
                with self.assertRaises(ValueError):
                    rag.experiment("held_out", self.work / "bad.json", questions_path=self.questions_path)
                api.assert_not_called()
        self.questions_path.write_text("{invalid")
        with self.assertRaisesRegex(ValueError, "Cannot read"):
            rag.load_questions(self.questions_path, "held_out")
        with self.assertRaisesRegex(ValueError, "Cannot read"):
            rag.load_questions(self.work / "missing.json", "held_out")

    def test_cli_validation_has_readable_errors(self):
        cases = [(["--top-k", value], "strictly positive integer")
                 for value in ("0", "-1", "1.5", "abc")]
        cases.append((["--questions", str(self.work / "missing.json")], "Cannot read"))
        for options, expected in cases:
            with self.subTest(options=options):
                error = io.StringIO()
                with patch.object(sys, "argv", [rag.__file__, "held_out", *options]), redirect_stderr(error):
                    with self.assertRaises(SystemExit) as result:
                        runpy.run_path(rag.__file__, run_name="__main__")
                self.assertEqual(result.exception.code, 2)
                self.assertIn(expected, error.getvalue())
                self.assertNotIn("Traceback", error.getvalue())


if __name__ == "__main__":
    unittest.main(verbosity=2)
