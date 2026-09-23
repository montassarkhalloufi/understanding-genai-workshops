"""Deterministic regressions; no networking or historical results changed."""
from contextlib import ExitStack, redirect_stdout
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
import horizon_tools as tools
import horizon_cooperation as coop
import horizon_documents as documents


def message(content="", tool=None, reason="stop"):
    value = {"role": "assistant", "content": content}
    if tool:
        name, args = tool
        value["tool_calls"] = [{"function": {
            "name": name, "arguments": args}}]
    return {"message": value, "done": True, "done_reason": reason}


def fake_api(path, payload=None):
    if path == "/api/embed":
        size = len(payload["input"]) if isinstance(payload["input"], list) else 1
        return {"embeddings": [[1.0, 0.0] for _ in range(size)]}
    if path == "/api/chat":
        return message(json.dumps({
            "answer": "Simulated example", "abstain": True, "citations": []}))
    return {}


class ReliabilityTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        directory = self.stack.enter_context(tempfile.TemporaryDirectory())
        self.work = Path(directory)
        self.stack.enter_context(redirect_stdout(io.StringIO()))
        self.stack.enter_context(patch.object(socket.socket, "connect",
            side_effect=AssertionError("Network forbidden")))
        for module in (rag, tools, coop, documents):
            self.stack.enter_context(patch.object(module, "api",
                side_effect=AssertionError("Unmocked API")))

    def read(self, name):
        return json.loads((self.work / name).read_text(encoding="utf-8"))

    def test_01_existing_report_is_never_overwritten(self):
        path = self.work / "old.json"
        path.write_text('{"historical": true}\n')
        before = path.read_bytes()
        with patch.object(rag, "api") as provider:
            with self.assertRaises(FileExistsError):
                rag.experiment("development", path)
        self.assertEqual(before, path.read_bytes())
        provider.assert_not_called()

    def test_02_metadata_failure_initializes_failed_attempt(self):
        with patch.object(rag, "api", side_effect=TimeoutError("x" * 500)):
            rag.experiment("development", self.work / "rag.json")
        report = self.read("rag.json")
        self.assertEqual(report["status"], "incomplete")
        self.assertEqual(report["steps"][0]["status"], "failed")
        self.assertEqual(len(report["error"]["message"]), 300)
        self.assertIn("run_id", report)

    def test_03_second_embedding_retains_success_and_failure(self):
        seen = 0

        def api(path, payload=None):
            nonlocal seen
            if path == "/api/embed" and isinstance(payload["input"], str):
                seen += 1
                if seen == 2:
                    raise TimeoutError("Second question")
            return fake_api(path, payload)

        with patch.object(rag, "api", side_effect=api):
            rag.experiment("development", self.work / "rag.json", True)
        report = self.read("rag.json")
        self.assertEqual(report["status"], "incomplete")
        self.assertEqual([c["status"] for c in report["cases"]],
                         ["complete", "failed"])
        self.assertTrue(report["cases"][0]["generation"]["shape_ok"])
        self.assertEqual(report["cases"][1]["error"]["type"], "TimeoutError")

    def test_04_generation_failure_is_recorded(self):
        def api(path, payload=None):
            if path == "/api/chat":
                raise TimeoutError("Generation")
            return fake_api(path, payload)

        with patch.object(rag, "api", side_effect=api):
            rag.experiment("development", self.work / "rag.json", True)
        report = self.read("rag.json")
        self.assertEqual(report["status"], "incomplete")
        self.assertEqual(report["cases"][0]["status"], "failed")
        self.assertIn("rankings", report["cases"][0])

    def test_05_projection_handles_dispatcher_errors(self):
        examples = [({"room": "Atlas", "extra": 1}, "Incorrect fields"),
                    ([], "Incorrect fields"),
                    ({"room": "Missing"}, "unknown room")]
        for args, expected in examples:
            with self.subTest(args=args):
                execution = {"name": "inspect_room",
                             "result": tools.execute("inspect_room", args)}
                report = {"turns": [{"executions": [execution]}]}
                self.assertIn(expected, tools.operational_message(report))
        legacy = {"turns": [{"executions": [{"result": {"status": "invalid"}}]}]}
        self.assertIn("Invalid arguments", tools.operational_message(legacy))

    def test_06_tool_output_limit_executes_nothing(self):
        tool = ("inspect_room", {"room": "Atlas"})
        with patch.object(tools, "api", return_value=
                message("Interrupted text", tool, "length")):
            with patch.object(tools, "execute") as execute:
                report = tools.run_agent("Inspect Atlas")
        execute.assert_not_called()
        self.assertEqual(report["stop"], "output_limit")
        self.assertEqual(report["status"], "incomplete")

    def test_07_tool_second_call_preserves_first_observation(self):
        tool = ("inspect_room", {"room": "Atlas"})
        with patch.object(tools, "api", side_effect=[
                message(tool=tool), TimeoutError("Synthesis")]):
            tools.experiment(self.work / "tools.json", ["Inspect Atlas"])
        report = self.read("tools.json")
        case = report["cases"][0]
        self.assertEqual(report["status"], "incomplete")
        self.assertEqual(case["model_calls"], 2)
        self.assertEqual(case["turns"][0]["executions"][0]["result"]["capacity"], 8)
        self.assertEqual(case["turns"][1]["status"], "failed")

    def test_08_cooperation_metadata_failure_has_a_report(self):
        with patch.object(coop, "api", side_effect=TimeoutError("Metadata")):
            coop.run(self.work / "coop.json")
        report = self.read("coop.json")
        self.assertEqual(report["status"], "incomplete")
        self.assertEqual(report["steps"][0]["status"], "failed")

    def test_09_workflow_failure_preserves_both_searches(self):
        with patch.object(coop, "api", return_value={}):
            with patch.object(coop, "generate", side_effect=TimeoutError("Synthesis")):
                coop.run(self.work / "coop.json")
        report = self.read("coop.json")
        variant = report["variants"][0]
        self.assertEqual(report["status"], "incomplete")
        self.assertEqual(variant["evidence_ids"], ["orion-v1", "teams-v1"])
        self.assertEqual(len(variant["observations"]), 2)
        self.assertEqual(variant["tool_calls"], 2)
        self.assertEqual(variant["model_calls"], 1)
        self.assertEqual(variant["status"], "failed")

    def test_10_research_failure_preserves_previous_tool(self):
        sequence = iter([message(tool=("search", {"query": "Orion"})),
                         TimeoutError("Second turn")])

        def api(path, payload=None):
            if path != "/api/chat":
                return {}
            result = next(sequence)
            if isinstance(result, Exception):
                raise result
            return result

        with patch.object(coop, "api", side_effect=api):
            with patch.object(coop, "generate", return_value={
                    "status": "complete", "model_calls": 1}):
                coop.run(self.work / "coop.json")
        report = self.read("coop.json")
        self.assertEqual(report["variants"][0]["status"], "complete")
        variant = report["variants"][1]
        self.assertEqual(variant["model_calls"], 2)
        self.assertEqual(variant["tool_calls"], 1)
        self.assertEqual(variant["evidence_ids"], ["orion-v1"])
        trace = variant["research"][0]
        self.assertEqual(trace["evidence_ids"], ["orion-v1"])
        self.assertEqual(trace["turns"][1]["status"], "failed")

    def test_11_final_generation_limit_is_separate_from_json_shape(self):
        content = json.dumps({"answer": "Example", "abstain": False,
                              "citations": []})
        with patch.object(rag, "api", return_value=message(content, reason="length")):
            with patch.object(coop, "api", return_value={}):
                coop.run(self.work / "coop.json")
        report = self.read("coop.json")
        generated = report["variants"][0]["generation"]
        self.assertTrue(generated["shape_ok"])
        self.assertEqual(generated["status"], "output_limit")
        self.assertEqual(report["status"], "incomplete")
        self.assertEqual(len(report["variants"]), 1)

    def test_12_pdf_missing_quantity_keeps_the_whole_row(self):
        (self.work / "data").mkdir()
        with patch.object(documents, "ROOT", self.work):
            source = documents.create_fixture(missing=True)
            result = documents.extract_pdf(source)
        self.assertEqual(len(result["items"]), 3)
        row = result["items"][1]
        self.assertEqual(row["item"], "Blue marker")
        self.assertIsNone(row["quantity"])
        self.assertIsNone(row["calculated_amount"])
        self.assertEqual(row["amount"], "10.00")
        self.assertEqual(len(row["evidence"]["cells"]), 4)
        self.assertTrue(all(row["evidence"]["cells"]))
        self.assertEqual(result["computed_total"], "71.80")
        self.assertEqual(result["status"], "needs_review")

    def test_13_cli_returns_nonzero_for_provider_failure(self):
        cases = [("horizon_rag.py", ["development", "--generate"]),
                 ("horizon_tools.py", ["live"]),
                 ("horizon_cooperation.py", [])]
        for name, args in cases:
            with self.subTest(program=name):
                path = Path(rag.__file__).parent / name
                output = self.work / (name + ".json")
                argv = [str(path), *args, "--output", str(output)]
                with patch.object(sys, "argv", argv):
                    with patch("urllib.request.urlopen", side_effect=TimeoutError("CLI")):
                        with self.assertRaises(SystemExit) as exit_info:
                            runpy.run_path(str(path), run_name="__main__")
                self.assertEqual(exit_info.exception.code, 1)
                self.assertEqual(json.loads(output.read_text(encoding="utf-8"))["status"], "incomplete")

    def test_14_completed_runs_keep_separate_technical_and_content_states(self):
        with patch.object(rag, "api", side_effect=fake_api):
            report = rag.experiment("development", self.work / "rag.json", True)
        self.assertEqual(report["status"], "complete")
        self.assertEqual(len(report["cases"]), 4)
        with patch.object(tools, "api", return_value=message("No proposal")):
            report = tools.experiment(self.work / "tools.json", ["Example"])
        self.assertEqual(report["status"], "complete")
        self.assertEqual(report["cases"][0]["effects"], 0)
        with patch.object(coop, "api", side_effect=fake_api):
            with patch.object(rag, "api", side_effect=fake_api):
                report = coop.run(self.work / "coop.json")
        self.assertEqual(report["status"], "complete")
        self.assertEqual([v["model_calls"] for v in report["variants"]], [1, 1, 2])
        self.assertEqual([v["tool_calls"] for v in report["variants"]], [2, 0, 0])
        self.assertEqual(report["variants"][1]["generation"]["source"], "application")


if __name__ == "__main__":
    unittest.main(verbosity=2)
