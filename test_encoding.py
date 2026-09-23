"""Check Unicode input with a simulated cp1252 default reader."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import shutil
import unittest
from unittest.mock import patch
import horizon_rag as rag
import horizon_cooperation as coop

READ_TEXT = Path.read_text

def locale_read(path, encoding=None, errors=None):
    return READ_TEXT(path, encoding=encoding or 'cp1252', errors=errors)

class EncodingTests(unittest.TestCase):
    def prepare_unicode_fixture(self, folder):
        fixture = Path(folder)
        shutil.copytree(rag.ROOT / "data", fixture / "data")
        corpus = fixture / "data/horizon-corpus.json"
        documents = json.loads(READ_TEXT(corpus, encoding="utf-8"))
        for document in documents:
            document["text"] += " Currency display: £10 → €12."
        corpus.write_text(json.dumps(documents, ensure_ascii=False), encoding="utf-8")
        return fixture

    def check_documents(self, documents):
        text = json.dumps(documents, ensure_ascii=False)
        self.assertIn('£10 → €12', text)
        self.assertNotIn('â‚¬', text)
        return rag.eligible(documents)

    def test_rag_uses_utf8_under_simulated_legacy_locale(self):
        seen = []
        def api(path, payload=None):
            if path == '/api/embed':
                items = payload['input']
                if isinstance(items, list):
                    seen.extend(items)
                    return {'embeddings': [[1, 0] for _ in items]}
                return {'embeddings': [[1, 0]]}
            return {}
        with tempfile.TemporaryDirectory() as folder, patch.object(Path, 'read_text', locale_read), patch.object(rag, 'api', side_effect=api), contextlib.redirect_stdout(io.StringIO()):
            fixture = self.prepare_unicode_fixture(folder)
            with patch.object(rag, 'ROOT', fixture):
                result = rag.experiment('development', Path(folder) / 'report.json')
        self.assertEqual(result['status'], 'complete')
        text = '\n'.join(seen)
        self.assertIn('£10 → €12', text)
        self.assertNotIn('â‚¬', text)

    def test_cooperation_reads_utf8_before_provider(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(Path, 'read_text', locale_read), patch.object(coop, 'eligible', side_effect=self.check_documents) as eligible, patch.object(coop, 'api', side_effect=RuntimeError('simulated stop before provider')):
            fixture = self.prepare_unicode_fixture(folder)
            with patch.object(coop, 'ROOT', fixture):
                coop.run(Path(folder) / 'report.json')
        eligible.assert_called_once()

if __name__ == '__main__':
    unittest.main()
