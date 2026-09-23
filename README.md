# Understanding GenAI — Book workshops

Companion workshops for **Understanding GenAI**, by **Montassar Khalloufi**.

All source code, comments, prompts, examples, fictional fixtures, and documentation use English. The same workshops support both language editions of the book. Complete books are not included. See [book correspondence](BOOK-EDITIONS.md).

Download the latest workshop archive from [Releases](https://github.com/montassarkhalloufi/understanding-genai-workshops/releases), extract it, and open a terminal inside the extracted folder.

All people, rooms, and delivery notes in the Horizon corpus are fictional. Tool calls cannot change a real booking service. No DevMethod checkout is required.

## Run the offline checks

Use Python 3.12 or a compatible later version in an activated virtual environment. The current checks were run with Python 3.12.14. Open a terminal in this folder, then run:

```sh
python -m pip install -r requirements-documents.txt
python first_network.py
python horizon_rag.py test
python horizon_tools.py test
python test_cooperation.py
python horizon_documents.py test
python -m unittest discover -v
```

The XOR program has four assertions and normally prints nothing. The four workshop checks contain 12 RAG, 15 tool, 12 cooperation, and 9 PDF/payload assertions. Test discovery runs 22 unittest cases (2 encoding, 6 RAG variants, and 14 reliability cases); it also imports the cooperation script and executes its 12 assertions. These counts describe written checks, not exhaustive coverage. PDF tests create three fictional PDF/image pairs and `results/documents-offline.json`. No model or network call is needed for these checks. See `verification/current-offline.json` for the actual verification environment and outcomes.

## Optional local model experiments

Install Ollama using its [official guide](https://docs.ollama.com/quickstart). Model weights are not included. Downloads require a network connection; the programs subsequently use the local interface at `http://127.0.0.1:11434`.

```sh
ollama pull qwen2.5:1.5b
ollama pull all-minilm:latest
ollama list
python horizon_rag.py development --generate
python horizon_rag.py held_out --generate
python horizon_tools.py live
python horizon_cooperation.py
```

These commands run the current English prompts and corpus. They create new reports in `results/`. Each attempt gets a separate report; technical interruptions remain visible. RAG, tools, and cooperation accept `--output` with a new path and refuse to overwrite an existing attempt. A completed attempt does not establish answer correctness. Fixed seeds and temperatures do not guarantee identical outputs across installations.

No live model experiment was performed as part of the English-code conversion. Changing language, prompts, corpus text, stop words, and document identifiers can change retrieval and generation. Historical French scores are not scores for this English implementation.

## Compare context sizes

Prepare a fresh dataset before inspecting outputs. Its structure follows `data/horizon-questions.json`; keep answer keys outside the model context. Example structure to fill with your own cases:

```json
{"held_out": [{"id": "fresh-1", "question": "Your question", "sources": [], "correction": "Expected answer"}]}
```

An empty `sources` list means that no eligible source supports an answer. This placeholder is not a ready evaluation dataset. Save your completed dataset as `data/unseen-questions.json` and compare:

```sh
python horizon_rag.py held_out --questions data/unseen-questions.json --top-k 3 --generate
python horizon_rag.py held_out --questions data/unseen-questions.json --top-k 5 --generate
```

`--top-k` controls retrieval, context selection, and recall measurements together; its default is 3. The published held-out cases are explained in the book and therefore no longer constitute unseen cases for a reader who has studied them.

## Contents

- `first_network.py`: XOR network, four assertions.
- `horizon_rag.py`: access filtering, lexical/dense/hybrid retrieval, structured answers, exact-quote checks, and attempt journal.
- `horizon_tools.py`: fictional room catalog, parameter validation, proposal, content-bound approval, sequential idempotence, and operational messages.
- `horizon_cooperation.py`: diagnostic comparison of a fixed workflow, one agent, and two agents with sequential handoff.
- `horizon_documents.py`: three fictional delivery notes, known-template extraction, coordinates, decimal checks, and an optional vision adapter.
- `test_*.py`: offline encoding, loop, retrieval-parameter, failure, and controller checks.
- `data/`: current English corpus, questions, and PDF/image fixtures.
- `results/`: newly produced outputs, separated from historical records.
- `verification/`: actual checks of this current English code.

## Historical evidence and current behavior

Read [EVIDENCE.md](EVIDENCE.md) before interpreting historical measurements discussed in the books. Original experimental records are retained separately in the author's production archive. They are not distributed here. Current English programs are not an exact reproduction of the earlier experiments, and historical scores do not establish their model quality.

Current controllers refuse synthesis without observations, separate interrupted provider output from valid JSON shape, preserve progress on failure, project tool states directly, and keep partially extracted PDF rows. Offline checks validate those software behaviors using mocks and fixtures. They provide no new measurement of model quality.

The optional vision adapter requires an explicitly selected compatible model and an image. See `python horizon_documents.py --help`. Only PDF extraction and construction of the image request were tested; no vision-model performance is claimed. Exact quote membership does not prove entailment. In-memory approvals do not replace authentication or concurrent database transactions.

Libraries and models are distributed separately by their respective projects. Checksums establish file integrity, not publisher authentication.

## License

Original code and accompanying guides: [MIT](LICENSE). See [third-party notices and book exclusions](THIRD-PARTY-NOTICES.md).
