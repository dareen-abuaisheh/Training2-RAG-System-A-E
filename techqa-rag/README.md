# English Document Search

Search the included TechQA documents in English. The browser interface returns the three highest-ranked distinct documents. Select a result to read its complete cleaned text. Similarity is a ranking score, not a probability of correctness.

## Run

Install Python 3.11 or newer. From the repository root:

```bash
python run.py english
```

On Windows, you can also double-click `start.bat` and choose **English**.

The launcher installs Python dependencies and opens http://127.0.0.1:8000 after the app is ready. The first launch needs internet access to download dependencies and the embedding model. Keep the launcher open; press Ctrl+C to stop.

The included index is ready to use. The browser interface does not require Ollama.

## Command-line tools

After launcher setup, use the virtual environment's Python:

```bash
# Git Bash on Windows:
./.venv/Scripts/python.exe techqa-rag/code/retrieve.py
./.venv/Scripts/python.exe techqa-rag/code/ask_question.py
```

On macOS/Linux, replace `./.venv/Scripts/python.exe` with `./.venv/bin/python`.

Command-line answer generation requires Ollama running with `llama3.2:3b`. Install [Ollama](https://ollama.com/download), open it, and run `ollama pull llama3.2:3b` first.

See the [main guide](../README.md) for setup and troubleshooting, and [REQUIREMENTS.md](../REQUIREMENTS.md) for manual installation and index rebuild commands.
