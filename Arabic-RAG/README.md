# Arabic Document Assistant

Ask questions in Arabic and receive answers grounded in the included Arabic document. The right-to-left interface shows the answer and three expandable source passages with PDF page numbers.

## Run

Install Python 3.11+ and [Ollama](https://ollama.com/download). From the repository root:

```bash
python run.py arabic
```

The launcher installs Python dependencies, starts Ollama if necessary, downloads `llama3.2:3b` if missing, and opens http://127.0.0.1:8001 after the app is ready. The first launch needs internet access for dependencies and models. Keep the launcher open; press Ctrl+C to stop.

The included index is ready to use. Tesseract is needed only to repeat OCR, not to use the interface.

## Command-line tools

After launcher setup, use the virtual environment's Python:

```bash
# Git Bash on Windows:
./.venv/Scripts/python.exe Arabic-RAG/code/retrieve.py
./.venv/Scripts/python.exe Arabic-RAG/code/ask_question.py
./.venv/Scripts/python.exe Arabic-RAG/code/evaluate.py
```

On macOS/Linux, replace `./.venv/Scripts/python.exe` with `./.venv/bin/python`.

See the [main guide](../README.md) for setup and troubleshooting, and [REQUIREMENTS.md](../REQUIREMENTS.md) for OCR and index rebuild commands.
