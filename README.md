# RAG Document Assistant

Search documents and ask questions in Arabic and English using local AI models.

| Interface | Features | Address |
| --- | --- | --- |
| Arabic | Arabic questions, generated answers, source passages and PDF page numbers | http://127.0.0.1:8001 |
| English | Search TechQA documents and read the full matching documents | http://127.0.0.1:8000 |

Both projects also support answer generation from the command line. The English browser interface provides document search only.

## Quick start

1. Install **Python 3.11 or newer** from [python.org](https://www.python.org/downloads/). On Windows, enable **Add Python to PATH** during installation.
2. For Arabic answers, install [Ollama](https://ollama.com/download). The launcher starts it and downloads the answer model when needed. English document search works without Ollama.
3. Download and extract this repository, or clone it with Git.
4. On **Windows**, double-click **start.bat**, then choose Arabic, English, or both.

The launcher creates a virtual environment, installs Python dependencies, prepares the Arabic answer model when needed, and opens the selected interfaces in your browser. No environment activation or index rebuilding is needed.

The first launch needs internet access and may take several minutes to download packages and models. Later launches reuse them. Keep the launcher window open while using the app. Press **Ctrl+C** to stop it.

## Run from a terminal

From the repository folder, in Git Bash, PowerShell, or a terminal:

```bash
python run.py
```

Choose an interface directly:

```bash
python run.py arabic
python run.py english
python run.py both
```

On macOS/Linux, use `python3` if `python` is unavailable. Python must include the `venv` module; on Debian/Ubuntu install the matching `python3-venv` package if needed.

Use `--no-browser` to prevent automatic browser opening. The launcher prints the local addresses.

## Included data

Source documents, cleaned text, embeddings, FAISS indexes, and evaluation files are included. The interfaces are ready to use after dependencies download. Python packages and downloadable OCR models are excluded from Git.

## How it works

Sentence Transformers embeds each question, and FAISS retrieves relevant passages. Arabic answers use Ollama with `llama3.2:3b` and cite the retrieved evidence. Expand the source passages to verify an answer.

Arabic uses `paraphrase-multilingual-MiniLM-L12-v2`; English uses `all-MiniLM-L6-v2`. The models run locally once downloaded. Both interfaces listen on localhost.

## More information

- [Arabic interface](Arabic-RAG/README.md)
- [English interface](techqa-rag/README.md)
- [Dependencies, manual setup, OCR, and rebuilding indexes](REQUIREMENTS.md)

## Troubleshooting

- **Python not found:** install Python and reopen the terminal, or try `py -3 run.py` on Windows.
- **Ollama not found:** install Ollama and rerun the launcher. Choose English to search without it.
- **Downloads fail:** check your internet connection and rerun the launcher. Completed installations are reused.
- **Port in use:** stop the previous launcher or the app using port 8000/8001.
- **Missing index:** follow the rebuild steps in REQUIREMENTS.md.
- **Browser does not open:** visit the address printed by the launcher.
