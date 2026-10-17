# Training2 RAG System — Arabic & English

Two local document question-answering projects built with Sentence Transformers,
FAISS, and Ollama. Retrieval selects relevant passages; answer generation uses
those passages as evidence and cites their source labels.

| Project | Documents | Browser interface | Port |
| --- | --- | --- | --- |
| Arabic-RAG | Arabic PDF processed with OCR | Arabic questions, generated answers, expandable source passages and PDF page numbers | 8001 |
| techqa-rag | English TechQA documents | Retrieves three distinct documents and displays their full cleaned text | 8000 |

Both projects also provide command-line retrieval and answer generation.
The TechQA browser interface performs retrieval only; its command-line
`ask_question.py` generates answers through Ollama.

## Install and run

Use Python 3.11 or newer. The commands below use **Git Bash on Windows** and
must be run from the repository root. For all installation commands, optional
OCR dependencies, and complete rebuild instructions, see [REQUIREMENTS.md](REQUIREMENTS.md).

```bash
git clone https://github.com/dareen-abuaisheh/Training2-RAG-System-A-E.git
cd Training2-RAG-System-A-E
python -m venv .venv
source .venv/Scripts/activate
python -m pip install -r requirements.txt
```

Install [Ollama](https://ollama.com/download/windows), open it, and download
the answer model:

```bash
ollama pull llama3.2:3b
```

The existing datasets, processed passages, embeddings, and FAISS indexes are
included. You can launch the interfaces without rebuilding them. Embedding
models download automatically on first use, so the first launch needs internet
access and can take longer.

In one terminal:

```bash
source .venv/Scripts/activate
python Arabic-RAG/code/app.py
```

Open **http://127.0.0.1:8001**. Ask an Arabic question, read the answer, and
expand the sources to inspect the supporting passages. Ollama must be running.

In a second terminal, from the repository root:

```bash
source .venv/Scripts/activate
python techqa-rag/code/app.py
```

Open **http://127.0.0.1:8000**. Enter an English question and select a result
to read its cleaned document. This interface does not require Ollama.

Stop either server with **Ctrl+C**. Choose another port with `--port 8002`.
Both servers listen only on the local computer.

## Command-line use

```bash
python Arabic-RAG/code/retrieve.py
python Arabic-RAG/code/ask_question.py
python Arabic-RAG/code/evaluate.py

python techqa-rag/code/retrieve.py
python techqa-rag/code/ask_question.py
```

## Project structure

Each project has a `code/` folder for processing and retrieval, a `data/`
folder for source and generated artifacts, and a `web/` folder for its interface.
Arabic evaluation questions and results are in `Arabic-RAG/data/evaluation/`;
TechQA evaluation questions are in `techqa-rag/evaluation/`.

Arabic embeddings use `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.
English embeddings use `sentence-transformers/all-MiniLM-L6-v2`.
Both answer-generation scripts use `llama3.2:3b` through Ollama at
`http://localhost:11434`.

## Troubleshooting

- **Cannot connect to Ollama:** open Ollama or run `ollama serve` in a separate
  terminal if it is not already running. Check `ollama list` for `llama3.2:3b`.
- **Missing or inconsistent index:** follow the rebuild commands in
  [REQUIREMENTS.md](REQUIREMENTS.md), keeping each project's embeddings and index together.
- **Port already in use:** start the relevant app with `--port 8002`.
- **OCR executable missing:** the Arabic OCR script expects
  `C:\Program Files\Tesseract-OCR\tesseract.exe`; adjust that script if your
  installation uses a different location.

Virtual environments, Python caches, and downloadable OCR models are excluded
from Git. Installation and download commands recreate those dependencies.
