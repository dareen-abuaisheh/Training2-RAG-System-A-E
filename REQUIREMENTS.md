# Requirements and commands

Run commands from the repository root in **Git Bash on Windows** unless a
block is marked PowerShell. Python 3.11+ and Git must already be installed.

## Python dependencies

```bash
python -m venv .venv
source .venv/Scripts/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
```

`requirements.txt` installs FAISS, NumPy, Sentence Transformers, and PyMuPDF.
Sentence Transformers installs its own dependencies, including PyTorch.
No Node.js, frontend build tool, or extra UI package is required.
These requirements are unpinned; they are not an exact environment lockfile.

For PowerShell, activate with `.\.venv\Scripts\Activate.ps1` instead.
For Linux/macOS, activate with `source .venv/bin/activate`; Arabic OCR also
requires updating the Windows executable path in `ocr_pages.py`.

## Ollama: generated answers

Download and install [Ollama for Windows](https://ollama.com/download/windows).
From Git Bash you can download and launch its installer:

```bash
curl -fL https://ollama.com/download/OllamaSetup.exe -o "$TEMP/OllamaSetup.exe"
"$TEMP/OllamaSetup.exe"
```

Complete the installer, reopen your terminal, and open Ollama. Then:

```bash
ollama pull llama3.2:3b
ollama list
```

If Ollama is not already running, use a separate terminal:

```bash
ollama serve
```

Ollama is needed for the Arabic web interface, both `ask_question.py` scripts,
and Arabic evaluation. The TechQA web interface needs only Python dependencies.

## Start both interfaces

Terminal 1:

```bash
source .venv/Scripts/activate
python Arabic-RAG/code/app.py
# http://127.0.0.1:8001
```

Terminal 2:

```bash
source .venv/Scripts/activate
python techqa-rag/code/app.py
# http://127.0.0.1:8000
```

## Optional Arabic OCR dependencies

The included OCR text and indexes allow normal use without installing Tesseract.
Install it only to repeat OCR from the original PDF. In **PowerShell**:

```powershell
winget install --id UB-Mannheim.TesseractOCR --exact
```

The script expects `C:\Program Files\Tesseract-OCR\tesseract.exe`.
Check the installed location and adjust `Arabic-RAG/code/ocr_pages.py` if needed.
Download the excluded Arabic OCR model in **Git Bash**, from the repository root:

```bash
mkdir -p Arabic-RAG/models/tessdata_best
curl -fL https://raw.githubusercontent.com/tesseract-ocr/tessdata_best/main/ara.traineddata -o Arabic-RAG/models/tessdata_best/ara.traineddata
```

Sources: [Windows Tesseract package](https://github.com/microsoft/winget-pkgs/tree/master/manifests/u/UB-Mannheim/TesseractOCR),
[Tesseract model documentation](https://tesseract-ocr.github.io/tessdoc/Data-Files.html).

## Rebuild Arabic data and index

The supplied PDF is at `Arabic-RAG/data/original/artificial_intelligence_ar.pdf`.
After installing OCR dependencies, run in this order. These scripts regenerate
their outputs. The current page range is PDF pages 11–26, set in the scripts.

```bash
python Arabic-RAG/code/render_page.py
python Arabic-RAG/code/ocr_pages.py
python Arabic-RAG/code/clean_ocr.py
python Arabic-RAG/code/chunk_documents.py
python Arabic-RAG/code/generate_embeddings.py
python Arabic-RAG/code/build_index.py
```

To rebuild from the included cleaned text, start at `chunk_documents.py`.

## Rebuild English data and index

The supplied TechQA dataset is at `techqa-rag/data/original/train.json`.

```bash
python techqa-rag/code/prepare_data.py
python techqa-rag/code/clean_documents.py
python techqa-rag/code/chunk_documents.py
python techqa-rag/code/generate_embeddings.py
python techqa-rag/code/build_index.py
```

## Command-line retrieval, answers, and evaluation

```bash
python Arabic-RAG/code/retrieve.py
python Arabic-RAG/code/ask_question.py
python Arabic-RAG/code/evaluate.py
python techqa-rag/code/retrieve.py
python techqa-rag/code/ask_question.py
```

Embedding models download automatically on first use and are cached outside the
repository. Downloaded Python dependencies, virtual environments, and the Arabic
OCR model are excluded from Git; project data and generated results are included.
