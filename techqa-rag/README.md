# TechQA document search

A simple local interface over the existing Sentence Transformers / FAISS index.
Enter a question, see the top three **distinct documents**, and click a result
to read its complete cleaned source text. Each document is ranked by its most
similar chunk. Similarity is a ranking score, not a probability of correctness.

## Run on this computer

From the `Internship II` folder in PowerShell:

```powershell
.\.venv\Scripts\python.exe .\techqa-rag\code\app.py
```

Then open http://127.0.0.1:8000. Stop the server with Ctrl+C.
Use `--port 8001` if port 8000 is occupied.

The existing virtual environment already contains the required `faiss-cpu`,
`numpy`, and `sentence-transformers` packages. No UI packages are required.
The embedding model must be cached locally or downloadable on first launch.
The server loads the model once and listens only on the local computer.

## Project structure

- `code/prepare_data.py`, `clean_documents.py`, `chunk_documents.py`: prepare source documents and passages.
- `code/generate_embeddings.py`, `build_index.py`: create embeddings and the FAISS index.
- `code/retrieve.py`: original chunk retrieval, reused by the interface.
- `code/ask_question.py`: existing command-line answer generation through Ollama.
- `code/app.py`: local HTTP server and distinct-document ranking.
- `web/index.html`: question form, search results, and document viewer.
- `data/cleaned/`: full documents displayed by the viewer.
- `data/index/`: existing index and matching metadata.

The browser interface performs document retrieval; it does not require Ollama.
