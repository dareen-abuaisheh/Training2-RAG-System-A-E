# Arabic-RAG interface

A simple Arabic, right-to-left browser interface for the existing RAG pipeline.
Ask a question, read the generated answer, and expand the three retrieved source
passages to inspect the evidence and PDF page numbers.

From the `Internship II` folder in PowerShell:

```powershell
.\.venv\Scripts\python.exe .\Arabic-RAG\code\app.py
```

Open http://127.0.0.1:8001. Use `--port 8002` to choose another port.
Stop with Ctrl+C.

The application uses the existing FAISS index in `data/index`, the existing
Sentence Transformers embedding model, and Ollama at `localhost:11434` with
`llama3.2:3b`. Start Ollama and ensure that model is installed before asking a
question. The embedding model must be cached or downloadable on first launch.
No additional UI packages are needed. The server listens on localhost only.
