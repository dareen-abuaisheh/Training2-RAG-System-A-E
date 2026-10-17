"""Simple local web interface for Arabic-RAG."""
import argparse
import json
import logging
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock

from ask_question import generate_answer
import faiss
from sentence_transformers import SentenceTransformer
from retrieve import retrieve

PROJECT = Path(__file__).resolve().parent.parent


class ArabicRAG:
    def __init__(self):
        folder = PROJECT / "data" / "index"
        self.metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
        self.index = faiss.read_index(str(folder / "chunks.faiss"))
        if (not self.index.ntotal or self.index.ntotal != len(self.metadata["chunks"])
                or self.index.d != self.metadata["dimensions"]):
            raise ValueError("Index and metadata are empty or inconsistent.")
        self.model = SentenceTransformer(self.metadata["model_name"], device="cpu")
        self.lock = Lock()

    def ask(self, question):
        if not isinstance(question, str) or not question.strip():
            raise ValueError("يرجى كتابة سؤال.")
        with self.lock:
            results = retrieve(question, self.model, self.index, self.metadata, k=3)
        return {"answer": generate_answer(question, results), "sources": results}


def handler_for(rag):
    class Handler(BaseHTTPRequestHandler):
        def respond(self, status, payload, content_type="application/json; charset=utf-8"):
            body = payload if isinstance(payload, bytes) else json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/":
                return self.respond(200, (PROJECT / "web" / "index.html").read_bytes(), "text/html; charset=utf-8")
            self.respond(404, {"error": "الصفحة غير موجودة."})

        def do_POST(self):
            if self.path != "/api/ask":
                return self.respond(404, {"error": "الصفحة غير موجودة."})
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 16000:
                    raise ValueError("يرجى إدخال سؤال أقصر.")
                payload = json.loads(self.rfile.read(length))
                if not isinstance(payload, dict):
                    raise ValueError("طلب غير صالح.")
                self.respond(200, rag.ask(payload.get("question")))
            except (ValueError, UnicodeDecodeError) as error:
                self.respond(400, {"error": str(error)})
            except Exception:
                logging.exception("Answer generation failed")
                self.respond(503, {"error": "تعذّر إنشاء الإجابة. تأكد من تشغيل Ollama وتوفّر النموذج llama3.2:3b ثم حاول مجددًا."})
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()
    print("Loading Arabic-RAG index and embedding model...", flush=True)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler_for(ArabicRAG()))
    print(f"Open http://127.0.0.1:{args.port} (Ctrl+C to stop).", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
