"""Local browser interface for the existing TechQA retrieval pipeline."""
import argparse
import json
import logging
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock
from urllib.parse import parse_qs, urlparse

import faiss
from sentence_transformers import SentenceTransformer

from retrieve import retrieve

PROJECT = Path(__file__).resolve().parent.parent


class DocumentSearch:
    def __init__(self):
        folder = PROJECT / "data" / "index"
        self.metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
        self.index = faiss.read_index(str(folder / "chunks.faiss"))
        if (not self.index.ntotal or self.index.ntotal != len(self.metadata["chunks"])
                or self.index.d != self.metadata["dimensions"]):
            raise ValueError("Index and metadata are empty or inconsistent.")
        self.sources = {chunk["source_file"] for chunk in self.metadata["chunks"]}
        self.model = SentenceTransformer(self.metadata["model_name"], device="cpu")
        self.lock = Lock()

    def document(self, source):
        # Only indexed filenames are exposed; never accept arbitrary file paths.
        if source not in self.sources or Path(source).name != source:
            raise FileNotFoundError("Document not found.")
        return (PROJECT / "data" / "cleaned" / source).read_text(encoding="utf-8")

    def search(self, question):
        if not isinstance(question, str):
            raise ValueError("Enter a question as text.")
        with self.lock:
            chunks = retrieve(question, self.model, self.index, self.metadata, k=self.index.ntotal)
        # The highest scoring passage determines each document's rank.
        results, seen = [], set()
        for chunk in chunks:
            source = chunk["source_file"]
            if source in seen:
                continue
            seen.add(source)
            text = self.document(source)
            title = next((line.strip() for line in text.splitlines() if line.strip()), source)
            results.append({**chunk, "title": title})
            if len(results) == 3:
                break
        return results


def handler_for(search):
    class Handler(BaseHTTPRequestHandler):
        def respond(self, status, payload, content_type="application/json; charset=utf-8"):
            body = payload if isinstance(payload, bytes) else json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            route = urlparse(self.path)
            if route.path == "/":
                return self.respond(200, (PROJECT / "web" / "index.html").read_bytes(), "text/html; charset=utf-8")
            if route.path == "/api/document":
                source = parse_qs(route.query).get("source", [""])[0]
                try:
                    return self.respond(200, {"source_file": source, "text": search.document(source)})
                except FileNotFoundError:
                    return self.respond(404, {"error": "Document not found."})
            self.respond(404, {"error": "Not found."})

        def do_POST(self):
            if self.path != "/api/search":
                return self.respond(404, {"error": "Not found."})
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 16000:
                    raise ValueError("Enter a shorter question.")
                payload = json.loads(self.rfile.read(length))
                if not isinstance(payload, dict):
                    raise ValueError("Expected a question object.")
                results = search.search(payload.get("question", ""))
                self.respond(200, {"results": results})
            except (ValueError, UnicodeDecodeError) as error:
                self.respond(400, {"error": str(error)})
            except Exception:
                logging.exception("Search failed")
                self.respond(500, {"error": "Search failed. Check that the indexed documents exist in data/cleaned."})

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    print("Loading the document index and embedding model...", flush=True)
    search = DocumentSearch()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler_for(search))
    print(f"Open http://127.0.0.1:{args.port} — press Ctrl+C to stop.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
