"""Set up and launch the local RAG interfaces using only Python's standard library."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.request import urlopen
import venv
import webbrowser

ROOT = Path(__file__).resolve().parent
ENV = ROOT / ".venv"
PYTHON = ENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
APPS = {"arabic": ("Arabic-RAG", 8001), "english": ("techqa-rag", 8000)}


def setup():
    if sys.version_info < (3, 11):
        raise RuntimeError("Install Python 3.11 or newer, then run this launcher again.")
    if not PYTHON.exists():
        print("Creating the Python environment...", flush=True)
        venv.EnvBuilder(with_pip=True).create(ENV)
    requirements = ROOT / "requirements.txt"
    fingerprint = hashlib.sha256(requirements.read_bytes()).hexdigest()
    marker = ENV / ".rag-requirements"
    if not marker.exists() or marker.read_text() != fingerprint:
        print("Installing dependencies (first launch can take several minutes)...", flush=True)
        subprocess.run([str(PYTHON), "-m", "pip", "install", "-r", str(requirements)], check=True)
        marker.write_text(fingerprint)
    if Path(sys.prefix).resolve() != ENV.resolve():
        return subprocess.call([str(PYTHON), str(ROOT / "run.py"), *sys.argv[1:]], cwd=ROOT)
    return None


def ollama_ready():
    try:
        with urlopen("http://127.0.0.1:11434/api/tags", timeout=2) as response:
            return json.load(response)
    except (OSError, ValueError):
        return None


def prepare_ollama(children):
    status = ollama_ready()
    executable = shutil.which("ollama")
    if not executable and os.name == "nt":
        candidate = Path(os.environ.get("LOCALAPPDATA", "")) / "Programs/Ollama/ollama.exe"
        if candidate.is_file():
            executable = str(candidate)
    if status is None:
        if not executable:
            raise RuntimeError("Arabic answers require Ollama. Install it from https://ollama.com/download, then run again. You can choose English without Ollama.")
        print("Starting Ollama...", flush=True)
        children.append(subprocess.Popen([executable, "serve"], stdout=subprocess.DEVNULL,
                                         stderr=subprocess.DEVNULL))
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            status = ollama_ready()
            if status is not None:
                break
            time.sleep(.5)
        if status is None:
            raise RuntimeError("Ollama did not start. Open Ollama and try again.")
    names = {item.get("name") for item in status.get("models", [])}
    if "llama3.2:3b" not in names:
        if not executable:
            raise RuntimeError("Download the answer model using: ollama pull llama3.2:3b")
        print("Downloading llama3.2:3b (first Arabic launch only)...", flush=True)
        subprocess.run([executable, "pull", "llama3.2:3b"], check=True)


def launch(selected, open_browser=True):
    children = []
    try:
        # Check every required port before starting services.
        for name in selected:
            port = APPS[name][1]
            with socket.socket() as probe:
                try:
                    probe.bind(("127.0.0.1", port))
                except OSError:
                    raise RuntimeError(f"Port {port} is in use. Close the other app or previous launcher and try again.")
        if "arabic" in selected:
            prepare_ollama(children)
        pending = []
        for name in selected:
            folder, port = APPS[name]
            if not (ROOT / folder / "data/index/chunks.faiss").is_file():
                raise RuntimeError(f"Missing {folder} index. See REQUIREMENTS.md to rebuild it.")
            process = subprocess.Popen([sys.executable, str(ROOT / folder / "code/app.py")], cwd=ROOT)
            children.append(process)
            pending.append((process, f"http://127.0.0.1:{port}"))
        print("Loading embedding models; first use downloads them. Keep this window open.", flush=True)
        while pending:
            for process, url in pending[:]:
                if process.poll() is not None:
                    raise RuntimeError("An interface stopped during startup. Check the error above.")
                try:
                    with urlopen(url, timeout=1) as response:
                        if response.status != 200:
                            continue
                    print(f"Ready: {url}", flush=True)
                    if open_browser:
                        webbrowser.open(url)
                    pending.remove((process, url))
                except (OSError, URLError):
                    pass
            time.sleep(.5)
        print("Press Ctrl+C to stop.", flush=True)
        while all(child.poll() is None for child in children):
            time.sleep(1)
    finally:
        for child in reversed(children):
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()


def main():
    parser = argparse.ArgumentParser(description="Install dependencies and open the RAG interfaces.")
    parser.add_argument("project", nargs="?", choices=["arabic", "english", "both"])
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    try:
        result = setup()
        if result is not None:
            return result
        choice = args.project
        if choice is None:
            print("\nRAG Document Assistant\n1. Arabic\n2. English\n3. Both")
            answer = input("Choose [1/2/3] (default: 3): ").strip()
            choice = {"1": "arabic", "2": "english", "3": "both", "": "both"}.get(answer)
            if choice is None:
                raise RuntimeError("Choose 1, 2, or 3 and try again.")
        launch(list(APPS) if choice == "both" else [choice], not args.no_browser)
    except KeyboardInterrupt:
        print("\nStopped.")
    except (RuntimeError, OSError, subprocess.CalledProcessError) as error:
        print(f"\nCould not start: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
