import html
import re
from pathlib import Path


project_path = Path(__file__).resolve().parent.parent

input_folder = project_path / "data" / "documents"
output_folder = project_path / "data" / "cleaned"

output_folder.mkdir(parents=True, exist_ok=True)


def clean_text(text):
    """Remove formatting noise while preserving technical content."""

    # Decode HTML entities, for example &amp; becomes &.
    text = html.unescape(text)

    # Use consistent line endings across Windows and other platforms.
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Replace non-breaking spaces with ordinary spaces.
    text = text.replace("\u00a0", " ")

    # Remove invisible direction markers and the byte-order mark.
    # Keep ordinary punctuation and configuration symbols.
    text = re.sub(
        r"[\u200e\u200f\u202a-\u202e\u2066-\u2069\ufeff]",
        "",
        text,
    )

    cleaned_lines = []

    for line in text.split("\n"):
        # Remove trailing spaces, but preserve leading indentation.
        # Indentation can help readers understand configuration examples.
        line = line.rstrip()

        # Remove empty bullets, preserving bullets that contain text.
        if line.strip() in {"*", "-", "•"}:
            continue

        # Remove the standalone extraction label, not document headings.
        if line.strip() == "Text:":
            continue

        cleaned_lines.append(line)

    text = "\n".join(cleaned_lines)

    # Keep paragraph separation, but remove excessive blank lines.
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip() + "\n"


document_files = sorted(input_folder.glob("*.txt"))

if not document_files:
    raise FileNotFoundError("No TXT files found in data/documents.")

for document_path in document_files:
    original_text = document_path.read_text(encoding="utf-8")
    cleaned_text = clean_text(original_text)

    if not cleaned_text.strip():
        raise ValueError(f"Cleaning emptied: {document_path.name}")

    # Save a separate cleaned copy using the original filename.
    output_path = output_folder / document_path.name
    output_path.write_text(cleaned_text, encoding="utf-8")

    print(
        f"{document_path.name}: "
        f"{len(original_text)} → {len(cleaned_text)} characters"
    )

print(f"\nCleaned documents saved: {len(document_files)}")