from pathlib import Path
import re
import unicodedata

project_path = Path(__file__).resolve().parent.parent

input_folder = project_path / "data" / "ocr"
output_folder = project_path / "data" / "cleaned"


def clean_text(text):
    """Clean formatting without guessing corrections to Arabic words."""
    text = unicodedata.normalize("NFC", text)

    # Standardize line endings and spaces.
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00a0", " ")

    # Remove invisible direction controls and the byte-order mark.
    text = re.sub(
        r"[\u200e\u200f\u202a-\u202e\u2066-\u2069\ufeff]",
        "",
        text,
    )

    # Remove isolated vertical bars observed in this document's OCR.
    text = re.sub(r"(?<!\S)\|+(?!\S)", "", text)

    # Preserve line breaks while removing unnecessary spaces.
    text = "\n".join(
        re.sub(r"[ \t]+", " ", line).strip()
        for line in text.split("\n")
    )

    return re.sub(r"\n{3,}", "\n\n", text).strip()


# Process only the selected non-blank pages.
# Stop if a required page is missing or contains no recognized text.
cleaned_pages = []

for page_number in range(11, 26):
    input_path = input_folder / f"page_{page_number:03d}.txt"
    raw_text = input_path.read_text(encoding="utf-8")

    if not raw_text.strip():
        raise ValueError(f"Unexpected empty OCR file: {input_path.name}")

    cleaned_text = clean_text(raw_text)

    if not cleaned_text:
        raise ValueError(f"Cleaning produced empty text: {input_path.name}")

    cleaned_pages.append((input_path.name, cleaned_text))

# Save separate copies, preserving the raw OCR files.
output_folder.mkdir(parents=True, exist_ok=True)

for filename, text in cleaned_pages:
    output_path = output_folder / filename
    output_path.write_text(text + "\n", encoding="utf-8")
    print(f"Cleaned: {filename}")

print(f"Total pages cleaned: {len(cleaned_pages)}")