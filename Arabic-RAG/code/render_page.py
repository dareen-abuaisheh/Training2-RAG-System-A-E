from pathlib import Path

import pymupdf

project_path = Path(__file__).resolve().parent.parent

pdf_path = (
    project_path / "data" / "original"
    / "artificial_intelligence_ar.pdf"
)

output_folder = project_path / "data" / "page_images"
output_folder.mkdir(parents=True, exist_ok=True)

# Inclusive PDF page numbers, starting from 1.
first_page = 11
last_page = 26
dpi = 300

if not pdf_path.is_file():
    raise FileNotFoundError(f"PDF not found: {pdf_path}")

with pymupdf.open(pdf_path) as document:
    if not 1 <= first_page <= last_page <= len(document):
        raise ValueError("The requested page range is invalid.")

    for page_number in range(first_page, last_page + 1):
        output_path = output_folder / f"page_{page_number:03d}.png"

        # Reuse images already generated during our first experiment.
        if output_path.exists():
            print(f"Already exists: {output_path.name}")
            continue

        page = document[page_number - 1]
        image = page.get_pixmap(dpi=dpi, alpha=False)
        image.save(output_path)

        print(f"Saved: {output_path.name}")

print(f"Finished: PDF pages {first_page}–{last_page}")