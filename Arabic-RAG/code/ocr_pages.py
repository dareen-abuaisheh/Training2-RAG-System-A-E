from pathlib import Path
import subprocess

project_path = Path(__file__).resolve().parent.parent

tesseract_path = Path(
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

images_folder = project_path / "data" / "page_images"
output_folder = project_path / "data" / "ocr"
model_folder = project_path / "models" / "tessdata_best"

first_page = 11
last_page = 26

# Check requirements before processing any pages.
if not tesseract_path.is_file():
    raise FileNotFoundError(f"Tesseract not found: {tesseract_path}")

if not (model_folder / "ara.traineddata").is_file():
    raise FileNotFoundError("The tessdata_best Arabic model is missing.")

image_paths = [
    images_folder / f"page_{number:03d}.png"
    for number in range(first_page, last_page + 1)
]

for image_path in image_paths:
    if not image_path.is_file():
        raise FileNotFoundError(f"Page image missing: {image_path}")

output_folder.mkdir(parents=True, exist_ok=True)

for image_path in image_paths:
    # Tesseract adds .txt to this output base automatically.
    output_base = output_folder / image_path.stem
    text_path = output_base.with_suffix(".txt")

    # Reuse completed results when restarting the script.
    if text_path.exists() and text_path.stat().st_size > 0:
        print(f"Already exists: {text_path.name}")
        continue

    print(f"Processing: {image_path.name}", flush=True)

    # Pass arguments separately to handle spaces in Windows paths.
    result = subprocess.run(
        [
            str(tesseract_path),
            str(image_path),
            str(output_base),
            "--tessdata-dir", str(model_folder),
            "-l", "ara",
            "--oem", "1",
            "--psm", "6",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"OCR failed for {image_path.name}:\n{result.stderr}"
        )

    if not text_path.is_file():
        raise RuntimeError(f"OCR output missing: {text_path}")

    extracted_text = text_path.read_text(encoding="utf-8")

    if not extracted_text.strip():
        print(f"WARNING: No text recognized in {image_path.name}")
    else:
        print(
            f"Saved: {text_path.name} "
            f"({len(extracted_text.split())} words)"
        )

print("OCR processing finished.")