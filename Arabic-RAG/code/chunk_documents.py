import json
from pathlib import Path
import re

from sentence_transformers import SentenceTransformer

project_path = Path(__file__).resolve().parent.parent
input_folder = project_path / "data" / "cleaned"
output_path = project_path / "data" / "chunks.json"

model_name = (
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
model = SentenceTransformer(model_name, device="cpu")
tokenizer = model.tokenizer

target_tokens = 110
overlap_tokens = 20


def count_tokens(text):
    # Include special tokens when checking the model's input limit.
    return len(
        tokenizer(
            text,
            add_special_tokens=True,
            truncation=False,
        )["input_ids"]
    )


chunks = []

for page_number in range(11, 26):
    input_path = input_folder / f"page_{page_number:03d}.txt"
    text = input_path.read_text(encoding="utf-8")

    # Locate complete whitespace-separated words in the original text.
    # Character positions let us preserve spelling and internal formatting.
    words = list(re.finditer(r"\S+", text))

    if not words:
        raise ValueError(f"Empty cleaned page: {input_path.name}")

    start = 0
    page_chunk_count = 0

    while start < len(words):
        end = start
        chunk_text = ""
        token_count = 0

        # Add complete words while staying within the target token count.
        while end < len(words):
            candidate = text[
                words[start].start():words[end].end()
            ]
            candidate_count = count_tokens(candidate)

            if candidate_count > target_tokens:
                break

            chunk_text = candidate
            token_count = candidate_count
            end += 1

        if end == start:
            raise ValueError(
                f"A single word exceeds the chunk target in "
                f"{input_path.name}. Inspect the OCR text."
            )

        if token_count > model.max_seq_length:
            raise ValueError("Chunk exceeds the model input limit.")

        page_chunk_count += 1

        chunks.append({
            "chunk_id": (
                f"page_{page_number:03d}_chunk_"
                f"{page_chunk_count:03d}"
            ),
            "source_file": "artificial_intelligence_ar.pdf",
            "pdf_page": page_number,
            "text": chunk_text,
            "token_count": token_count,
            "start_char": words[start].start(),
            "end_char": words[end - 1].end(),
        })

        if end == len(words):
            break

        # Reuse trailing whole words, up to approximately 20 tokens.
        # Advance by at least one word to guarantee progress.
        next_start = end

        while next_start > start + 1:
            candidate_start = next_start - 1
            overlap_text = text[
                words[candidate_start].start():words[end - 1].end()
            ]

            overlap_count = len(
                tokenizer(
                    overlap_text,
                    add_special_tokens=False,
                    truncation=False,
                )["input_ids"]
            )

            if overlap_count > overlap_tokens:
                break

            next_start = candidate_start

        start = next_start

    print(f"{input_path.name}: {page_chunk_count} chunks")

# Replace the previous one-page chunks with the expanded collection.
output_path.write_text(
    json.dumps(chunks, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

print(f"\nTotal chunks: {len(chunks)}")
print(f"Largest chunk: {max(c['token_count'] for c in chunks)} tokens")
print(f"Model limit: {model.max_seq_length} tokens")
print(f"Saved to: {output_path}")