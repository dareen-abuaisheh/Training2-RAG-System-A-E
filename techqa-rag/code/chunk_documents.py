import json
from pathlib import Path
from sentence_transformers import SentenceTransformer


# The script is inside code/, so go up to the project folder.
project_path = Path(__file__).resolve().parent.parent
documents_path = project_path / "data" / "cleaned"
output_path = project_path / "data" / "chunks.json"

# Use the same tokenizer that will be used for embeddings.
model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)
tokenizer = model.tokenizer

# Measure chunk size and overlap in tokens, not words.
# Leave room for the model's special tokens.
chunk_size = 240
overlap = 40

if not tokenizer.is_fast:
    raise ValueError("A fast tokenizer is required for character offsets.")

if not 0 <= overlap < chunk_size:
    raise ValueError("Overlap must be smaller than chunk size.")

document_files = sorted(documents_path.glob("*.txt"))

if not document_files:
    raise FileNotFoundError("No TXT documents found in data/cleaned/.")

chunks = []

for document_path in document_files:
    text = document_path.read_text(encoding="utf-8")

    # Offsets identify where each token appears in the original text.
    # They let us preserve line breaks and configuration formatting.
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        truncation=False,
        return_offsets_mapping=True,
    )
    offsets = encoded["offset_mapping"]

    if not offsets:
        raise ValueError(f"Empty document: {document_path.name}")

    start = 0
    chunk_number = 1

    while start < len(offsets):
        end = min(start + chunk_size, len(offsets))

        start_char = offsets[start][0]
        end_char = offsets[end - 1][1]
        chunk_text = text[start_char:end_char]

        # Count the actual chunk with special tokens included.
        token_count = len(tokenizer(
            chunk_text,
            add_special_tokens=True,
            truncation=False,
        )["input_ids"])

        # Tokenization can change slightly at a chunk boundary.
        # Shorten the chunk if necessary to guarantee it fits.
        while token_count > model.max_seq_length:
            end -= 1

            if end <= start:
                raise ValueError("Unable to create a chunk within the limit.")

            end_char = offsets[end - 1][1]
            chunk_text = text[start_char:end_char]

            token_count = len(tokenizer(
                chunk_text,
                add_special_tokens=True,
                truncation=False,
            )["input_ids"])

        # Attach source information for retrieval and citations.
        chunks.append({
            "chunk_id": (
                f"{document_path.stem}_chunk_{chunk_number:03d}"
            ),
            "source_file": document_path.name,
            "text": chunk_text,
            "token_count": token_count,
            "word_count": len(chunk_text.split()),
            "start_char": start_char,
            "end_char": end_char,
        })

        # Finish once we reach the last token in this document.
        if end == len(offsets):
            break

        # Repeat the final 40 tokens in the next chunk.
        # Always move forward to avoid an infinite loop.
        start = max(start + 1, end - overlap)
        chunk_number += 1

    print(f"{document_path.name}: {chunk_number} chunks")

# Replace the old word-based chunks with the new token-based chunks.
output_path.write_text(
    json.dumps(chunks, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

print(f"\nTotal chunks: {len(chunks)}")
print(
    "Largest chunk:",
    max(chunk["token_count"] for chunk in chunks),
    "tokens",
)
print(f"Model input limit: {model.max_seq_length} tokens")
print(f"Saved to: {output_path}")