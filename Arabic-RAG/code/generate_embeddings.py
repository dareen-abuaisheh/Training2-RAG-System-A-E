import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

project_path = Path(__file__).resolve().parent.parent

chunks_path = project_path / "data" / "chunks.json"
output_folder = project_path / "data" / "embeddings"

model_name = (
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

chunks = json.loads(chunks_path.read_text(encoding="utf-8"))

if not chunks:
    raise ValueError("No chunks were found.")

model = SentenceTransformer(model_name, device="cpu")

# Check actual text lengths before encoding to avoid silent truncation.
for chunk in chunks:
    token_count = len(
        model.tokenizer(
            chunk["text"],
            add_special_tokens=True,
            truncation=False,
        )["input_ids"]
    )

    if token_count > model.max_seq_length:
        raise ValueError(f"Chunk exceeds model limit: {chunk['chunk_id']}")

# Keep the same order so each vector matches its chunk's metadata.
texts = [chunk["text"] for chunk in chunks]

embeddings = model.encode(
    texts,
    batch_size=8,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True,
)

if not np.isfinite(embeddings).all():
    raise ValueError("Embeddings contain invalid numbers.")

output_folder.mkdir(parents=True, exist_ok=True)

np.save(output_folder / "vectors.npy", embeddings)

metadata = {
    "model_name": model_name,
    "normalized": True,
    "dimensions": int(embeddings.shape[1]),
    "chunks": chunks,
}

(output_folder / "metadata.json").write_text(
    json.dumps(metadata, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

print(f"Embedding shape: {embeddings.shape}")
print(f"Saved to: {output_folder}")