import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


project_path = Path(__file__).resolve().parent.parent
chunks_path = project_path / "data" / "chunks.json"
output_folder = project_path / "data" / "embeddings"

output_folder.mkdir(parents=True, exist_ok=True)

with chunks_path.open(encoding="utf-8") as file:
    chunks = json.load(file)

if not chunks:
    raise ValueError("The chunks file is empty.")

model_name = "sentence-transformers/all-MiniLM-L6-v2"
model = SentenceTransformer(model_name, device="cpu")

# Keep texts in the same order as their chunk records.
# Embedding row 0 will correspond to chunk record 0.
texts = [chunk["text"] for chunk in chunks]

# Check actual token lengths before encoding to prevent truncation.
for chunk in chunks:
    token_count = len(model.tokenizer(
        chunk["text"],
        add_special_tokens=True,
        truncation=False,
    )["input_ids"])

    if token_count > model.max_seq_length:
        raise ValueError(f"Chunk exceeds limit: {chunk['chunk_id']}")

# Process eight chunks at a time.
# Normalization gives each vector length 1 for similarity comparison.
embeddings = model.encode(
    texts,
    batch_size=8,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True,
)

# Validate that every chunk produced a finite numerical vector.
if embeddings.shape[0] != len(chunks):
    raise ValueError("Embedding count does not match chunk count.")

if not np.isfinite(embeddings).all():
    raise ValueError("Embeddings contain invalid numerical values.")

# Save the numerical vectors in NumPy's binary format.
np.save(output_folder / "vectors.npy", embeddings)

# Save an accompanying snapshot so vectors remain linked to their text.
metadata = {
    "model_name": model_name,
    "normalized": True,
    "dimensions": embeddings.shape[1],
    "chunks": chunks,
}

(output_folder / "metadata.json").write_text(
    json.dumps(metadata, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

print(f"\nEmbedding shape: {embeddings.shape}")
print(f"Saved embeddings to: {output_folder}")