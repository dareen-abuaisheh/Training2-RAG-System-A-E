import json
from pathlib import Path

import faiss
import numpy as np


project_path = Path(__file__).resolve().parent.parent
embeddings_folder = project_path / "data" / "embeddings"
index_folder = project_path / "data" / "index"

index_folder.mkdir(parents=True, exist_ok=True)

# Load the vectors and their matching chunk information.
vectors = np.load(
    embeddings_folder / "vectors.npy",
    allow_pickle=False,
)

with (embeddings_folder / "metadata.json").open(
    encoding="utf-8"
) as file:
    metadata = json.load(file)

# FAISS expects a contiguous array of 32-bit floating-point numbers.
vectors = np.ascontiguousarray(vectors, dtype=np.float32)

if vectors.ndim != 2 or vectors.shape[0] == 0:
    raise ValueError("Expected a non-empty matrix of vectors.")

if vectors.shape[0] != len(metadata["chunks"]):
    raise ValueError("Vector count does not match chunk count.")

if vectors.shape[1] != metadata["dimensions"]:
    raise ValueError("Vector dimensions do not match metadata.")

if not np.isfinite(vectors).all():
    raise ValueError("Vectors contain invalid numerical values.")

# Confirm that vectors have length 1 for cosine similarity search.
lengths = np.linalg.norm(vectors, axis=1)

if not np.allclose(lengths, 1.0, atol=1e-5):
    raise ValueError("Expected normalized vectors.")

# Each vector has 384 dimensions for our embedding model.
dimensions = vectors.shape[1]

# This exact-search index does not require training.
index = faiss.IndexFlatIP(dimensions)

# Insertion order determines the returned search positions.
# Position 0 maps to metadata["chunks"][0], and so on.
index.add(vectors)

index_path = index_folder / "chunks.faiss"
faiss.write_index(index, str(index_path))

# Save matching metadata alongside the index.
(index_folder / "metadata.json").write_text(
    json.dumps(metadata, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

# Confirm the saved index can be loaded with the expected size.
saved_index = faiss.read_index(str(index_path))

if (
    saved_index.ntotal != len(metadata["chunks"])
    or saved_index.d != dimensions
):
    raise ValueError("Saved index does not match the input vectors.")

print(f"Vectors indexed: {saved_index.ntotal}")
print(f"Dimensions: {saved_index.d}")
print(f"Saved index: {index_path}")