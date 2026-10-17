import json
from pathlib import Path

import faiss
import numpy as np

project_path = Path(__file__).resolve().parent.parent
embeddings_folder = project_path / "data" / "embeddings"
index_folder = project_path / "data" / "index"

# Load the vectors and their matching text/source information.
vectors = np.load(
    embeddings_folder / "vectors.npy",
    allow_pickle=False,
)

metadata = json.loads(
    (embeddings_folder / "metadata.json").read_text(encoding="utf-8")
)

# Validate the files before building the index.
if vectors.ndim != 2 or vectors.shape[0] == 0:
    raise ValueError("Expected a non-empty, two-dimensional vector array.")

if vectors.shape[0] != len(metadata["chunks"]):
    raise ValueError("The vector count does not match the chunk count.")

if vectors.shape[1] != metadata["dimensions"]:
    raise ValueError("Vector dimensions do not match the metadata.")

if not np.isfinite(vectors).all():
    raise ValueError("Vectors contain invalid numbers.")

# FAISS expects contiguous arrays of 32-bit floating-point numbers.
vectors = np.ascontiguousarray(vectors, dtype=np.float32)

# Unit-length vectors make inner product equivalent to cosine similarity.
norms = np.linalg.norm(vectors, axis=1)
if not np.allclose(norms, 1.0, atol=1e-5):
    raise ValueError("Expected normalized vectors.")

# Add vectors in their original order to preserve the metadata mapping.
index = faiss.IndexFlatIP(vectors.shape[1])
index.add(vectors)

index_folder.mkdir(parents=True, exist_ok=True)

faiss.write_index(index, str(index_folder / "chunks.faiss"))

(index_folder / "metadata.json").write_text(
    json.dumps(metadata, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

print(f"Vectors indexed: {index.ntotal}")
print(f"Dimensions: {index.d}")
print(f"Saved index: {index_folder / 'chunks.faiss'}")