import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


def retrieve(question, model, index, metadata, k=3):
    """Return the k most similar chunks for a question."""

    question = question.strip()

    if not question:
        raise ValueError("The question cannot be empty.")

    if k < 1:
        raise ValueError("k must be at least 1.")

    # Avoid silently truncating a long question.
    tokens = model.tokenizer(
        question,
        add_special_tokens=True,
        truncation=False,
    )["input_ids"]

    if len(tokens) > model.max_seq_length:
        raise ValueError("Question is too long. Please shorten it.")

    # Use the same model and normalization as the document embeddings.
    # Passing a list creates a matrix with one row.
    query_vector = model.encode(
        [question],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    query_vector = np.ascontiguousarray(
        query_vector,
        dtype=np.float32,
    )

    # Never request more results than the index contains.
    k = min(k, index.ntotal)

    # FAISS returns similarity scores and vector positions.
    scores, positions = index.search(query_vector, k)

    results = []

    for score, position in zip(scores[0], positions[0]):
        if position < 0:
            continue

        # Translate the vector position back into its source chunk.
        chunk = metadata["chunks"][int(position)]

        results.append({
            "chunk_id": chunk["chunk_id"],
            "source_file": chunk["source_file"],
            "score": float(score),
            "text": chunk["text"],
        })

    return results


def main():
    project_path = Path(__file__).resolve().parent.parent
    index_folder = project_path / "data" / "index"

    index = faiss.read_index(str(index_folder / "chunks.faiss"))

    with (index_folder / "metadata.json").open(
        encoding="utf-8"
    ) as file:
        metadata = json.load(file)

    # Verify that the index matches its accompanying metadata.
    if index.ntotal == 0:
        raise ValueError("The index is empty.")

    if (
        index.ntotal != len(metadata["chunks"])
        or index.d != metadata["dimensions"]
    ):
        raise ValueError("Index and metadata do not match.")

    model = SentenceTransformer(
        metadata["model_name"],
        device="cpu",
    )

    question = input("\nEnter your question: ")

    results = retrieve(
        question,
        model,
        index,
        metadata,
        k=3,
    )

    for rank, result in enumerate(results, start=1):
        print(f"\nResult {rank}")
        print(f"Source: {result['source_file']}")
        print(f"Chunk: {result['chunk_id']}")
        print(f"Similarity: {result['score']:.4f}")
        print(f"Text:\n{result['text']}")


# Run the interactive part only when this file is executed directly.
# Later, another script can import retrieve() without starting it.
if __name__ == "__main__":
    main()