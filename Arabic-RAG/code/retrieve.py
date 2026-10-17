import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


def retrieve(question, model, index, metadata, k=3):
    """Return the k chunks most similar to the question."""
    question = question.strip()

    if not question:
        raise ValueError("The question cannot be empty.")

    if k < 1:
        raise ValueError("k must be at least 1.")

    # Avoid silently truncating long questions.
    token_count = len(
        model.tokenizer(
            question,
            add_special_tokens=True,
            truncation=False,
        )["input_ids"]
    )

    if token_count > model.max_seq_length:
        raise ValueError("The question is too long. Please shorten it.")

    # Use the same model and normalization used for document embeddings.
    question_vector = model.encode(
        [question],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    question_vector = np.ascontiguousarray(
        question_vector, dtype=np.float32
    )

    # Search returns scores and vector positions.
    scores, positions = index.search(
        question_vector, min(k, index.ntotal)
    )

    results = []

    for score, position in zip(scores[0], positions[0]):
        if position < 0:
            continue

        # Vector positions match the order of chunks in the metadata.
        chunk = metadata["chunks"][int(position)]

        results.append({
            **chunk,
            "score": float(score),
        })

    return results


def main():
    project_path = Path(__file__).resolve().parent.parent
    index_folder = project_path / "data" / "index"

    index = faiss.read_index(str(index_folder / "chunks.faiss"))
    metadata = json.loads(
        (index_folder / "metadata.json").read_text(encoding="utf-8")
    )

    if index.ntotal == 0:
        raise ValueError("The index is empty.")

    if index.ntotal != len(metadata["chunks"]):
        raise ValueError("Index and metadata counts do not match.")

    if index.d != metadata["dimensions"]:
        raise ValueError("Index and metadata dimensions do not match.")

    model = SentenceTransformer(
        metadata["model_name"], device="cpu"
    )

    question = input("\nEnter your question in Arabic: ")
    results = retrieve(question, model, index, metadata)

    for rank, result in enumerate(results, start=1):
        print(f"\nResult {rank}")
        print(f"Source: {result['source_file']}")
        print(f"PDF page: {result['pdf_page']}")
        print(f"Chunk: {result['chunk_id']}")
        print(f"Similarity: {result['score']:.4f}")
        print(f"Text:\n{result['text']}")


if __name__ == "__main__":
    main()