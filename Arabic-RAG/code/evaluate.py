import json
import re
from datetime import datetime
from pathlib import Path
from time import perf_counter

import faiss
from sentence_transformers import SentenceTransformer

from ask_question import generate_answer
from retrieve import retrieve


def main():
    project_path = Path(__file__).resolve().parent.parent

    questions_path = (
        project_path / "data" / "evaluation" / "test_questions.json"
    )
    index_folder = project_path / "data" / "index"
    results_folder = project_path / "data" / "evaluation" / "results"

    # utf-8-sig also accepts files saved with a UTF-8 byte-order mark.
    questions = json.loads(
        questions_path.read_text(encoding="utf-8-sig")
    )

    if not isinstance(questions, list) or not questions:
        raise ValueError("Expected a non-empty list of test questions.")

    required_fields = {
        "id", "question", "reference_answer", "source_file", "pdf_page"
    }

    # Validate the test records before starting model requests.
    seen_ids = set()

    for item in questions:
        if not isinstance(item, dict) or not required_fields <= item.keys():
            raise ValueError(
                f"Each question must contain: {sorted(required_fields)}"
            )

        if item["id"] in seen_ids:
            raise ValueError(f"Duplicate question ID: {item['id']}")

        seen_ids.add(item["id"])

        if not isinstance(item["question"], str) or not item["question"].strip():
            raise ValueError(f"Empty question: {item['id']}")

    metadata = json.loads(
        (index_folder / "metadata.json").read_text(encoding="utf-8")
    )
    index = faiss.read_index(str(index_folder / "chunks.faiss"))

    if index.ntotal == 0:
        raise ValueError("The index is empty.")

    if index.ntotal != len(metadata["chunks"]):
        raise ValueError("Index and metadata counts do not match.")

    if index.d != metadata["dimensions"]:
        raise ValueError("Index and metadata dimensions do not match.")

    # Load the embedding model once, not once per question.
    model = SentenceTransformer(
        metadata["model_name"], device="cpu"
    )

    results_folder.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_path = results_folder / f"evaluation_{timestamp}.txt"

    completed = 0
    failed = 0
    page_hits = 0
    citations_found = 0

    # Write each result immediately so completed tests survive interruption.
    with output_path.open("w", encoding="utf-8") as report:
        report.write("ARABIC RAG EVALUATION\n")
        report.write(f"Date: {datetime.now().isoformat(timespec='seconds')}\n")
        report.write(f"Embedding model: {metadata['model_name']}\n")
        report.write("Answer function: ask_question.generate_answer\n")
        report.write(f"Indexed chunks: {index.ntotal}\n")
        report.write(f"Questions: {len(questions)}\n")
        report.write("Top-k: 3\n")
        report.write(
            "Expected-page matches and citation presence are automatic "
            "checks; answer accuracy requires manual review.\n\n"
        )
        report.flush()

        for number, item in enumerate(questions, start=1):
            print(
                f"[{number}/{len(questions)}] Testing {item['id']}...",
                flush=True,
            )

            report.write("=" * 70 + "\n")
            report.write(f"ID: {item['id']}\n")
            report.write(f"Question: {item['question']}\n\n")
            report.write(
                f"Reference answer:\n{item['reference_answer']}\n\n"
            )
            report.write(
                f"Expected source: {item['source_file']} "
                f"| PDF page: {item['pdf_page']}\n\n"
            )

            started = perf_counter()

            try:
                results = retrieve(
                    item["question"], model, index, metadata, k=3
                )
                answer = generate_answer(item["question"], results)

                # Check the expected filename AND page, not just filename.
                # All current chunks belong to the same PDF.
                expected_rank = next(
                    (
                        rank
                        for rank, chunk in enumerate(results, start=1)
                        if chunk["source_file"] == item["source_file"]
                        and chunk["pdf_page"] == item["pdf_page"]
                    ),
                    None,
                )

                # Check for a label that refers to an available result.
                labels = [
                    int(value)
                    for value in re.findall(r"\[S([0-9]+)\]", answer)
                ]
                has_citation = any(
                    1 <= label <= len(results) for label in labels
                )

                completed += 1
                page_hits += int(expected_rank is not None)
                citations_found += int(has_citation)

                report.write(f"Generated answer:\n{answer}\n\n")
                report.write(
                    f"Expected page in top 3: "
                    f"{'YES' if expected_rank is not None else 'NO'}\n"
                )
                report.write(
                    f"Expected page rank: {expected_rank or 'Not found'}\n"
                )
                report.write(
                    f"Valid source label present: "
                    f"{'YES' if has_citation else 'NO'}\n\n"
                )

                report.write("Retrieved passages:\n")

                for rank, chunk in enumerate(results, start=1):
                    report.write(
                        f"\n[S{rank}] {chunk['source_file']} "
                        f"| PDF page: {chunk['pdf_page']} "
                        f"| {chunk['chunk_id']} "
                        f"| Similarity: {chunk['score']:.4f}\n"
                    )
                    report.write(chunk["text"] + "\n")

                report.write(
                    "\nManual answer assessment: [Fully / Partly / Incorrect]\n"
                    "Manual citation assessment: [Supported / Unsupported / Missing]\n"
                    "Review notes:\n"
                )

            except Exception as error:
                # Record this failure and continue with the next question.
                failed += 1
                report.write(
                    f"ERROR: {type(error).__name__}: {error}\n"
                )
                print(f"  Failed: {error}", flush=True)

            elapsed = perf_counter() - started
            report.write(f"\nTime taken: {elapsed:.2f} seconds\n\n")
            report.flush()

        report.write("=" * 70 + "\n")
        report.write("SUMMARY\n")
        report.write(f"Completed: {completed}\n")
        report.write(f"Failed: {failed}\n")

        if completed:
            report.write(
                f"Expected-page matches: {page_hits}/{completed} "
                f"({page_hits / completed:.1%})\n"
            )
            report.write(
                f"Answers with valid source labels: "
                f"{citations_found}/{completed}\n"
            )

        report.write(
            "\nA matching page may not contain the required passage. "
            "A source label does not prove the cited claim is supported. "
            "Review the answers and retrieved text manually.\n"
        )

    print(f"\nCompleted: {completed}; failed: {failed}")
    print(f"Report saved to: {output_path}")


if __name__ == "__main__":
    main()