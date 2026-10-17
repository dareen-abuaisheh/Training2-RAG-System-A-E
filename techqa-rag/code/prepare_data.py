import json
from pathlib import Path


# This script is inside techqa-rag/code/.
# Going up two levels gives us the main techqa-rag folder.
project_path = Path(__file__).resolve().parent.parent

dataset_path = project_path / "data" / "original" / "train.json"
documents_path = project_path / "data" / "documents"
evaluation_path = project_path / "evaluation"

# Create the output folders if they do not already exist.
documents_path.mkdir(parents=True, exist_ok=True)
evaluation_path.mkdir(parents=True, exist_ok=True)


# Read the JSON file into a Python list of question records.
# The original dataset file is only read, never modified.
with dataset_path.open(encoding="utf-8") as file:
    records = json.load(file)


# Store documents by filename so we can detect duplicates.
# Example: {"document.txt": "Document contents..."}
selected_documents = {}

# Keep questions and reference answers separate from searchable documents.
test_questions = []

target_document_count = 8
minimum_words = 300
maximum_words = 1500


for record in records:
    # Start with questions the dataset marks as answerable.
    if record["is_impossible"]:
        continue

    # Use examples with exactly one supporting document.
    # This makes checking the expected source easier in our first version.
    contexts = record["contexts"]
    if len(contexts) != 1:
        continue

    document = contexts[0]

    # Keep only the filename, without any directory components.
    filename = Path(document["filename"]).name
    text = document["text"]

    # Several questions may refer to the same file.
    # This script selects one document and one question per unique filename.
    if filename in selected_documents:
        continue

    # Count whitespace-separated words to estimate document length.
    # Choose moderate lengths so we have useful material to chunk later.
    word_count = len(text.split())
    if not minimum_words <= word_count <= maximum_words:
        continue

    # Remove surrounding whitespace and reject missing reference answers.
    reference_answer = record["answer"].strip()
    if not reference_answer or reference_answer == "-":
        continue

    selected_documents[filename] = text

    # expected_source is an evaluation label, not a hint for retrieval.
    # Later, we will check whether retrieval finds this file independently.
    test_questions.append({
        "id": record["id"],
        "question": record["question"],
        "reference_answer": reference_answer,
        "expected_source": filename,
    })

    # Stop once we have selected the required number of unique documents.
    if len(selected_documents) == target_document_count:
        break


# Check the selection before writing any documents.
if len(selected_documents) < target_document_count:
    raise ValueError(
        f"Found only {len(selected_documents)} suitable documents; "
        f"expected {target_document_count}."
    )


# Save each supporting document as a UTF-8 TXT file.
# Running this again overwrites files with the same names.
for filename, text in selected_documents.items():
    output_path = documents_path / filename
    output_path.write_text(text, encoding="utf-8")

    print(f"Saved: {filename} — {len(text.split())} words")


# Save the answer key outside the searchable document folder.
# indent=2 makes the JSON readable.
# ensure_ascii=False preserves non-English characters if present.
questions_path = evaluation_path / "test_questions.json"

questions_path.write_text(
    json.dumps(test_questions, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

print(f"\nDocuments saved: {len(selected_documents)}")
print(f"Test questions saved: {len(test_questions)}")
print(f"Documents folder: {documents_path}")
print(f"Evaluation file: {questions_path}")