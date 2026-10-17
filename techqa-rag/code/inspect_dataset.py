import json
from pathlib import Path

dataset_path = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "original"
    / "train.json"
)

with dataset_path.open(encoding="utf-8") as file:
    records = json.load(file)

print("Total records:", len(records))

for record in records:
    if not record["is_impossible"] and record["contexts"]:
        print("\nQuestion ID:", record["id"])
        print("\nQuestion:", record["question"])
        print("\nReference answer:", record["answer"])

        for document in record["contexts"]:
            print("\nSource filename:", document["filename"])
            print("Document word count:", len(document["text"].split()))
            print("\nDocument preview:")
            print(document["text"][:1000])

        break