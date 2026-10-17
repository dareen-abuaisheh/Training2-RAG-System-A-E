from sentence_transformers import SentenceTransformer

# Use a multilingual model for Arabic documents and questions.
model_name = (
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

# The first run downloads the model. Later runs use the cached copy.
model = SentenceTransformer(model_name, device="cpu")

# Confirm that the model can generate an embedding for Arabic text.
sample_text = "الذكاء الاصطناعي يساعد أجهزة الكمبيوتر على تنفيذ مهام متنوعة."
embedding = model.encode(sample_text, normalize_embeddings=True)

print(f"Model: {model_name}")
print(f"Maximum input length: {model.max_seq_length} tokens")
print(f"Embedding dimensions: {embedding.shape[0]}")