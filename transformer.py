# Load model directly
from transformers import AutoModel
model = AutoModel.from_pretrained("Qdrant/bm25", torch_dtype="auto"),