import os

class Settings:
    def __init__(self):
        # ...existing code...
        # Set fastembed cache directory globally
        os.environ["FASTEMBED_CACHE_PATH"] = os.path.join(os.path.dirname(__file__), "models")
import amqp
from fastembed import ImageEmbedding, LateInteractionTextEmbedding, SparseTextEmbedding, TextEmbedding
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "Raven Retriever"
    notify_email: str = "alibaghernezhad@gmail.com"

    qdrant_url: str = "http://localhost:32772"
    qdrant_collection_name: str = "movie_collection"
    qdrant_collection_name_relateds: str = "movie_collection_relations"
    qdrant_api_key: str = "61d67203-edad-40ef-9999-b9adb1bca800"

    rabbitmq_broker_url: str = "amqp://rabbituser:rabbitpassword@localhost:32775"
    embedding_task_queue: str = "EmbeddingTasks"

    minio_conn_str: str = "Endpoint=http://localhost:32770;AccessKey=minioadmin;SecretKey=minioadmin;Secure=False"
    postgres_conn_str: str = "Host=localhost;Port=32774;Username=postgresroot;Password=Ert@123;Database=Retraven"
    local_dataset_path: str = "datasets/gapfilm"


# Dependency resolver for the model
def load_embedding_models():
    # Load once and cache as a global variable
    if not hasattr(load_embedding_models, "dense_embedding_model"):
        load_embedding_models.dense_embedding_model = TextEmbedding("sentence-transformers/all-MiniLM-L6-v2")
    if not hasattr(load_embedding_models, "bm25_embedding_model"):
        load_embedding_models.bm25_embedding_model = SparseTextEmbedding("Qdrant/bm25")
    if not hasattr(load_embedding_models, "late_interaction_embedding_model"):
        load_embedding_models.late_interaction_embedding_model = LateInteractionTextEmbedding("colbert-ir/colbertv2.0")
    if not hasattr(load_embedding_models, "image_embedding_model"):
        load_embedding_models.image_embedding_model = ImageEmbedding("Qdrant/clip-ViT-B-32-vision")
    return (load_embedding_models.dense_embedding_model, load_embedding_models.bm25_embedding_model, load_embedding_models.late_interaction_embedding_model, load_embedding_models.image_embedding_model)

