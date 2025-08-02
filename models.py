import datetime

class EmbeddingTask():
    id : int
    storage_provider : str
    storage_bucket_name : str
    storage_object_key : str
    vector_db_provider : str
    embeddings_provider : str
    correlation_key : str
    chunk_size : int
    chunk_overlap : int
    chunk_strategy : str
    data_bag : dict
    status : str
    created_at : datetime
    updated_at : datetime
    document_id : str
