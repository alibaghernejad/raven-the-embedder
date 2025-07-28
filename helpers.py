import uuid
from minio import Minio

from models import EmbeddingTask

def json_to_text(json_data):
    """
    Convert JSON data to plain text.
    """
    # Use Performant way to generate texts
    text = "\n".join(item.get("text", "") for item in json_data)
    return text

def text_to_json(text):
    """
    Convert plain text to JSON format.
    """
    json_data = []
    for line in text.strip().split("\n"):
        json_data.append({"text": line})
    return json_data


# Flatten all string values in the JSON
def extract_texts(obj):
    if isinstance(obj, dict):
        return " ".join([extract_texts(v) for v in obj.values()])
    elif isinstance(obj, list):
        return " ".join([extract_texts(i) for i in obj])
    elif isinstance(obj, str):
        return obj
    else:
        return ""

def get_minio_client(minio_params):
    """
    Create a MinIO client using the provided parameters.
    """
    return Minio(
        minio_params.get('Endpoint').replace('http://', '').replace('https://', ''),
        access_key=minio_params.get('AccessKey'),
        secret_key=minio_params.get('SecretKey'),
        secure=minio_params.get('Secure', 'False').lower() == 'true'
    )

def get_minio_url(minio_params, bucket, object_key):
    """
    Generate a MinIO URL for the specified bucket and object key.
    """
    return f"http://{minio_params.get('Endpoint')}/{bucket}/{object_key}"


def get_task_from_row(row):
    """
    Convert a database row to an EmbeddingTask object.
    """
    task = EmbeddingTask()
    (
        task.id,
        task.storage_provider,
        task.storage_bucket_name,
        task.storage_object_key,
        task.vector_db_provider,
        task.embeddings_provider,
        task.correlation_key,
        task.chunk_size,
        task.chunk_overlap,
        task.chunk_strategy,
        task.data_bag,
        task.status,
        task.created_at,
        task.updated_at,
        task.document_id
    ) = row
    return task

def get_new_id(document_id, chunk_id):
    """
    Generate a new unique ID for a document chunk.
    """
    if document_id:
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{document_id}-{chunk_id}"))
    return str(uuid.uuid4())