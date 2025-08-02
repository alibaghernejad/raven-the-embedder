from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from qdrant_client import QdrantClient
from datasets import Dataset

# from Settings import Settings
from Settings import Settings, load_embedding_models
import dataretrieve
import datasetloader
import qdrant

from pydantic_settings import BaseSettings

router = APIRouter()
settings = Settings()

class EmbedRequest(BaseModel):
    text: str
@router.get("/embed")
def embed_text(req: EmbedRequest):
    """
    Embed text using multiple models.
    TODO: Make different models optional and selective
    """
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty or whitespace.")

    dense_model, bm25_model, late_interaction_model, image_model = load_embedding_models()
    dense_vector = next(dense_model.query_embed(req.text))
    bm25_vector = next(bm25_model.query_embed(req.text))
    sparse_vector = {
    "indices": bm25_vector.indices.tolist(),
    "values": bm25_vector.values.tolist()
    }
    late_interaction_vector = next(late_interaction_model.query_embed(req.text))
    return {"dense_embedding_vector": dense_vector.tolist(), "late_interaction_embedding_vector": late_interaction_vector.tolist(), "bm25_embedding_vector": sparse_vector}


class CollectionFromDatasetRequest(BaseModel):
    collection_name: str
@router.post("/datasets/load")
def collection_from_dataset(req: CollectionFromDatasetRequest):
    """    
    TODO: Make different models optional and selective
    """
    if not req.collection_name.strip():
        raise HTTPException(status_code=400, detail="Collection name cannot be empty or whitespace.")
   
    # fetch embedding models
    dense_model, bm25_model, late_interaction_model, image_model = load_embedding_models()

    # Load the required dataset from the local directory
    dataset_list = datasetloader.load_dataset_from_dir(settings.local_dataset_path)[0:10]
    print(f"Number of items in dataset: {len(dataset_list)}")

    # Convert the local dataset to hugging face-compatible dataset
    dataset = Dataset.from_list(dataset_list)

    # Extract Sample Embeddings
    sparse_embeddings = list(bm25_model.passage_embed(dataset["text"][0:1]))
    dense_embeddings = list(dense_model.passage_embed(dataset["text"][0:1]))
    late_interaction_embeddings = list(late_interaction_model.passage_embed(dataset["text"][0:1]))

    # Create client
    client = QdrantClient(settings.qdrant_url, timeout=600, api_key=settings.qdrant_api_key)

    # Create Collection If required
    qdrant.create_collection(client, req.collection_name)

    # Upload data points
    qdrant.batch_upload_data_points(client , req.collection_name, dataset, dense_model, late_interaction_model, bm25_model)
    return {"Result": "Data points uploaded successfully."}

@router.get("/datapoints/retrievers/rerank")
def retrieve_data(query: str):
    """
    Retrieve data points with query using multiple models.
    """
    if not query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty or whitespace.")

    dense_model, bm25_model, late_interaction_model, image_model = load_embedding_models()
    client = QdrantClient(settings.qdrant_url, api_key=settings.qdrant_api_key, timeout=30)
    response = dataretrieve.retrieve_data_rerank(query, client, settings.qdrant_collection_name, dense_model, late_interaction_model, bm25_model)
    print(type(response))
    print(type(response.points))
    return response


@router.get("/datapoints/retrievers/hybrid")
def retrieve_data(query: str):
    """
    Retrieve data points with query using multiple models.
    """
    if not query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty or whitespace.")

    dense_model, bm25_model, late_interaction_model, image_model = load_embedding_models()
    client = QdrantClient(settings.qdrant_url, api_key=settings.qdrant_api_key, timeout=30)

    response = dataretrieve.retrieve_data_hybrid(query, client, settings.qdrant_collection_name, dense_model, late_interaction_model, bm25_model)
    print(type(response))
    print(type(response.points))
    return response


@router.get("/datapoints/retrievers/hybridwithrelation")
def retrieve_data_with_relation(query: str):
    """
    Retrieve data points with query using multiple models.
    """
    if not query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty or whitespace.")

    dense_model, bm25_model, late_interaction_model, image_model = load_embedding_models()
    client = QdrantClient(settings.qdrant_url, api_key=settings.qdrant_api_key)
    response = dataretrieve.retrieve_relations(query, client, settings.qdrant_collection_name_relateds, dense_model)
    print(type(response))
    print(type(response.points))
    return response
