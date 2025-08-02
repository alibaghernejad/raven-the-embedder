from datetime import datetime
from celery import uuid
from qdrant_client import QdrantClient, models
import tqdm

import helpers
from models import EmbeddingTask

def create_collection ( client, collection_name ) :
    """
    Create a collection in Qdrant
    """
    # TODO: Read from sample and actual model
    dense_embed_dimentions = 384
    image_clip_dimentions = 512
    late_interaction_embed_dimentions = 128
    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name,
            vectors_config={
            "text-all-MiniLM-L6-v2": models.VectorParams(
                    size= dense_embed_dimentions,
                distance=models.Distance.COSINE,
            ),
            "text-colbertv2.0": models.VectorParams(
                size=late_interaction_embed_dimentions,
                distance=models.Distance.COSINE,
                multivector_config=models.MultiVectorConfig(
                    comparator=models.MultiVectorComparator.MAX_SIM,
                ),
                
            ),
            "transcript-all-MiniLM-L6-v2": models.VectorParams(
                size=dense_embed_dimentions,
                distance=models.Distance.COSINE,
            ),

            "image-clip-viT-b-32-vision": models.VectorParams(
                size=image_clip_dimentions,
                distance=models.Distance.COSINE,
            ),

        },
        sparse_vectors_config={
            "text-bm25": models.SparseVectorParams(
                modifier=models.Modifier.IDF,
            )
        }
    )


def batch_upload_data_points(client, collection_name, dataset, dense_embed, late_itr_embed, sparse_embed, batch_size=4):
    """
    Upload a batch of documents to Qdrant
    """
    for batch in tqdm.tqdm(dataset.iter(batch_size=batch_size), 
                        total=len(dataset) // batch_size):
        dense_embeddings = list(dense_embed.passage_embed(batch["text"]))
        bm25_embeddings = list(sparse_embed.passage_embed(batch["text"]))
        late_interaction_embeddings = list(late_itr_embed.passage_embed(batch["text"]))
        
        client.upload_points(
            collection_name,
            points=[
                models.PointStruct(
                    id=int(batch["_id"][i]),
                    vector={
                        "text-all-MiniLM-L6-v2": dense_embeddings[i].tolist(),
                        "text-bm25": bm25_embeddings[i].as_object(),
                        "text-colbertv2.0": late_interaction_embeddings[i].tolist(),
                    },
                    payload={
                        "_id": batch["_id"][i],
                        "title": batch["title"][i],
                        "text": batch["text"][i],
                        "text_en": batch["text_en"][i],
                        "source": batch["_source"][i],
                    }
                )
                for i, _ in enumerate(batch["_id"])
            ],
            batch_size=batch_size,  
        )


def get_data_points(embedding_task:EmbeddingTask, docs, dense_model, late_itr_embed, sparse_embed):
    # In case of large documents with multiple chunks, we need to control payload repetition if matters
    points=[
        models.PointStruct(
            id= helpers.get_new_id(embedding_task.document_id, i),
            vector={
                "text-all-MiniLM-L6-v2": next(dense_model.passage_embed(doc.page_content)),
                "text-bm25": next(sparse_embed.passage_embed(doc.page_content)).as_object(),
                "text-colbertv2.0": next(late_itr_embed.passage_embed(doc.page_content)),
            },
            payload={
                "document_id": embedding_task.document_id if embedding_task.document_id else None,
                "embedding_task_id": embedding_task.id,
                "correlation_id": embedding_task.correlation_key if embedding_task.correlation_key else None,
                "version": 1.0,
                "chunks_count": len(docs),
                "chunk_id": i,
                "updated_at": datetime.now(),
                "text": (doc.page_content if doc and doc.page_content else None),
                "source": getattr(doc, "payload", None)
            }
        )
        for i, doc in enumerate(docs)
    ]
    return points

def get_update_vector_points(embedding_task:EmbeddingTask, is_image:bool, docs, dense_model, image_model):
    points=[
        models.PointStruct(
            id= helpers.get_new_id(embedding_task.correlation_key, i),
            vector={ "image-clip-viT-b-32-vision": next(image_model.passage_embed(doc.page_content)) }
                if is_image else {
                    "transcript-all-MiniLM-L6-v2": next(dense_model.passage_embed(doc.page_content))
                },
            payload={"image_caption": doc.page_content if doc and doc.page_content else None}
                if is_image else {
                    "transcript_text": doc.page_content if doc and doc.page_content else None
                }
        )
        for i, doc in enumerate(docs)
    ]
    return points

def update_vector(client, collection_name,embedding_task:EmbeddingTask, is_image:bool, docs, dense_model, image_model):
    full_doc = " ".join(doc.page_content for doc in docs if doc and doc.page_content)
    client.update_vectors(
        collection_name= collection_name,
        points=[
            models.PointVectors(
                id=helpers.get_new_id(embedding_task.correlation_key, 0),
                vector={ "image-clip-viT-b-32-vision": next(image_model.embed(docs[0].page_content)) }
                    if is_image else {
                        "transcript-all-MiniLM-L6-v2": next(dense_model.passage_embed(full_doc))
                    }
            )
        ],
    )

def update_payload(client, collection_name,embedding_task:EmbeddingTask, is_image:bool, docs, dense_model, image_model):
    payload = {}
    if is_image:
        image_caption = docs[0].payload.get("caption") if docs[0] else None
        payload ={"image_caption":  image_caption if image_caption else None}
    else:
        full_doc = " ".join(doc.page_content for doc in docs if doc and doc.page_content)
        payload = {"transcript_text":full_doc if full_doc else None}
    # Add any additional metadata to the payload if needed
    payload["updated_at"] = datetime.now()
    
    client.set_payload(
        collection_name= collection_name,
        payload=payload,    
        points=models.Filter(
            must=[
                    models.HasIdCondition(has_id=[helpers.get_new_id(embedding_task.correlation_key, 0)])
                ],
        ),                    
    )
