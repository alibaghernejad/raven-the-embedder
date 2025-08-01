
import psycopg2
from minio import Minio
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from Settings import Settings
from qdrant_client import QdrantClient
from fastembed import ImageEmbedding
from fastembed import TextEmbedding, SparseTextEmbedding, LateInteractionTextEmbedding
import datetime
from qdrant_client import models
import datasetloader
import helpers
import qdrant
import os
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from Settings import load_embedding_models

# Load BLIP model and processor once
blip_processor = None
blip_model = None
def get_blip_model():
    global blip_processor, blip_model
    if blip_processor is None or blip_model is None:
        blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return blip_processor, blip_model

# DB Connection Helper
def get_pg_conn():
    settings = Settings()
    # Parse connection string for psycopg2
    # Example: Host=localhost;Port=5432;Username=postgres;Password=postgres;Database=retraven
    conn_params = dict(
        item.split('=') for item in settings.postgres_conn_str.split(';') if item
    )
    return psycopg2.connect(
        host=conn_params.get('Host'),
        port=conn_params.get('Port'),
        dbname=conn_params.get('Database'),
        user=conn_params.get('Username'),
        password=conn_params.get('Password')
    )

# Main Pipeline 
def process_embedding_task(task_id: int):
    settings = Settings()
    conn = get_pg_conn()
    cur = conn.cursor()
    try:
        # Fetch embedding task
        cur.execute("SELECT * FROM embedding_tasks WHERE id = %s", (task_id,))
        row = cur.fetchone()
        if not row:
            print(f"Task {task_id} not found.")
            return

        # Convert row to EmbeddingTask static type
        embedding_task = helpers.get_task_from_row(row)

        # Download file from Minio
        # Parse minio_conn_str: Endpoint=...;AccessKey=...;SecretKey=...;Secure=...
        minio_params = dict(
            item.split('=') for item in settings.minio_conn_str.split(';') if item
        )
        minio_client = helpers.get_minio_client(minio_params)
        bucket = embedding_task.storage_bucket_name
        object_key = embedding_task.storage_object_key
        local_path = f"/tmp/{object_key.split('/')[-1]}"
        minio_client.fget_object(bucket, object_key, local_path)
        
        # get minio url 
        object_storage_url = helpers.get_minio_url(minio_params, bucket, object_key)
        print(f"Minio URL: {object_storage_url}")

        # Extract text or image features with LangChain or custom logic       
        ext = os.path.splitext(local_path)[1].lower()
        docs = []
        is_image = False
        if ext in [".txt", ".pdf", ".docx", ".odt"]:
            loader = UnstructuredFileLoader(local_path)
            docs = loader.load()
        elif ext == ".json":
            # Load and flatten JSON, then treat as a text doc for chunking
            file_entry = datasetloader.load_json_file(local_path)
            formatted_entry = datasetloader.format_json_item(file_entry)
            payload = formatted_entry["_source"] if formatted_entry else None
            # Add metadata attribute for compatibility with LangChain splitters
            docs = [type("Doc", (), {"page_content": formatted_entry['text'], "metadata": {}, "payload":payload})()]
        elif ext in [".png", ".jpg", ".jpeg"]:
            is_image = True
            # Use CLIP model to embed image
            docs = [type("Doc", (), {"page_content": local_path, "payload": {}})()]
        else:
            raise ValueError(f"Unsupported file extension: {ext}")
        # Chunk text if not image
        if not is_image:
            chunk_size = embedding_task.chunk_size or 1000
            chunk_overlap = embedding_task.chunk_overlap or 100
            splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            texts = splitter.split_documents(docs)
        else:
            texts = docs

        client = QdrantClient(settings.qdrant_url, api_key=settings.qdrant_api_key)
        # fetch embedding models
        dense_model, sparse_model, late_interaction_model, image_model = load_embedding_models()
        # Create Collection If required
        qdrant.create_collection(client, settings.qdrant_collection_name)

        # Check if the document is a parent or a relation with taskcorrelation_id.
        is_data_point_upsert_request = embedding_task.correlation_key is None
        if is_data_point_upsert_request:
            # Delete previous document 
            # Consider that the previous document might contains multiple chunks, so
            # Doing Upsert is not enough and cause the old docs to remains 
            if embedding_task.document_id:
                client.delete(
                    collection_name=settings.qdrant_collection_name,
                    points_selector=models.FilterSelector(
                        filter=models.Filter(
                            must=[
                                models.FieldCondition(key="document_id", match=models.MatchValue(value=embedding_task.document_id))
                            ]
                        )
                    )
                )
            # upsert new docs
            points = qdrant.get_data_points(embedding_task, docs, dense_model, late_interaction_model, sparse_model)
            client.upsert(collection_name=settings.qdrant_collection_name, points=points)

        # vector update image
        elif is_image:
            doc = docs[0]
            # Image embedding using CLIP and BLIP captioning
            image_model = ImageEmbedding(model_name="Qdrant/clip-ViT-B-32-vision")
            image_vec = list(image_model.embed(doc.page_content))
            # Generate image caption using BLIP
            # this should be separated as a different function due to slow nature of image captioning
            # This is not suitable for a production environment
            blip_processor, blip_model = get_blip_model()
            try:
                pil_image = Image.open(doc.page_content).convert("RGB")
                inputs = blip_processor(pil_image, return_tensors="pt")
                out = blip_model.generate(**inputs)
                caption = blip_processor.decode(out[0], skip_special_tokens=True)
                doc.payload["caption"] = caption
            except Exception as e:
                caption = None
                print(f"[ work offlne without request to server?] Failed to caption image {doc.page_content}: {e}")
            # uncommnet to use caption vector
            # caption_vector = list(dense_model.passage_embed(caption))

            doc.payload["caption"] = caption
            qdrant.update_vector(client, settings.qdrant_collection_name, embedding_task, is_image, docs, dense_model, image_model)
            qdrant.update_payload(client, settings.qdrant_collection_name, embedding_task, is_image, docs, dense_model, image_model)
        # vector update text
        else:
            qdrant.update_vector(client, settings.qdrant_collection_name, embedding_task, is_image, docs, dense_model, image_model)
            qdrant.update_payload(client, settings.qdrant_collection_name, embedding_task, is_image, docs, dense_model, image_model)

        # Update status
        cur.execute("UPDATE embedding_tasks SET status = %s, updated_at = %s WHERE id = %s", ("generated", datetime.datetime.now(), task_id))
        conn.commit()
    except Exception as e:
        cur.execute("UPDATE embedding_tasks SET status = %s, updated_at = %s WHERE id = %s", ("retrievalfailed", datetime.datetime.now(), task_id))
        conn.commit()
        print(f"Error processing task {task_id}: {e}")
    finally:
        cur.close()
        conn.close()
