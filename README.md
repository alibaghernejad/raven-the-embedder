# Raven – The Embedder

**Raven-The-Embedder** is a core component of the broader **Retraven** project, though it can also be used as a standalone module, serving as a launchpad, playground, or base for other embedding-driven applications.

This module leverages **Python** along with **Qdrant** as the vector database, integrating powerful text and image embedding models. A complete **data ingestion pipeline** is included, with built-in support for **MinIO** (an S3-compatible object storage system) and **RabbitMQ** for asynchronous task handling.


## Models
Leveraged the **FastEmbed** library to generate **text and image** embeddings, providing a streamlined API and seamless **Qdrant integration** for vector-based retrieval tasks.
The library pre-cached embedding and indexing models used by the system for efficient 
and fast retrieval. These models are compatible with FastEmbed and are utilized for both text and image embedding tasks.
Here are the main models used n this project:
```txt
models
└── fastembed_cache
    ├── models--colbert-ir--colbertv2.0
    ├── models--qdrant--all-MiniLM-L6-v2-onnx
    ├── models--Qdrant--bm25
    └── models--Qdrant--clip-ViT-B-32-vision
```

