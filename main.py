from collections import namedtuple
import os
from unittest import loader
from fastembed import SparseTextEmbedding, TextEmbedding, LateInteractionTextEmbedding
from datasets import load_dataset
from datasets import Dataset
from qdrant_client import QdrantClient
import uvicorn
from datatypes import DatasetEmbedding
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from api import  router
import subprocess

app = FastAPI()
app.include_router(router)

if __name__ == "__main__":
    # Start the pika-based consumer as a background process 
    # (Not recommended for production)
    import subprocess
    consumer_proc = subprocess.Popen([
        "poetry", "run", "python", "pika_worker.py"
    ])
    try:
        uvicorn.run(app, host="0.0.0.0", port=8000)
    finally:
        consumer_proc.terminate()
        consumer_proc.wait()
    