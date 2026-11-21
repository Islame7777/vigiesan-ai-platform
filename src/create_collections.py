from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from config import QDRANT_URL, TEXT_COLLECTION, IMAGE_COLLECTION

client = QdrantClient(url=QDRANT_URL, timeout=60)

COLLECTIONS = {TEXT_COLLECTION: 1024, IMAGE_COLLECTION: 512}

for name, dim in COLLECTIONS.items():
    if client.collection_exists(name):
        client.delete_collection(name)
    client.create_collection(
        collection_name=name,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )
    print(f"OK : {name} (dim={dim})")