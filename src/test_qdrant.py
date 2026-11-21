from qdrant_client import QdrantClient
from src.config import QDRANT_URL


def test_connection():
    print("Connexion à Qdrant...")

    client = QdrantClient(url=QDRANT_URL)

    collections = client.get_collections()

    print("Connexion Qdrant : OK")
    print("\nCollections disponibles :")

    if collections.collections:
        for collection in collections.collections:
            print("-", collection.name)
    else:
        print("Aucune collection pour le moment.")

    client.close()


if __name__ == "__main__":
    test_connection()
