import os
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

DATA_DIR = PROJECT_ROOT / "data"
DOCUMENTS_DIR = DATA_DIR / "documents"
IMAGES_DIR = DATA_DIR / "images"

PDF_PATH = DOCUMENTS_DIR / "Manuel_Maintenance_SAN_Projet_Integrateur.pdf"
DOC_ID = "SAN-MAN-001"
REVISION = "1.1"
ACCESS_GROUPS = ["technicien", "responsable"]

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
TEXT_COLLECTION = "maintidoc_text"      # vecteurs BGE-M3, 1024 dim
IMAGE_COLLECTION = "maintidoc_images"   # vecteurs CLIP, 512 dim

EMBEDDING_MODEL = "BAAI/bge-m3"
CLIP_MODEL = "sentence-transformers/clip-ViT-B-32"
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
OLLAMA_URL = "http://localhost:11434"
LLM_MODEL = "qwen2.5:3b"