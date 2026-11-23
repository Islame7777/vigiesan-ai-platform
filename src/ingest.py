import io
import re
import uuid
from ocr import ocr_text

import pymupdf
from PIL import Image
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from sentence_transformers import SentenceTransformer

from config import (
    PDF_PATH,
    IMAGES_DIR,
    DOC_ID,
    REVISION,
    ACCESS_GROUPS,
    QDRANT_URL,
    TEXT_COLLECTION,
    IMAGE_COLLECTION,
    EMBEDDING_MODEL,
    CLIP_MODEL,
)

IMAGES_DIR.mkdir(parents=True, exist_ok=True)

FIGS = {
    2: "FIG-01",
    3: "FIG-02",
    4: "FIG-03",
    5: "FIG-04",
    6: "FIG-05",
}

EVAL_TABLES = {(7, 2)}  # tableau des questions (page 7)
EVAL_PAGE = 7  # Jeu de questions : sert à t'évaluer, jamais de preuve

TABLE_CONTEXT = {
    (3, 1): "Termes Fibre Channel",
    (4, 1): "Elements, symptomes et verifications",
    (5, 1): "Codes d'alarmes ALARM-TAB-01",
    (6, 1): "Operations et autorisation dans MaintiDoc",
    (7, 1): "Historique des interventions",
    (7, 3): "Exemple de ticket a preparer",
}


def fix(s: str) -> str:
    """Nettoyage léger : espaces et codes du type AL-01."""
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(
        r"\b(AL|INC)\s*-?\s*(\d{2,3})(?:\s*-(?!\w))?",
        r"\1-\2",
        s,
    )
    return s


def pid(name: str) -> str:
    """Identifiant stable : relancer le script ne crée pas de doublons."""
    return str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"{DOC_ID}/{name}",
        )
    )


def base(page, element_id, kind):
    return {
        "document_id": DOC_ID,
        "revision": REVISION,
        "page": page,
        "element_id": element_id,
        "type": kind,
        "access_groups": ACCESS_GROUPS,
    }


def cell_text(page, bbox):
    """Texte d'une cellule, reconstruit à partir des mots."""
    words = page.get_text("words", clip=pymupdf.Rect(bbox))
    words.sort(key=lambda w: (round(w[1]), w[0]))
    return " ".join(w[4] for w in words)


def table_rows(page, tab):
    """Lignes du tableau via les mots ; sinon méthode classique."""
    try:
        rows = [
            [fix(cell_text(page, c)) if c else "" for c in row.cells]
            for row in tab.rows
        ]

        if rows and any(any(c for c in r) for r in rows):
            return rows

    except Exception as e:
        print(f"  (méthode mots indisponible : {e})")

    return [
        [fix(str(c or "")) for c in r]
        for r in tab.extract()
    ]


# 1) Lire le PDF

doc = pymupdf.open(PDF_PATH)

text_items = []
image_items = []

for i, page in enumerate(doc):
    p = i + 1
    is_eval = p == EVAL_PAGE

    txt = fix(page.get_text("text", sort=True))
    if is_eval:
        m = re.search(r"7\.\s*Jeu de questions", txt)
        if m:
            eval_text = txt[m.start():]
            txt = txt[:m.start()].strip()
            text_items.append(
                (eval_text, {**base(p, f"P{p:02d}-TEXT-EVAL", "eval_question"),
                             "text": eval_text}))
        else:
            print("ATTENTION : separation de la page 7 impossible, page ignoree")
            txt = ""

    if txt:
        kind = "texte"

        text_items.append(
            (
                txt,
                {
                    **base(p, f"P{p:02d}-TEXT", kind),
                    "text": txt,
                },
            )
        )

    try:
        for t, tab in enumerate(page.find_tables().tables, 1):
            rows = table_rows(page, tab)

            if len(rows) < 2:
                continue

            header = rows[0]

            for r, row in enumerate(rows[1:], 1):
                line = " ; ".join(
                    f"{h}: {v}"
                    for h, v in zip(header, row)
                )

                ctx = TABLE_CONTEXT.get((p, t), "")

                if ctx:
                    line = f"{ctx} : {line}"

                kind = "eval_question" if (p, t) in EVAL_TABLES else "tableau_ligne"

                text_items.append(
                    (
                        line,
                        {
                            **base(
                                p,
                                f"P{p:02d}-TAB{t:02d}-L{r:02d}",
                                kind,
                            ),
                            "text": line,
                            "row": row,
                        },
                    )
                )

    except Exception as e:
        print(f"Page {p}: tableaux ignorés ({e})")

    for j, info in enumerate(page.get_images(full=True), 1):
        try:
            raw = doc.extract_image(info[0])

            im = Image.open(
                io.BytesIO(raw["image"])
            ).convert("RGB")

            if im.width < 120 or im.height < 85:
                continue

            fig = FIGS.get(p, f"P{p:02d}-IMG")
            eid = fig if j == 1 else f"{fig}-{j}"

            path = IMAGES_DIR / f"{eid}.png"

            im.save(path)

            image_items.append(
                (
                    im,
                    {
                        **base(p, eid, "image"),
                        "image_path": str(path),
                        "width": im.width,
                        "height": im.height,
                    },
                )
            )

            if p in FIGS and j == 1:
                ocr_str, conf = ocr_text(path)
                if ocr_str:
                    card = (
                        f"{fig} - schema de la page {p} "
                        f"(texte lu par OCR) : {ocr_str}"
                    )
                    text_items.append(
                        (
                            card,
                            {
                                **base(p, f"{fig}-OCR", "schema_ocr"),
                                "text": card,
                                "image_path": str(path),
                                "ocr_confidence": round(conf, 2),
                            },
                        )
                    )

        except Exception as e:
            print(f"Page {p}: image {j} ignorée ({e})")

doc.close()

print(
    f"Trouvé : {len(text_items)} fiches texte, "
    f"{len(image_items)} images"
)

print("\nContrôle : 4 lignes de tableau de la page 5")

shown = 0

for _, pl in text_items:
    if (
        pl["page"] == 5
        and pl["type"] == "tableau_ligne"
        and shown < 4
    ):
        print("  ", pl["text"][:110])
        shown += 1


# 2) Embeddings

print("\nChargement BGE-M3 (1er lancement : ~2 Go à télécharger)...")

bge = SentenceTransformer(
    EMBEDDING_MODEL,
    device="cpu",
)

bge.max_seq_length = 512

text_vecs = bge.encode(
    [t for t, _ in text_items],
    normalize_embeddings=True,
    show_progress_bar=True,
)

print("Chargement CLIP...")

clip = SentenceTransformer(
    CLIP_MODEL,
    device="cpu",
)

img_vecs = clip.encode(
    [im for im, _ in image_items],
    normalize_embeddings=True,
)


# 3) Ranger dans Qdrant

client = QdrantClient(url=QDRANT_URL, timeout=60)

client.upsert(
    collection_name=TEXT_COLLECTION,
    points=[
        PointStruct(
            id=pid(pl["element_id"]),
            vector=v.tolist(),
            payload=pl,
        )
        for v, (_, pl) in zip(text_vecs, text_items)
    ],
)

client.upsert(
    collection_name=IMAGE_COLLECTION,
    points=[
        PointStruct(
            id=pid(pl["element_id"]),
            vector=v.tolist(),
            payload=pl,
        )
        for v, (_, pl) in zip(img_vecs, image_items)
    ],
)

for name in (TEXT_COLLECTION, IMAGE_COLLECTION):
    print(
        name,
        "->",
        client.count(name).count,
        "fiches",
    )