from config import IMAGES_DIR

try:
    from rapidocr import RapidOCR          # paquet récent
    NEW = True
except ImportError:
    from rapidocr_onnxruntime import RapidOCR   # ancien paquet
    NEW = False

engine = RapidOCR()


def ocr_image(path):
    if NEW:
        out = engine(str(path))
        txts = list(out.txts) if out.txts else []
        scores = [float(s) for s in out.scores] if out.scores else []
    else:
        res, _ = engine(str(path))
        res = res or []
        txts = [r[1] for r in res]
        scores = [float(r[2]) for r in res]
    return txts, scores


for p in sorted(IMAGES_DIR.glob("FIG-*.png")):
    txts, scores = ocr_image(p)
    moy = sum(scores) / len(scores) if scores else 0
    print(f"\n=== {p.name} : {len(txts)} blocs, confiance moyenne {moy:.2f}")
    print("   ", " | ".join(txts))