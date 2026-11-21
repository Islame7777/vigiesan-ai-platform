from rapidocr import RapidOCR

_engine = None


def ocr_text(path, min_conf=0.5):
    """Lit une image. Renvoie (texte, confiance moyenne)."""
    global _engine
    if _engine is None:
        _engine = RapidOCR()
    out = _engine(str(path))
    txts = list(out.txts) if out.txts else []
    scores = [float(s) for s in out.scores] if out.scores else []
    pairs = [(t, s) for t, s in zip(txts, scores) if s >= min_conf]
    if not pairs:
        return "", 0.0
    mean = sum(s for _, s in pairs) / len(pairs)
    return " | ".join(t for t, _ in pairs), mean