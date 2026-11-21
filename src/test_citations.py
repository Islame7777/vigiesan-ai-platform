from pydantic import ValidationError

from search import search_text, rerank
from citations import Answer, check_citations

q = "Que signifie AL-01 ?"
evidence = rerank(q, search_text(q, 10), 3)
print("Preuves données au modèle :", [e["element_id"] for e in evidence])

CASES = {
    "1. bonne citation": '''{"status":"answered","answer":"AL-01 : chemin FC degrade",
      "citations":[{"document_id":"SAN-MAN-001","element_id":"P05-TAB01-L01","page":5}]}''',
    "2. source inventee": '''{"status":"answered","answer":"x",
      "citations":[{"document_id":"SAN-MAN-001","element_id":"P05-TAB01-L09","page":5}]}''',
    "3. mauvaise page": '''{"status":"answered","answer":"x",
      "citations":[{"document_id":"SAN-MAN-001","element_id":"P05-TAB01-L01","page":4}]}''',
    "4. reponse sans source": '{"status":"answered","answer":"x"}',
    "5. JSON mal forme": '{"status":"peut-etre","answer":"x"}',
}

for name, raw in CASES.items():
    try:
        ans = Answer.model_validate_json(raw)
        _, verdict = check_citations(ans, evidence)
    except ValidationError:
        verdict = "Format invalide (schema Pydantic) : refuse"
    print(f"{name:25} -> {verdict}")