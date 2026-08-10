import json
from pathlib import Path

from indexing.chunk import build_units
from indexing.parse import DOCS, parse_document
from indexing.validate import validate


def load_gold_pairs(path: Path) -> set[tuple[str, int]]:
    g = json.loads(path.read_text(encoding="utf-8"))
    questions = g.get("questions", [])
    expected_ids = [f"P{i:02d}" for i in range(1, 11)]
    pairs = [
        (x["doc"], x["article"])
        for q in questions
        for x in q.get("gold_articles", [])
    ]
    if (
        [q.get("id") for q in questions] != expected_ids
        or len(pairs) != 12
        or len(set(pairs)) != 12
        or any(doc not in DOCS or not isinstance(article, int) for doc, article in pairs)
    ):
        raise ValueError("gold 계약 불일치: P01~P10, 고유 (문서, 조) 12쌍 필요")
    return set(pairs)


def build_snapshot(data_dir: Path, out_path: Path) -> dict:
    docs = [parse_document(name, data_dir) for name in DOCS]
    articles, units = build_units(docs)
    errors = validate(
        docs, articles, units,
        load_gold_pairs(data_dir / "gold_questions_public10.json"),
    )
    if errors:
        raise SystemExit("검증 게이트 실패:\n" + "\n".join(f"  - {e}" for e in errors))
    snapshot = {
        "documents": [
            {"name": d.name, "effective_date": d.effective_date, "sha256": d.sha256}
            for d in docs
        ],
        "articles": [
            {
                "doc": a.doc, "article": a.n, "title": a.title, "chapter": a.chapter,
                "text": a.text, "citation": f"{a.doc} 제{a.n}조({a.title})",
            }
            for a in articles
        ],
        "units": [
            {"parent": u.parent, "order": u.order, "text": u.text, "embed_text": u.embed_text}
            for u in units
        ],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(snapshot, ensure_ascii=False), encoding="utf-8")
    print(f"조 {len(articles)} / 항 {len(units)} → {out_path}")
    return snapshot


if __name__ == "__main__":
    build_snapshot(Path("data"), Path("artifacts/terms_snapshot.json"))
