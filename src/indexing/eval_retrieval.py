import json
import time
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def article_ranking(unit_scores: np.ndarray, parents: np.ndarray, n_articles: int) -> np.ndarray:
    """항 점수를 부모 조로 max 집계해 조 순위(내림차순 인덱스)를 반환."""
    scores = np.full(n_articles, -np.inf)
    np.maximum.at(scores, parents, unit_scores)
    return np.argsort(-scores)


def sparse_ranker(snapshot: dict):
    texts = [u["embed_text"] for u in snapshot["units"]]
    parents = np.array([u["parent"] for u in snapshot["units"]])
    n = len(snapshot["articles"])
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True)
    X = vec.fit_transform(texts)

    def rank(question: str) -> np.ndarray:
        q = vec.transform([question])
        return article_ranking((X @ q.T).toarray().ravel(), parents, n)

    return rank


def evaluate(rank_fn, snapshot: dict, questions: list[dict], ks=(1, 3, 4, 5)) -> dict:
    arts = snapshot["articles"]
    hits = {k: 0 for k in ks}
    recalls = {k: [] for k in ks}
    rr = []
    for q in questions:
        gold = {(g["doc"], g["article"]) for g in q["gold_articles"]}
        ranked = [(arts[i]["doc"], arts[i]["article"]) for i in rank_fn(q["question"])]
        first = next((r + 1 for r, p in enumerate(ranked) if p in gold), None)
        rr.append(1 / first if first else 0.0)
        for k in ks:
            hits[k] += any(p in gold for p in ranked[:k])
            recalls[k].append(len(gold & set(ranked[:k])) / len(gold))
    n = len(questions)
    return {
        **{f"hit@{k}": hits[k] / n for k in (1, 3, 5)},
        **{f"all_gold_recall@{k}": sum(recalls[k]) / n for k in ks},
        "MRR": sum(rr) / n,
    }


def main():
    snapshot = json.loads(Path("artifacts/terms_snapshot.json").read_text(encoding="utf-8"))
    questions = json.loads(
        Path("data/gold_questions_public10.json").read_text(encoding="utf-8")
    )["questions"]
    result = evaluate(sparse_ranker(snapshot), snapshot, questions)
    print("sparse(char 2-5gram):", {k: round(v, 3) for k, v in result.items()})


if __name__ == "__main__":
    main()
