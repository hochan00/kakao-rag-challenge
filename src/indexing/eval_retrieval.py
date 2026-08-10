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


def dense_ranker(
    snapshot: dict,
    model_name: str,
    revision: str,
    query_prefix: str = "",
    passage_prefix: str = "",
):
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, revision=revision)
    texts = [passage_prefix + u["embed_text"] for u in snapshot["units"]]
    tokenized = model.tokenizer(texts, padding=False, truncation=False)["input_ids"]
    truncated_units = [
        (i, len(ids), snapshot["units"][i]["parent"])
        for i, ids in enumerate(tokenized)
        if len(ids) > model.max_seq_length
    ]
    parents = np.array([u["parent"] for u in snapshot["units"]])
    n = len(snapshot["articles"])
    started = time.perf_counter()
    P = np.asarray(
        model.encode(texts, normalize_embeddings=True, show_progress_bar=False),
        dtype=np.float32,
    )
    encode_seconds = time.perf_counter() - started
    if not np.isfinite(P).all():
        raise ValueError(f"{model_name}: non-finite embedding")
    if not np.allclose(np.linalg.norm(P, axis=1), 1.0, atol=1e-4):
        raise ValueError(f"{model_name}: embeddings are not L2-normalized")

    def rank(question: str) -> np.ndarray:
        q = model.encode([query_prefix + question], normalize_embeddings=True)
        return article_ranking((P @ q.T).ravel(), parents, n)

    rank.truncated_units = truncated_units
    rank.model_revision = revision
    rank.model_name = model_name
    rank.query_prefix = query_prefix
    rank.passage_prefix = passage_prefix
    rank.encode_seconds = encode_seconds
    rank.embedding_bytes = P.nbytes
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
    from huggingface_hub import HfApi

    snapshot = json.loads(Path("artifacts/terms_snapshot.json").read_text(encoding="utf-8"))
    questions = json.loads(
        Path("data/gold_questions_public10.json").read_text(encoding="utf-8")
    )["questions"]
    configs = {
        "e5-base": ("intfloat/multilingual-e5-base", "query: ", "passage: "),
        "bge-m3": ("BAAI/bge-m3", "", ""),
    }
    rankers = {"sparse": sparse_ranker(snapshot)}
    for label, (model_name, query_prefix, passage_prefix) in configs.items():
        revision = HfApi().model_info(model_name).sha
        rankers[label] = dense_ranker(
            snapshot, model_name, revision,
            query_prefix=query_prefix, passage_prefix=passage_prefix,
        )
        print(
            f"{label}: revision={revision}, "
            f"truncated_units={rankers[label].truncated_units}"
        )
    results = {}
    for name, fn in rankers.items():
        results[name] = evaluate(fn, snapshot, questions)
        print(f"{name:8s}", {k: round(v, 3) for k, v in results[name].items()})

    eligible = [label for label in configs if not rankers[label].truncated_units]
    if not eligible:
        raise RuntimeError("무손실 게이트를 통과한 dense 후보가 없음")
    selected = max(
        eligible,
        key=lambda label: (
            results[label]["all_gold_recall@4"],
            results[label]["all_gold_recall@3"],
            results[label]["MRR"],
            -rankers[label].embedding_bytes,
            -rankers[label].encode_seconds,
        ),
    )
    fn = rankers[selected]
    candidates = {
        label: {
            "model_name": rankers[label].model_name,
            "model_revision": rankers[label].model_revision,
            "query_prefix": rankers[label].query_prefix,
            "passage_prefix": rankers[label].passage_prefix,
            "truncated_units": rankers[label].truncated_units,
            "encode_seconds": rankers[label].encode_seconds,
            "embedding_bytes": rankers[label].embedding_bytes,
            "metrics": results[label],
        }
        for label in configs
    }
    selection = {
        "label": selected,
        "model_name": fn.model_name,
        "model_revision": fn.model_revision,
        "query_prefix": fn.query_prefix,
        "passage_prefix": fn.passage_prefix,
        "truncated_units": fn.truncated_units,
        "encode_seconds": fn.encode_seconds,
        "embedding_bytes": fn.embedding_bytes,
        "metrics": results[selected],
        "candidates": candidates,
    }
    Path("artifacts/selected_model.json").write_text(
        json.dumps(selection, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("selected:", json.dumps(selection, ensure_ascii=False))


if __name__ == "__main__":
    main()
