import json
from pathlib import Path

import numpy as np
import pytest

from indexing.make_baseline import build_baseline, score, validate_contract
from indexing.retrieve import HybridRetriever

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"
ARTIFACTS = ROOT / "artifacts"

pytestmark = pytest.mark.skipif(
    not (ARTIFACTS / "dense_embeddings.npy").exists(),
    reason="artifacts 미생성 — uv run python -m indexing.build && ... embed 먼저 실행",
)


@pytest.fixture(scope="module")
def fixtures():
    snapshot = json.loads((ARTIFACTS / "terms_snapshot.json").read_text(encoding="utf-8"))
    questions = json.loads(
        (DATA / "gold_questions_public10.json").read_text(encoding="utf-8")
    )["questions"]
    dense = np.load(ARTIFACTS / "dense_embeddings.npy")
    queries = np.load(ARTIFACTS / "public10_query_embeddings.npy")
    return snapshot, questions, HybridRetriever(snapshot, dense), queries


def test_retrieve_returns_between_one_and_four(fixtures):
    snapshot, questions, retriever, queries = fixtures
    for i, q in enumerate(questions):
        got = retriever.retrieve(q["question"], queries[i], 4)
        assert 1 <= len(got) <= 4
        assert len({(a["doc"], a["article"]) for a in got}) == len(got)  # 중복 없음


def test_rank_covers_every_article_exactly_once(fixtures):
    snapshot, questions, retriever, queries = fixtures
    ranked = retriever.rank(questions[0]["question"], queries[0])
    assert sorted(ranked.tolist()) == list(range(len(snapshot["articles"])))


def test_sparse_only_path_works_without_dense(fixtures):
    snapshot, questions, _, _ = fixtures
    sparse_only = HybridRetriever(snapshot, dense_matrix=None)
    got = sparse_only.retrieve(questions[0]["question"], None, 4)
    assert len(got) == 4


def test_baseline_meets_runner_contract(fixtures):
    snapshot, questions, retriever, queries = fixtures
    baseline = build_baseline(snapshot, questions, retriever, queries)
    assert validate_contract(baseline, questions) == []


def test_baseline_recovers_all_public_gold(fixtures):
    snapshot, questions, retriever, queries = fixtures
    baseline = build_baseline(snapshot, questions, retriever, queries)
    metrics = score(baseline, questions)
    # 개발 시점 실측. 회귀하면 검색 변경이 공개셋을 깨뜨린 것이다.
    assert metrics["hit@1"] == 1.0
    assert metrics["all_gold_recall@4"] == 1.0
