import json
from pathlib import Path

import numpy as np

from indexing.eval_retrieval import evaluate
from indexing.parse import DOCS

DATA = Path(__file__).parent.parent / "data"


def test_public_gold_contract():
    questions = json.loads(
        (DATA / "gold_questions_public10.json").read_text(encoding="utf-8")
    )["questions"]
    pairs = [
        (g["doc"], g["article"])
        for q in questions
        for g in q["gold_articles"]
    ]
    assert [q["id"] for q in questions] == [f"P{i:02d}" for i in range(1, 11)]
    assert len(pairs) == 12
    assert all(doc in DOCS and isinstance(article, int) for doc, article in pairs)


def test_all_gold_recall_does_not_collapse_multi_gold_to_hit():
    snapshot = {
        "articles": [
            {"doc": "A", "article": 1},
            {"doc": "B", "article": 2},
            {"doc": "C", "article": 3},
        ]
    }
    questions = [{
        "question": "q",
        "gold_articles": [
            {"doc": "A", "article": 1},
            {"doc": "B", "article": 2},
        ],
    }]
    result = evaluate(lambda _: np.array([0, 2, 1]), snapshot, questions)
    assert result["hit@1"] == 1.0
    assert result["all_gold_recall@1"] == 0.5
    assert result["all_gold_recall@3"] == 1.0
