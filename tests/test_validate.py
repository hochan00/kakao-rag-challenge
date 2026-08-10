import json
from copy import deepcopy
from pathlib import Path

import pytest

from indexing.build import build_snapshot, load_gold_pairs
from indexing.chunk import build_units
from indexing.parse import DOCS, parse_document
from indexing.validate import validate

DATA = Path(__file__).parent.parent / "data"


@pytest.fixture(scope="module")
def parsed():
    docs = [parse_document(name, DATA) for name in DOCS]
    articles, units = build_units(docs)
    return docs, articles, units


def test_real_data_passes(parsed):
    docs, articles, units = parsed
    gold = load_gold_pairs(DATA / "gold_questions_public10.json")
    assert validate(docs, articles, units, gold) == []


def test_missing_gold_detected(parsed):
    docs, articles, units = parsed
    errors = validate(docs, articles, units, {("카카오계정 약관", 99)})
    assert any("골드 누락" in e for e in errors)


def test_truncated_gold_contract_rejected(tmp_path):
    bad = tmp_path / "gold.json"
    bad.write_text(json.dumps({"questions": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="gold 계약"):
        load_gold_pairs(bad)


def test_article_count_mismatch_detected(parsed):
    docs, _, _ = parsed
    broken = deepcopy(docs)           # module-scope fixture 오염 방지
    broken[0].articles.pop()          # 조 하나 제거
    articles, units = build_units(broken)
    assert validate(broken, articles, units, set())


def test_unit_loss_detected_without_fixture_contamination(parsed):
    docs, articles, units = parsed
    broken_units = deepcopy(units)
    broken_units.pop()
    errors = validate(docs, articles, broken_units, set())
    assert any("항 수" in e or "본문 손실" in e for e in errors)
    assert len(units) == 229            # module-scope fixture 원본 보존


def test_build_snapshot(tmp_path):
    out = tmp_path / "terms_snapshot.json"
    build_snapshot(DATA, out)
    assert out.exists()
    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert len(loaded["articles"]) == 72
    assert len(loaded["units"]) == 229
    assert {d["name"] for d in loaded["documents"]} == set(DOCS)
    assert all("sha256" in d for d in loaded["documents"])
    # units의 parent가 articles 인덱스 범위 안
    assert all(0 <= u["parent"] < 72 for u in loaded["units"])
