from pathlib import Path

import pytest

from indexing.chunk import build_units
from indexing.parse import DOCS, parse_document

DATA = Path(__file__).parent.parent / "data"


@pytest.fixture(scope="module")
def parsed():
    docs = [parse_document(name, DATA) for name in DOCS]
    return build_units(docs)


def test_total_units(parsed):
    articles, units = parsed
    # 현재 읽기 전용 원문과 문서별 HEAD 규칙의 결정론적 실측값이다.
    # 노이즈/푸터 제거는 항 헤드 수를 바꾸지 않으므로 허용 오차를 두지 않는다.
    expected = {
        "카카오계정 약관": 47,
        "카카오 위치정보 이용약관": 43,
        "카카오 통합서비스약관": 63,
        "카카오 통합 약관": 76,
    }
    actual = {
        name: sum(1 for u in units if articles[u.parent].doc == name)
        for name in expected
    }
    assert actual == expected
    assert len(units) == 229


def test_every_unit_has_valid_parent(parsed):
    articles, units = parsed
    for u in units:
        assert 0 <= u.parent < len(articles)
        assert u.text  # 빈 청크 금지


def test_units_cover_article_text(parsed):
    articles, units = parsed
    # 청크를 합치면 조 본문이 손실 없이 복원된다
    for idx, art in enumerate(articles):
        joined = "\n".join(u.text for u in units if u.parent == idx)
        assert joined == art.text, f"{art.doc} 제{art.n}조 본문 손실"


def test_embed_text_prefix(parsed):
    articles, units = parsed
    u = units[0]
    a = articles[u.parent]
    assert u.embed_text.startswith(f"{a.doc} 제{a.n}조({a.title})\n")


def test_numdot_is_unit_head_only_in_tonghap(parsed):
    articles, units = parsed
    # 통합 약관 제10조는 5개 항(1.~5.)이 각각 유닛이어야 한다
    tonghap10 = [u for u in units
                 if articles[u.parent].doc == "카카오 통합 약관" and articles[u.parent].n == 10]
    assert len(tonghap10) == 5
    # 계정 약관 제12조의 1.~17. 호는 ① 유닛에 붙어야 한다 (독립 유닛 금지)
    acct12 = [u for u in units
              if articles[u.parent].doc == "카카오계정 약관" and articles[u.parent].n == 12]
    assert all(u.text[0] in "①②③④⑤⑥⑦⑧⑨⑩" for u in acct12)
