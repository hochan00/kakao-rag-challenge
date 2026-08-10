from pathlib import Path

import pytest

from indexing.parse import DOCS, NOISE, normalize_line, parse_document

DATA = Path(__file__).parent.parent / "data"
EXPECTED_COUNTS = {
    "카카오계정 약관": 17,
    "카카오 위치정보 이용약관": 16,
    "카카오 통합서비스약관": 18,
    "카카오 통합 약관": 21,
}


@pytest.fixture(scope="module")
def docs():
    return {name: parse_document(name, DATA) for name in DOCS}


def test_article_counts(docs):
    for name, d in docs.items():
        assert len(d.articles) == EXPECTED_COUNTS[name], name


def test_article_numbers_contiguous(docs):
    for name, d in docs.items():
        assert [a.n for a in d.articles] == list(range(1, len(d.articles) + 1)), name


def test_titles_and_bodies_nonempty(docs):
    for d in docs.values():
        for a in d.articles:
            assert a.title, f"{d.name} 제{a.n}조 제목 없음"
            assert len(a.text) >= 50, f"{d.name} 제{a.n}조 본문 {len(a.text)}자"


def test_effective_date_from_footer(docs):
    for d in docs.values():
        assert d.footer["시행일자"] == d.effective_date, d.name


def test_noise_removed(docs):
    loc = docs["카카오 위치정보 이용약관"]
    full = "\n".join(a.text for a in loc.articles)
    assert "자세히 보기" not in full
    assert "바로가기" not in full
    assert "<시행일자>" not in full


def test_noise_regex_matches_only_known_raw_lines():
    # 현재 working tree에는 current만, 저장소 기준 원문에는 current+legacy가 있다.
    # 어느 상태에서도 알려진 UI/푸터 문구 외의 정상 조문을 제거하면 안 된다.
    matched = []
    for name, (fname, _, _) in DOCS.items():
        for raw_line in (DATA / fname).read_text(encoding="utf-8").splitlines():
            line = normalize_line(raw_line)
            if NOISE.fullmatch(line):
                matched.append((name, line))
    current = {
        ("카카오 위치정보 이용약관", "위치정보 전용문의 게시판 (바로가기)"),
        ("카카오 위치정보 이용약관", "<시행일자>"),
    }
    legacy = current | {
        ("카카오 위치정보 이용약관", "위치정보 제공 현황 자세히 보기"),
        (
            "카카오 통합 약관",
            "서비스(위치기반서비스 포함) 관련 문의사항이 있으시면 언제든지 "
            "고객센터에 방문 또는 연락해 주시기 바랍니다.",
        ),
    }
    assert set(matched) in (current, legacy)


def test_footer_not_in_body(docs):
    for d in docs.values():
        assert "공고일자" not in d.articles[-1].text, d.name


def test_normalization(docs):
    # 곡선 따옴표·중점 변형이 본문에 남지 않는다
    for d in docs.values():
        full = "\n".join(a.text for a in d.articles)
        for ch in "‘’“”・":
            assert ch not in full, f"{d.name}: {ch!r} 잔존"
