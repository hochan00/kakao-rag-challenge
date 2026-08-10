# 약관 4종 인덱싱 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**실행자:** Opus 5 · **작성일:** 2026-08-10

**Goal:** `data/`의 약관 4종을 파싱→청킹→검증→임베딩까지 완료하고, 공개 골드셋 실측으로 임베딩 모델을 확정해 재현 가능한 인덱스 아티팩트를 만든다.

**Architecture:** 문서별 규칙 분기 파서 → 항(unit) 청크 + 조(article) 부모 매핑 → 문서·청크 검증 게이트 통과 시 `terms_snapshot.json` 생성 → TF-IDF 희소 + dense 임베딩(numpy 전수 cosine, 벡터 DB 없음) → 공개 10문항의 다중 정답 보존 Recall 실측으로 모델 확정 → 모델 revision과 스냅샷 해시를 고정한 dense 인덱스 생성.

**Tech Stack:** Python 3.13, uv, numpy, scikit-learn, sentence-transformers, pytest

**참고 문서(참고용일 뿐, 본 계획의 코드·수치가 우선):**
- [인덱싱 설계](2026-08-10-indexing-design.md) — 설계 결정과 실측 근거
- [데이터 특성](2026-08-10-terms-data-characteristics.md) — "3+1" 구조 분기, near-duplicate 지도, 노이즈 목록
- [규정 경계](2026-08-10-official-rules-and-adversarial-review.md) — 임베딩 모델·DB는 규정 미고정(로컬 실행이면 가능 해석)
- [상위 전략](2026-08-10-indexing-and-result-generator-strategy.md) — 최종 제출은 노트북 1번 셀 내장(본 계획 범위 밖)

## Global Constraints

- 문서명은 반환 계약 4종 문자열 그대로: `카카오계정 약관` `카카오 위치정보 이용약관` `카카오 통합서비스약관` `카카오 통합 약관`
- `data/` 원문 4개 파일은 읽기 전용. 수정 금지 (`카카오_통합서비스약관`은 확장자 없음 — glob 금지, 명시 매핑)
- 원격 임베딩·리랭커·관리형 벡터 DB 금지. 로컬 모델 추론만
- 벡터 DB 미사용: numpy float32 행렬 + 전수 cosine (229 청크 규모에서 ANN은 손해만)
- 검증 게이트 실패 시 스냅샷 생성 중단 (조 수 17/16/18/21, 항 수 47/43/63/76, 골드 12쌍 커버리지, 본문 무손실 등)
- 개발 환경은 로컬 macOS(CPU/MPS). Colab 이식은 별도 단계
- 커밋은 태스크 단위, 메시지는 한국어 `feat:`/`test:` 프리픽스 유지

## 파일 구조

```
src/indexing/
├── __init__.py        # 빈 파일
├── parse.py           # 문서별 파싱 + 정규화 + 노이즈 제거
├── chunk.py           # 항 단위 청크 + 부모 조 매핑
├── validate.py        # 검증 게이트
├── build.py           # 스냅샷 빌드 CLI (파싱→검증→JSON)
├── eval_retrieval.py  # Hit@k / all-gold Recall@k / MRR 평가 + 모델 비교
└── embed.py           # 확정 모델의 dense 인덱스 + 재현 메타데이터 생성
tests/
├── test_parse.py
├── test_chunk.py
├── test_validate.py
├── test_eval_retrieval.py
└── test_embed.py
artifacts/             # 생성물 (gitignore)
├── terms_snapshot.json
├── selected_model.json
├── dense_embeddings.npy
├── dense_parents.npy
└── dense_index_manifest.json
```

---

### Task 1: 파서 (parse.py)

**Files:**
- Create: `src/indexing/__init__.py`, `src/indexing/parse.py`
- Test: `tests/test_parse.py`
- Modify: `pyproject.toml` (의존성 + pytest 설정), `.gitignore` (`artifacts/` 추가)

**Interfaces:**
- Produces: `DOCS: dict[str, tuple[str, str, str]]` (문서명 → 파일명, 시행일, 마커 방식), `parse_document(name: str, data_dir: Path) -> ParsedDoc`, `ParsedDoc(name, effective_date, marker, sha256, articles: list[Article], footer: dict)`, `Article(doc, n, title, chapter, lines)` with `.text` property

- [ ] **Step 1: 의존성과 pytest 설정**

```bash
uv add numpy scikit-learn pytest
```

`pyproject.toml`에 추가:

```toml
[tool.pytest.ini_options]
pythonpath = ["src"]
```

`.gitignore`에 `artifacts/` 한 줄 추가.

- [ ] **Step 2: 실패하는 테스트 작성**

```python
# tests/test_parse.py
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
        ("카카오 통합 약관", "서비스(위치기반서비스 포함) 관련 문의사항이 있으시면 언제든지 고객센터에 방문 또는 연락해 주시기 바랍니다."),
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
```

- [ ] **Step 3: 실패 확인**

Run: `uv run pytest tests/test_parse.py -q`
Expected: FAIL — `ModuleNotFoundError: indexing`

- [ ] **Step 4: 구현**

```python
# src/indexing/parse.py
import hashlib
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

# 문서명(반환 계약 고정) → (파일명, 시행일, 항 마커 방식)
DOCS = {
    "카카오계정 약관": ("카카오계정_약관.txt", "2026-05-29", "circled"),
    "카카오 위치정보 이용약관": ("카카오_위치정보_이용약관.txt", "2026-07-16", "circled"),
    "카카오 통합서비스약관": ("카카오_통합서비스약관", "2026-05-29", "circled"),  # 확장자 없음
    "카카오 통합 약관": ("카카오_통합_약관.txt", "2022-08-25", "numdot"),
}

# 조 헤더: `제 1 조 (목적)`(3개 문서)과 `제1조 목적`(통합 약관)을 모두 흡수
ART = re.compile(r"^제\s*(\d+)\s*조\s*[\(（]?\s*([^)）]*?)\s*[\)）]?\s*$")
CHAP = re.compile(r"^제\s*(\d+)\s*장\s*(.*)$")
# 웹 복사 잔여 UI 문구 (데이터 특성 문서 §4). 부분 문자열 검색으로 정상
# 조문까지 지우지 않도록 줄 전체가 알려진 문구와 같을 때만 제거한다.
# 첫 패턴은 현재 스냅샷에서는 이미 삭제됐지만 원본 복원 시에도 안전하게 제외한다.
NOISE = re.compile(
    r"^(?:위치정보 제공 현황 자세히 보기|"
    r"위치정보 전용문의 게시판 \(바로가기\)|<시행일자>|"
    r"서비스\(위치기반서비스 포함\) 관련 문의사항이 있으시면 언제든지 "
    r"고객센터에 방문 또는 연락해 주시기 바랍니다\.)$"
)
FOOTER = re.compile(r"^(공고일자|시행일자)\s*:\s*(\d{4})년\s*(\d{1,2})월\s*(\d{1,2})일$")

_TRANS = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"', "・": "·"})


def normalize_line(s: str) -> str:
    s = unicodedata.normalize("NFC", s).translate(_TRANS)
    return re.sub(r"[ \t\u00a0]+", " ", s).strip()


@dataclass
class Article:
    doc: str
    n: int
    title: str
    chapter: int | None
    lines: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n".join(self.lines)


@dataclass
class ParsedDoc:
    name: str
    effective_date: str
    marker: str
    sha256: str
    articles: list[Article]
    footer: dict[str, str]


def parse_document(name: str, data_dir: Path) -> ParsedDoc:
    fname, expected_date, marker = DOCS[name]
    raw = (data_dir / fname).read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    articles: list[Article] = []
    footer: dict[str, str] = {}
    cur: Article | None = None
    chapter: int | None = None
    for line in raw.decode("utf-8").splitlines():
        s = normalize_line(line)
        if not s or NOISE.fullmatch(s):
            continue
        if m := FOOTER.match(s):
            footer[m.group(1)] = f"{m.group(2)}-{int(m.group(3)):02d}-{int(m.group(4)):02d}"
            continue
        if m := ART.match(s):
            cur = Article(doc=name, n=int(m.group(1)), title=m.group(2), chapter=chapter)
            articles.append(cur)
            continue
        if m := CHAP.match(s):
            chapter = int(m.group(1))
            continue
        if cur is not None:  # 문서 제목 줄 등 첫 조 이전 텍스트는 버린다
            cur.lines.append(s)
    return ParsedDoc(name, expected_date, marker, sha, articles, footer)
```

주의점 (설계 문서 근거):
- `ART`는 줄 전체 매칭(`^...$`)이라 본문 중 상호참조(`제14조 제1항에 따라...`)는 걸리지 않는다. 단, 본문 줄이 `제N조`로 **시작**하면 오탐 가능 — Step 2의 조 수 테스트가 이를 잡는다.
- `공고일자`/`시행일자` 줄을 본문에서 분리하지 않으면 마지막 조 본문에 섞인다. `FOOTER`가 이를 막는다.
- 위치정보 약관 제16조의 사업자 정보(주소·전화·책임자)는 정식 조문이므로 유지한다. 현재 working tree에 남은 노이즈는 제16조 뒤 2줄이다. 저장소 기준 원문에는 제7조 뒤 UI 링크와 통합 약관 시행일 뒤 고객센터 안내도 있으므로 네 문구를 모두 exact match로 제외한다. 이로써 두 원문 상태의 정규화 본문은 같고, 어느 상태에서도 정상 조문을 부분 문자열로 삭제하지 않는다.

- [ ] **Step 5: 통과 확인 후 커밋**

Run: `uv run pytest tests/test_parse.py -q`
Expected: PASS (8 tests)

```bash
git add src/indexing tests/test_parse.py pyproject.toml uv.lock .gitignore
git commit -m "feat: 약관 4종 문서별 분기 파서 구현"
```

---

### Task 2: 청커 (chunk.py)

**Files:**
- Create: `src/indexing/chunk.py`
- Test: `tests/test_chunk.py`

**Interfaces:**
- Consumes: `parse.ParsedDoc`, `parse.Article`
- Produces: `build_units(docs: list[ParsedDoc]) -> tuple[list[Article], list[Unit]]`, `Unit(parent: int, order: int, text: str, embed_text: str)` — `parent`는 전역 articles 리스트 인덱스

- [ ] **Step 1: 실패하는 테스트 작성**

```python
# tests/test_chunk.py
from pathlib import Path
import pytest
from indexing.parse import DOCS, parse_document
from indexing.chunk import build_units

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
```

- [ ] **Step 2: 실패 확인**

Run: `uv run pytest tests/test_chunk.py -q`
Expected: FAIL — `ModuleNotFoundError: indexing.chunk`

- [ ] **Step 3: 구현**

```python
# src/indexing/chunk.py
import re
from dataclasses import dataclass

from indexing.parse import Article, ParsedDoc

# 항 헤드 마커: 3개 문서는 원문자(①~⑳)가 항, `N.`은 호(직전 항에 붙임).
# 통합 약관만 `N.`이 항이고 호는 마커가 없다. (데이터 특성 문서 §1)
HEAD = {
    "circled": re.compile(r"^[①-⑳]"),
    "numdot": re.compile(r"^\d+\."),
}


@dataclass
class Unit:
    parent: int      # 전역 articles 리스트 인덱스
    order: int       # 조 내 순번 (1부터)
    text: str
    embed_text: str  # "{문서명} 제{N}조({제목})\n{본문}" — near-duplicate 문서 판별용


def build_units(docs: list[ParsedDoc]) -> tuple[list[Article], list[Unit]]:
    articles: list[Article] = []
    units: list[Unit] = []
    for d in docs:
        head = HEAD[d.marker]
        for art in d.articles:
            idx = len(articles)
            articles.append(art)
            blocks: list[list[str]] = []
            for ln in art.lines:
                if head.match(ln) or not blocks:
                    blocks.append([ln])       # 새 항 시작 (조 첫 줄 전단 포함)
                else:
                    blocks[-1].append(ln)     # 호·마커 없는 줄은 직전 항에 붙임
            prefix = f"{art.doc} 제{art.n}조({art.title})"
            for k, blk in enumerate(blocks, 1):
                text = "\n".join(blk)
                units.append(Unit(idx, k, text, f"{prefix}\n{text}"))
    return articles, units
```

- [ ] **Step 4: 통과 확인 후 커밋**

Run: `uv run pytest tests/test_chunk.py -q`
Expected: PASS (5 tests). 문서별 47/43/63/76, 합계 229가 정확히 일치해야 한다.

```bash
git add src/indexing/chunk.py tests/test_chunk.py
git commit -m "feat: 항 단위 청커와 부모 조 매핑 구현"
```

---

### Task 3: 검증 게이트와 스냅샷 빌드 (validate.py, build.py)

**Files:**
- Create: `src/indexing/validate.py`, `src/indexing/build.py`
- Test: `tests/test_validate.py`

**Interfaces:**
- Consumes: `parse.ParsedDoc`, `chunk.build_units`
- Produces: `validate(docs: list[ParsedDoc], articles: list[Article], units: list[Unit], gold_pairs: set[tuple[str, int]]) -> list[str]` (빈 리스트 = 통과), `build_snapshot(data_dir: Path, out_path: Path) -> dict` — 스냅샷 스키마는 아래 JSON 형태

- [ ] **Step 1: 실패하는 테스트 작성**

```python
# tests/test_validate.py
import json
from pathlib import Path
import pytest
from indexing.chunk import build_units
from indexing.parse import DOCS, parse_document
from indexing.validate import validate
from indexing.build import load_gold_pairs, build_snapshot

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
    from copy import deepcopy
    docs, _, _ = parsed
    broken = deepcopy(docs)           # module-scope fixture 오염 방지
    broken[0].articles.pop()          # 조 하나 제거
    articles, units = build_units(broken)
    assert validate(broken, articles, units, set())

def test_unit_loss_detected_without_fixture_contamination(parsed):
    from copy import deepcopy
    docs, articles, units = parsed
    broken_units = deepcopy(units)
    broken_units.pop()
    errors = validate(docs, articles, broken_units, set())
    assert any("항 수" in e or "본문 손실" in e for e in errors)
    assert len(units) == 229            # module-scope fixture 원본 보존

def test_build_snapshot(tmp_path):
    out = tmp_path / "terms_snapshot.json"
    snap = build_snapshot(DATA, out)
    assert out.exists()
    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert len(loaded["articles"]) == 72
    assert len(loaded["units"]) == 229
    assert {d["name"] for d in loaded["documents"]} == set(DOCS)
    assert all("sha256" in d for d in loaded["documents"])
    # units의 parent가 articles 인덱스 범위 안
    assert all(0 <= u["parent"] < 72 for u in loaded["units"])
```

- [ ] **Step 2: 실패 확인**

Run: `uv run pytest tests/test_validate.py -q`
Expected: FAIL — `ModuleNotFoundError: indexing.validate`

- [ ] **Step 3: 구현**

```python
# src/indexing/validate.py
from indexing.chunk import Unit
from indexing.parse import Article, ParsedDoc

EXPECTED_COUNTS = {
    "카카오계정 약관": 17,
    "카카오 위치정보 이용약관": 16,
    "카카오 통합서비스약관": 18,
    "카카오 통합 약관": 21,
}
EXPECTED_UNITS = {
    "카카오계정 약관": 47,
    "카카오 위치정보 이용약관": 43,
    "카카오 통합서비스약관": 63,
    "카카오 통합 약관": 76,
}


def validate(
    docs: list[ParsedDoc],
    articles: list[Article],
    units: list[Unit],
    gold_pairs: set[tuple[str, int]],
) -> list[str]:
    """검증 실패 사유 목록을 반환한다. 빈 리스트면 통과."""
    errors: list[str] = []
    if [d.name for d in docs] != list(EXPECTED_COUNTS):
        errors.append(f"문서 집합/순서 불일치: {[d.name for d in docs]}")
    for d in docs:
        nums = [a.n for a in d.articles]
        if len(nums) != EXPECTED_COUNTS[d.name]:
            errors.append(f"{d.name}: 조 수 {len(nums)} != {EXPECTED_COUNTS[d.name]}")
        if nums != list(range(1, len(nums) + 1)):
            errors.append(f"{d.name}: 조 번호 불연속 {nums}")
        for a in d.articles:
            if not a.title:
                errors.append(f"{d.name} 제{a.n}조: 제목 없음")
            if len(a.text) < 50:
                errors.append(f"{d.name} 제{a.n}조: 본문 {len(a.text)}자")
        if d.footer.get("시행일자") != d.effective_date:
            errors.append(f"{d.name}: 시행일 {d.footer.get('시행일자')} != {d.effective_date}")
    have = {(d.name, a.n) for d in docs for a in d.articles}
    for pair in sorted(gold_pairs - have):
        errors.append(f"골드 누락: {pair}")

    if len(articles) != sum(EXPECTED_COUNTS.values()):
        errors.append(f"전역 조 수 {len(articles)} != 72")
    if len(units) != sum(EXPECTED_UNITS.values()):
        errors.append(f"전역 항 수 {len(units)} != 229")

    valid_units: list[Unit] = []
    for u in units:
        if not 0 <= u.parent < len(articles):
            errors.append(f"유효하지 않은 parent: {u.parent}")
        else:
            valid_units.append(u)
    actual_units = {
        name: sum(1 for u in valid_units if articles[u.parent].doc == name)
        for name in EXPECTED_UNITS
    }
    for name, expected in EXPECTED_UNITS.items():
        if actual_units[name] != expected:
            errors.append(f"{name}: 항 수 {actual_units[name]} != {expected}")

    for idx, article in enumerate(articles):
        own = [u for u in valid_units if u.parent == idx]
        if [u.order for u in own] != list(range(1, len(own) + 1)):
            errors.append(f"{article.doc} 제{article.n}조: 항 순서 불연속")
        if "\n".join(u.text for u in own) != article.text:
            errors.append(f"{article.doc} 제{article.n}조: 청킹 본문 손실")
    return errors
```

```python
# src/indexing/build.py
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
```

- [ ] **Step 4: 통과 확인 + 실제 빌드**

Run: `uv run pytest tests/ -q` — 전체 PASS
Run: `uv run python -m indexing.build` — `조 72 / 항 229 → artifacts/terms_snapshot.json` 출력 확인

- [ ] **Step 5: 커밋**

```bash
git add src/indexing/validate.py src/indexing/build.py tests/test_validate.py
git commit -m "feat: 검증 게이트와 terms_snapshot 빌드 추가"
```

---

### Task 4: 희소 검색 기준선 평가 (eval_retrieval.py)

**Files:**
- Create: `src/indexing/eval_retrieval.py`
- Test: `tests/test_eval_retrieval.py`

**Interfaces:**
- Consumes: `artifacts/terms_snapshot.json`, `data/gold_questions_public10.json`
- Produces: `sparse_ranker(snapshot) -> Callable[[str], np.ndarray]`, `evaluate(rank_fn, snapshot, questions) -> dict` — 키 `hit@1 hit@3 hit@5 all_gold_recall@1 all_gold_recall@3 all_gold_recall@4 all_gold_recall@5 MRR`

- [ ] **Step 1: 실패하는 계약·다중 정답 지표 테스트 작성**

```python
# tests/test_eval_retrieval.py
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
```

- [ ] **Step 2: 실패 확인**

Run: `uv run pytest tests/test_eval_retrieval.py -q`
Expected: FAIL — `ModuleNotFoundError: indexing.eval_retrieval`

- [ ] **Step 3: 구현**

```python
# src/indexing/eval_retrieval.py
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
```

- [ ] **Step 4: 테스트와 기준선 실행**

Run: `uv run pytest tests/test_eval_retrieval.py -q` — PASS (2 tests)
Run: `uv run python -m indexing.eval_retrieval`
Expected: 지표 딕셔너리 출력, 오류 없음. 출력 값을 이 문서 하단 "실측 기록"에 그대로 붙여넣는다.

- [ ] **Step 5: 커밋**

```bash
git add src/indexing/eval_retrieval.py tests/test_eval_retrieval.py
git commit -m "feat: 희소 검색 기준선 평가 추가"
```

---

### Task 5: 밀집 모델 비교와 확정

**Files:**
- Modify: `src/indexing/eval_retrieval.py` (dense_ranker와 비교 main 추가)
- Modify: `docs/harry/indexing/indexing-plan.md` (실측 기록 채움)

**Interfaces:**
- Consumes: Task 4의 `evaluate`, `article_ranking`
- Produces: `dense_ranker(snapshot, model_name, revision, query_prefix="", passage_prefix="") -> Callable[[str], np.ndarray]`; 반환 함수의 `truncated_units` 속성에 모델 최대 길이를 넘는 청크를 기록

- [ ] **Step 1: sentence-transformers 설치**

```bash
uv add sentence-transformers
```

macOS에서 torch가 함께 설치된다. 실패 시 `uv add torch sentence-transformers`로 분리 설치.

- [ ] **Step 2: dense_ranker 추가**

`eval_retrieval.py`에 추가:

```python
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
```

`main()`을 비교 실행으로 교체:

```python
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
```

- [ ] **Step 3: 비교 실행**

Run: `uv run python -m indexing.eval_retrieval`
Expected: 세 줄 출력. 최초 실행은 모델 다운로드(bge-m3 약 2.3GB)로 수 분 소요.

- [ ] **Step 4: 모델 확정 — 결정 규칙**

공개 파일은 10문항이지만 P02가 3개 정답 조문을 가지므로, 하나만 회수해도 만점이 되는 `hit@3`를 모델 선택의 주 지표로 쓰지 않는다. 또한 “1문항 차이는 잡음”이라는 통계적 근거 없는 임계값을 두지 않는다. 아래 순서로 기계적으로 결정한다.

1. **무손실 게이트:** `truncated_units`가 비어 있고 229개 벡터가 모두 finite·L2 정규화되어야 후보 자격이 있다. 공개 골드에 포함되지 않은 긴 청크도 비공개 질문의 근거가 될 수 있으므로, “골드 조문만 안 잘리면 허용”하지 않는다. e5-base가 이 게이트를 실패하면 이 계획에는 2차 분할 구현이 없으므로 채택하지 않는다.
2. **계약 지표 우선순위:** 자격 후보 중 `all_gold_recall@4`가 높은 모델을 선택한다. 같으면 `all_gold_recall@3`, 다시 같으면 MRR 순으로 비교한다. `@4`는 결과 계약이 실제 근거를 최대 4개 반환할 수 있고 P02의 세 정답을 모두 보존해야 한다는 점에 맞춘다.
3. **완전 동률:** 위 세 지표가 모두 같을 때만 전체 229개 인코딩 시간과 임베딩 바이트 수가 작은 모델을 선택한다. e5-base가 절단 게이트를 통과한 경우에만 이 자원 우위를 사용할 수 있다.
4. 각 후보의 resolved Hugging Face revision, 접두어, `truncated_units`, 인코딩 시간·바이트 수, 집계 지표를 `selected_model.json.candidates`에 함께 기록한다. 작은 공개셋에서 얻은 승패를 통계적 유의성으로 표현하지 않는다.

- [ ] **Step 5: 실측 기록 후 커밋**

아래 "실측 기록" 표를 채우고 확정 모델명·resolved revision·접두어·절단 청크 수를 명시한다.

```bash
git add src/indexing/eval_retrieval.py docs/harry/indexing/indexing-plan.md
git commit -m "feat: 밀집 모델 비교 평가 및 임베딩 모델 확정"
```

---

### Task 6: 확정 모델 임베딩 아티팩트 생성

**Files:**
- Create: `src/indexing/embed.py`
- Test: `tests/test_embed.py`

**Interfaces:**
- Consumes: `artifacts/terms_snapshot.json`, Task 5가 생성한 `artifacts/selected_model.json`
- Produces: `build_dense_index(...) -> dict`, `artifacts/dense_embeddings.npy`, `artifacts/dense_parents.npy`, `artifacts/dense_index_manifest.json`

- [ ] **Step 1: 실패하는 직렬화·무결성 테스트 작성**

```python
# tests/test_embed.py
import hashlib
import json

import numpy as np

from indexing.embed import save_dense_index


def test_save_dense_index_records_reproducibility_metadata(tmp_path):
    snapshot_path = tmp_path / "terms_snapshot.json"
    snapshot = {
        "units": [
            {"parent": 0, "embed_text": "A"},
            {"parent": 1, "embed_text": "B"},
        ]
    }
    snapshot_path.write_text(
        json.dumps(snapshot, ensure_ascii=False), encoding="utf-8"
    )
    vectors = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    parents = np.array([0, 1], dtype=np.int64)

    manifest = save_dense_index(
        snapshot_path, tmp_path, vectors, parents,
        model_name="org/model", model_revision="a" * 40,
        query_prefix="query: ", passage_prefix="passage: ",
    )

    assert np.array_equal(
        np.load(tmp_path / "dense_embeddings.npy", allow_pickle=False), vectors
    )
    assert np.array_equal(
        np.load(tmp_path / "dense_parents.npy", allow_pickle=False), parents
    )
    assert manifest["rows"] == 2
    assert manifest["dimension"] == 2
    assert manifest["dtype"] == "float32"
    assert manifest["model_revision"] == "a" * 40
    assert manifest["snapshot_sha256"] == hashlib.sha256(
        snapshot_path.read_bytes()
    ).hexdigest()
```

- [ ] **Step 2: 실패 확인**

Run: `uv run pytest tests/test_embed.py -q`
Expected: FAIL — `ModuleNotFoundError: indexing.embed`

- [ ] **Step 3: 구현**

```python
# src/indexing/embed.py
import hashlib
import json
from pathlib import Path

import numpy as np


def save_dense_index(
    snapshot_path: Path,
    out_dir: Path,
    vectors: np.ndarray,
    parents: np.ndarray,
    *,
    model_name: str,
    model_revision: str,
    query_prefix: str,
    passage_prefix: str,
) -> dict:
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    vectors = np.asarray(vectors, dtype=np.float32)
    parents = np.asarray(parents, dtype=np.int64)
    if vectors.ndim != 2 or vectors.shape[0] != len(snapshot["units"]):
        raise ValueError("embedding row count does not match snapshot units")
    if parents.shape != (len(snapshot["units"]),):
        raise ValueError("parent row count does not match snapshot units")
    if not np.isfinite(vectors).all():
        raise ValueError("non-finite embedding")
    if not np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-4):
        raise ValueError("embeddings are not L2-normalized")

    out_dir.mkdir(parents=True, exist_ok=True)
    vectors_path = out_dir / "dense_embeddings.npy"
    parents_path = out_dir / "dense_parents.npy"
    np.save(vectors_path, vectors, allow_pickle=False)
    np.save(parents_path, parents, allow_pickle=False)
    manifest = {
        "snapshot_sha256": hashlib.sha256(snapshot_path.read_bytes()).hexdigest(),
        "model_name": model_name,
        "model_revision": model_revision,
        "query_prefix": query_prefix,
        "passage_prefix": passage_prefix,
        "rows": vectors.shape[0],
        "dimension": vectors.shape[1],
        "dtype": str(vectors.dtype),
        "normalized": True,
        "embeddings_sha256": hashlib.sha256(vectors_path.read_bytes()).hexdigest(),
        "parents_sha256": hashlib.sha256(parents_path.read_bytes()).hexdigest(),
    }
    (out_dir / "dense_index_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def build_dense_index(
    snapshot_path: Path,
    out_dir: Path,
    *,
    model_name: str,
    model_revision: str,
    query_prefix: str,
    passage_prefix: str,
) -> dict:
    from sentence_transformers import SentenceTransformer

    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    model = SentenceTransformer(model_name, revision=model_revision)
    texts = [passage_prefix + u["embed_text"] for u in snapshot["units"]]
    tokenized = model.tokenizer(texts, padding=False, truncation=False)["input_ids"]
    over = [(i, len(ids)) for i, ids in enumerate(tokenized)
            if len(ids) > model.max_seq_length]
    if over:
        raise ValueError(f"would truncate units: {over}")
    vectors = model.encode(
        texts, normalize_embeddings=True, show_progress_bar=True
    )
    parents = np.array([u["parent"] for u in snapshot["units"]])
    return save_dense_index(
        snapshot_path, out_dir, vectors, parents,
        model_name=model_name, model_revision=model_revision,
        query_prefix=query_prefix, passage_prefix=passage_prefix,
    )


def main():
    selection = json.loads(
        Path("artifacts/selected_model.json").read_text(encoding="utf-8")
    )
    manifest = build_dense_index(
        Path("artifacts/terms_snapshot.json"), Path("artifacts"),
        model_name=selection["model_name"],
        model_revision=selection["model_revision"],
        query_prefix=selection["query_prefix"],
        passage_prefix=selection["passage_prefix"],
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 단위 테스트와 실제 확정 모델 빌드**

Run: `uv run pytest tests/test_embed.py -q` — PASS (1 test)

Task 5가 기록한 모델명·resolved revision·접두어를 `selected_model.json`에서 그대로 읽으므로 수동 재입력하지 않는다.

```bash
uv run python -m indexing.embed
```

Expected: manifest의 `rows`가 229이고, `snapshot_sha256`·`model_revision`·두 `.npy` SHA-256이 채워진다. `dense_embeddings.npy`는 float32이며 모든 행의 L2 norm이 1이다.

- [ ] **Step 5: 전체 회귀 테스트 후 커밋**

Run: `uv run pytest tests/ -q` — 전체 PASS

```bash
git add src/indexing/embed.py tests/test_embed.py
git commit -m "feat: 확정 모델 dense 인덱스 아티팩트 생성"
```

---

## 실측 기록 (2026-08-10 실행 완료)

공개 10문항(고유 골드 12쌍) 기준. 실행 환경 macOS CPU, Python 3.13.13.

| ranker | hit@1 | hit@3 | hit@5 | all-gold R@1 | all-gold R@3 | all-gold R@4 | MRR | 절단 청크 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sparse (char 2-5gram) | 0.90 | 1.00 | 1.00 | 0.833 | 1.00 | 1.00 | 0.950 | — |
| e5-base | 0.90 | 1.00 | 1.00 | 0.833 | 1.00 | 1.00 | 0.950 | **2개** |
| **bge-m3 (확정)** | **1.00** | 1.00 | 1.00 | **0.933** | 1.00 | 1.00 | **1.000** | **0개** |

- 항 청크 수 직접 실측: **229개** (계정 47 / 위치정보 43 / 통합서비스 63 / 통합 76)
- **확정 모델: `BAAI/bge-m3`**, revision `5617a9f61b028005a4858fdac845db406aefb181`, 접두어 없음
  - Step 4 결정 규칙 1(무손실 게이트)에서 e5-base 탈락 → 2·3단계로 갈 것 없이 확정
- **e5-base 절단 2건** (max_seq_length 512 초과):
  - unit 34 = 617토큰 → `카카오계정 약관 제12조(회원의 의무)` (1,680자)
  - unit 130 = 888토큰 → `카카오 통합서비스약관 제12조(통합서비스 이용 방법 및 주의점)` (3,482자)
  - 설계 문서가 예측한 "512토큰 모델은 2차 분할 로직이 필요하다"가 실측으로 확인됨. 이 계획에는 2차 분할이 없으므로 채택 불가
- 인코딩 비용 (229청크 1회): bge-m3 10.49초 / 937,984B, e5-base 3.51초 / 703,488B — 둘 다 무시 가능
- 스냅샷·dense 인덱스 sha256: `artifacts/terms_snapshot.json`, `artifacts/dense_index_manifest.json` 참조

### 해석 주의: 공개셋은 이미 천장이다

sparse 단독으로도 hit@3 = all-gold R@3 = 1.00이다. 공개 10문항은 **모델 간 변별력이 사실상 없다.**
bge-m3가 이긴 지표는 hit@1(0.9→1.0)과 MRR(0.95→1.0), 즉 **1문항 차이**뿐이다.

따라서 이 표를 "bge-m3가 e5-base보다 낫다"의 근거로 쓰지 않는다. 확정 근거는 **절단 게이트(무손실)** 하나이며,
이는 표본 크기와 무관한 결정론적 사실이다. 비공개 30문항에서 순위가 뒤집힐 가능성은 열려 있다.

## 완료 조건

1. `uv run pytest tests/ -q` 전체 통과
2. `uv run python -m indexing.build`가 72조·229항·골드 12쌍·본문 무손실 검증 게이트를 통과하고 `artifacts/terms_snapshot.json` 생성
3. 공개 10문항에서 sparse / e5-base / bge-m3의 Hit·all-gold Recall·MRR과 절단 감사 결과가 위 표에 기록됨
4. 임베딩 모델이 결정 규칙에 따라 확정되고 모든 후보의 resolved revision·접두어·절단 감사·지표와 선택 결과가 `artifacts/selected_model.json`에 기록됨
5. `uv run python -m indexing.embed`가 229행 float32 L2 정규화 임베딩과 부모 배열, 스냅샷·모델·배열 해시 manifest를 생성

## 비범위 (다음 계획)

- 하이브리드 결합(RRF), 문서 가산점, 다양성 규칙 — 검색 파이프라인 계획에서 다룬다
- Qwen 생성·프롬프트·retrieved 반환 계약 — 결과기 계획에서 다룬다
- 노트북 1번 셀 내장(gzip+base64) 및 Colab 이식 — 제출 패키징 계획에서 다룬다

## 검증 이력 (2026-08-10)

두 갈래로 검증했다. (1) Codex 적대적 검증 — 반영 내역은 아래 "적대적 검증 반영 이력" 참조. (2) Claude 자체 실행 검증 — 계획의 Task 1~3 코드(Codex 수정 전·후 NOISE 규칙 각각)를 원문 `data/`에 그대로 실행. **양쪽 모두 39개 검사 전부 통과.**

- 조 수 17/16/18/21, 조 번호 연속, 제목·본문(최소 85자) 전부 존재
- 시행일·공고일자 푸터 추출 4개 문서 일치, 푸터가 마지막 조 본문에 섞이지 않음
- 현재 working tree의 잔존 노이즈는 위치정보 약관 말미 2줄(`위치정보 전용문의 게시판 (바로가기)`, `<시행일자>`)뿐임을 grep으로 확인 — Codex의 exact-match 4문구 NOISE로 제거됨. 정규화 잔존 문자(곡선따옴표·중점 변형) 없음
- **유닛 수 실측 229** = 문서별 47/43/63/76 — Codex가 조인 `test_total_units`의 정확값과 일치
- 유닛 → 조 본문 무손실 복원(72개 조 전부), 통합 약관 제10조 = 5유닛, 계정 약관 제12조 호 병합 확인
- 골드 12쌍 커버리지 누락 없음
- 주의: 코드가 `int | None` 문법을 쓰므로 Python 3.10+ 필수. 반드시 `uv run`(3.13)으로 실행할 것 — 시스템 python3(3.9)에서는 dataclass 정의부터 실패한다

Task 4~6(평가·모델 비교·임베딩 아티팩트)은 실행 결과가 곧 검증이므로 사전 실행 검증 대상에서 제외했다.

## 적대적 검증 반영 이력

- 현재 `data/` 원문에서 조 수 17/16/18/21, 마커 수 42/56·36/18·57/44·0/70, 잔존 노이즈 2줄을 직접 재확인하고, 저장소 기준 원문의 추가 2줄까지 포함한 알려진 네 문구만 exact match로 제거해 두 원문 상태의 본문을 동일하게 만들었다.
- 항 수의 `229±3` 허용을 제거하고 문서별 47/43/63/76·합계 229 및 청킹 본문 무손실을 테스트와 실제 스냅샷 게이트 양쪽에서 강제했다.
- module-scope fixture를 복제한 손실 주입 테스트를 추가해 검증 테스트가 공유 fixture를 오염시키지 않으면서 청크 손실을 검출하게 했다.
- 공개 골드 10문항·고유 12쌍·P01~P10 계약을 테스트와 실제 빌드 로더 양쪽에서 강제하고, P02의 3개 정답을 하나의 hit로 축소하지 않는 all-gold Recall@k를 추가했다.
- 임의의 hit@3 문항 차이 규칙을 무손실 감사→all-gold Recall@4/@3→MRR→실측 자원 순 결정으로 교체하고, 모든 후보의 revision·절단·비용·지표를 선택 JSON에 함께 고정했다.
- 모델 선택만 하고 임베딩을 남기지 않던 누락을 보완해 229행 dense 배열·부모 배열과 스냅샷/모델/배열 해시 manifest 생성 태스크를 추가했다.
