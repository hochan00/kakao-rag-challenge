import hashlib
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

# 문서명(반환 계약 고정) → (파일명, 시행일, 항 마커 방식)
DOCS = {
    "카카오계정 약관": ("카카오계정_약관.txt", "2026-05-29", "circled"),
    "카카오 위치정보 이용약관": ("카카오_위치정보_이용약관.txt", "2026-07-16", "circled"),
    "카카오 통합서비스약관": ("카카오_통합서비스약관.txt", "2026-05-29", "circled"),
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
    return re.sub(r"[ \t ]+", " ", s).strip()


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
