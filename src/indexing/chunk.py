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
