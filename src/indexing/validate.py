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
