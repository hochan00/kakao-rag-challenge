"""검색 전용 베이스라인 답변 파일을 만든다.

제출 파일은 2번 공통 러너가 만든다. 이 파일은 **검색 품질만** 따로 재기 위한
로컬 기준선이며, answer는 생성 모델 없이 검색된 조문에서 추출한 것이다.
따라서 answer 품질 평가에 쓰지 않는다.
"""
import json
from pathlib import Path

import numpy as np

from indexing.retrieve import HybridRetriever

TEAM = "14"
MAX_RETRIEVED = 4


def build_baseline(
    snapshot: dict,
    questions: list[dict],
    retriever: HybridRetriever,
    query_vectors: np.ndarray,
) -> dict:
    answers = []
    for i, q in enumerate(questions):
        top = retriever.retrieve(q["question"], query_vectors[i], MAX_RETRIEVED)
        answers.append({
            "qid": q["id"],
            "retrieved": [[a["doc"], a["article"]] for a in top],
            # 생성 모델 없는 추출 베이스라인: 1위 조문 원문을 그대로 싣는다
            "answer": f"{top[0]['citation']}\n{top[0]['text']}",
        })
    return {"team": TEAM, "answers": answers}


def score(baseline: dict, questions: list[dict]) -> dict:
    gold = {q["id"]: {(g["doc"], g["article"]) for g in q["gold_articles"]} for q in questions}
    hit1 = coverage = precision = 0.0
    for a in baseline["answers"]:
        got = [tuple(r) for r in a["retrieved"]]
        g = gold[a["qid"]]
        hit1 += got[0] in g
        coverage += len(g & set(got)) / len(g)
        precision += len(g & set(got)) / len(got)
    n = len(baseline["answers"])
    return {
        "hit@1": hit1 / n,
        "all_gold_recall@4": coverage / n,
        "precision@4": precision / n,
    }


def validate_contract(baseline: dict, questions: list[dict]) -> list[str]:
    """2번 러너의 형식 검사와 같은 규칙을 미리 확인한다."""
    from indexing.parse import DOCS

    errors = []
    if baseline["team"] != TEAM:
        errors.append(f"team {baseline['team']} != {TEAM}")
    if [a["qid"] for a in baseline["answers"]] != [q["id"] for q in questions]:
        errors.append("qid 순서/집합 불일치")
    for a in baseline["answers"]:
        if not 1 <= len(a["retrieved"]) <= 4:
            errors.append(f"{a['qid']}: retrieved {len(a['retrieved'])}개 (1~4 위반)")
        for doc, article in a["retrieved"]:
            if doc not in DOCS:
                errors.append(f"{a['qid']}: 허용되지 않은 문서명 {doc!r}")
            if not isinstance(article, int):
                errors.append(f"{a['qid']}: 조번호가 정수가 아님 {article!r}")
        if len({tuple(r) for r in a["retrieved"]}) != len(a["retrieved"]):
            errors.append(f"{a['qid']}: retrieved 중복")
        if not a["answer"].strip():
            errors.append(f"{a['qid']}: answer 비어 있음")
    return errors


def main():
    snapshot = json.loads(Path("artifacts/terms_snapshot.json").read_text(encoding="utf-8"))
    questions = json.loads(
        Path("data/gold_questions_public10.json").read_text(encoding="utf-8")
    )["questions"]
    dense = np.load("artifacts/dense_embeddings.npy")
    query_vectors = np.load("artifacts/public10_query_embeddings.npy")

    retriever = HybridRetriever(snapshot, dense)
    baseline = build_baseline(snapshot, questions, retriever, query_vectors)

    errors = validate_contract(baseline, questions)
    if errors:
        raise SystemExit("형식 검사 실패:\n" + "\n".join(f"  - {e}" for e in errors))

    out = Path(f"artifacts/answers_public_{TEAM}_retrieval_baseline.json")
    out.write_text(json.dumps(baseline, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"형식 검사 통과 → {out}")
    print("검색 지표:", {k: round(v, 3) for k, v in score(baseline, questions).items()})


if __name__ == "__main__":
    main()
