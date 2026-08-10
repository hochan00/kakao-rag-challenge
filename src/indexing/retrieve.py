"""하이브리드 검색기: 희소(문자 n-gram) + 밀집(bge-m3)을 RRF로 결합한다.

항(unit) 단위로 점수를 내고 부모 조(article)로 max 집계한다. 반환 계약이
`[문서명, 조번호]` 1~4개이므로 최종 출력 단위는 조다.
"""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

RRF_K = 60          # RRF 상수. 상위 순위 가중을 완만하게 하는 표준값
CANDIDATE_DEPTH = 8  # 각 검색기에서 RRF에 넘길 조 후보 수


def _rrf(rankings: list[np.ndarray], n: int, depth: int) -> np.ndarray:
    """여러 순위 배열을 Reciprocal Rank Fusion으로 결합한 점수를 반환한다."""
    fused = np.zeros(n)
    for ranked in rankings:
        for rank, idx in enumerate(ranked[:depth]):
            fused[idx] += 1.0 / (RRF_K + rank + 1)
    return fused


def _aggregate(unit_scores: np.ndarray, parents: np.ndarray, n_articles: int) -> np.ndarray:
    """항 점수를 부모 조로 max 집계한다. 평균은 긴 조문에 길이 페널티를 준다."""
    scores = np.full(n_articles, -np.inf)
    np.maximum.at(scores, parents, unit_scores)
    return scores


class HybridRetriever:
    def __init__(self, snapshot: dict, dense_matrix: np.ndarray | None = None):
        self.articles = snapshot["articles"]
        self.units = snapshot["units"]
        self.parents = np.array([u["parent"] for u in self.units])
        self.n = len(self.articles)
        self.vectorizer = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True
        )
        self.sparse_matrix = self.vectorizer.fit_transform(
            u["embed_text"] for u in self.units
        )
        self.dense_matrix = dense_matrix

    def _sparse_article_scores(self, question: str) -> np.ndarray:
        q = self.vectorizer.transform([question])
        unit_scores = (self.sparse_matrix @ q.T).toarray().ravel()
        return _aggregate(unit_scores, self.parents, self.n)

    def _dense_article_scores(self, query_vector: np.ndarray) -> np.ndarray:
        unit_scores = self.dense_matrix @ query_vector.ravel()
        return _aggregate(unit_scores, self.parents, self.n)

    def rank(
        self,
        question: str,
        query_vector: np.ndarray | None = None,
        depth: int = CANDIDATE_DEPTH,
    ) -> np.ndarray:
        """조 인덱스를 관련도 내림차순으로 반환한다."""
        sparse = self._sparse_article_scores(question)
        if query_vector is None or self.dense_matrix is None:
            return np.argsort(-sparse)
        dense = self._dense_article_scores(query_vector)
        fused = _rrf([np.argsort(-sparse), np.argsort(-dense)], self.n, depth)
        # RRF 동점은 밀집 점수로 가른다. 두 검색기 모두 상위에 못 올린 조는
        # fused가 0이므로 dense 점수 순으로 뒤에 붙는다.
        return np.lexsort((-dense, -fused))

    def retrieve(
        self,
        question: str,
        query_vector: np.ndarray | None = None,
        max_articles: int = 4,
    ) -> list[dict]:
        """반환 계약에 맞춘 근거 조문 1~max_articles개를 관련도 순으로 반환한다."""
        ranked = self.rank(question, query_vector)[:max_articles]
        return [self.articles[i] for i in ranked]
