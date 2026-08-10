# =====================================================================================
# 1. 팀별 자유 구현 영역 — 14조 결과기
# =====================================================================================
#  구성
#    · 약관 데이터: 개발 단계에서 파싱·청킹을 끝낸 스냅샷을 gzip+base64로 이 셀에 내장
#                   (운영진 안내 "필요한 데이터를 결과기 코랩 코드 안에 포함하기")
#    · 검색: 문자 n-gram 희소 + bge-m3 밀집을 RRF로 결합, 항 점수를 부모 조로 max 집계
#    · 저장: 벡터 DB 없음. 229 × 1024 numpy 행렬 전수 cosine (0.94MB)
#    · 생성: Qwen2.5-Instruct 4-bit, 검색된 근거 조문만 프롬프트에 투입
#
#  근거 수치(개발 단계 실측): 4개 약관 = 72개 조, 229개 항 청크.
#  공개 10문항 all-gold Recall@4 = 1.00 (희소 단독 1.00, 밀집 단독 1.00).
# -------------------------------------------------------------------------------------
import base64
import gzip
import hashlib
import json
import re
import subprocess
import sys as _sys

import numpy as np


def _install_team_packages():
    subprocess.run(
        [_sys.executable, "-m", "pip", "install", "-q",
         "sentence-transformers", "scikit-learn", "accelerate", "bitsandbytes"],
        check=True,
    )


_install_team_packages()

from sentence_transformers import SentenceTransformer  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig  # noqa: E402

# -------------------------------------------------------------------------------------
# 1.1 내장 약관 스냅샷 — 개발 단계에서 생성한 고정 문자열
# -------------------------------------------------------------------------------------
TERMS_SNAPSHOT_SHA256 = "__SHA256__"
TERMS_SNAPSHOT_B64 = (
__PAYLOAD__
)

# 개발 단계에서 확정한 임베딩 모델. revision을 고정해 재현성을 보장한다.
EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"
EMBEDDING_QUERY_PREFIX = ""
EMBEDDING_PASSAGE_PREFIX = ""

# T4에서 bge-m3(약 1.1GB)와 함께 올릴 생성 모델. Pro+ A100이면 7B로 올려도 된다.
GENERATION_MODEL = "Qwen/Qwen2.5-3B-Instruct"

EXPECTED_ARTICLES = 72
EXPECTED_UNITS = 229
EXPECTED_ARTICLES_PER_DOC = {
    "카카오계정 약관": 17,
    "카카오 위치정보 이용약관": 16,
    "카카오 통합서비스약관": 18,
    "카카오 통합 약관": 21,
}


def load_terms_snapshot():
    """내장 스냅샷을 풀고 무결성·구조를 검증한다. 실패하면 즉시 중단한다."""
    raw = gzip.decompress(base64.b64decode(TERMS_SNAPSHOT_B64))
    if hashlib.sha256(raw).hexdigest() != TERMS_SNAPSHOT_SHA256:
        raise RuntimeError("내장 약관 스냅샷 무결성 검증 실패")
    snapshot = json.loads(raw.decode("utf-8"))
    if len(snapshot["articles"]) != EXPECTED_ARTICLES:
        raise RuntimeError(f"조 수 {len(snapshot['articles'])} != {EXPECTED_ARTICLES}")
    if len(snapshot["units"]) != EXPECTED_UNITS:
        raise RuntimeError(f"항 수 {len(snapshot['units'])} != {EXPECTED_UNITS}")
    per_doc = {}
    for a in snapshot["articles"]:
        per_doc[a["doc"]] = per_doc.get(a["doc"], 0) + 1
    if per_doc != EXPECTED_ARTICLES_PER_DOC:
        raise RuntimeError(f"문서별 조 수 불일치: {per_doc}")
    if set(per_doc) != set(OFFICIAL_DOCUMENT_NAMES):
        raise RuntimeError(f"문서명이 반환 계약과 불일치: {sorted(per_doc)}")
    return snapshot


# -------------------------------------------------------------------------------------
# 1.2 하이브리드 검색기
# -------------------------------------------------------------------------------------
RRF_K = 60           # RRF 상수
CANDIDATE_DEPTH = 8  # 각 검색기에서 융합에 넘길 조 후보 수


class HybridRetriever:
    """희소 + 밀집을 RRF로 결합하고 항 점수를 부모 조로 max 집계한다."""

    def __init__(self, snapshot, embedder):
        self.articles = snapshot["articles"]
        self.units = snapshot["units"]
        self.parents = np.array([u["parent"] for u in self.units])
        self.n = len(self.articles)
        self.vectorizer = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True
        )
        # embed_text에는 "문서명 제N조(제목)" 접두어가 붙어 있다. 문서 간 거의 같은
        # 조항(최대 유사도 0.97, 30쌍)을 구분하는 신호가 여기서 나온다.
        texts = [u["embed_text"] for u in self.units]
        self.sparse_matrix = self.vectorizer.fit_transform(texts)
        self.embedder = embedder
        self.dense_matrix = np.asarray(
            embedder.encode(
                [EMBEDDING_PASSAGE_PREFIX + t for t in texts],
                normalize_embeddings=True,
                show_progress_bar=False,
            ),
            dtype=np.float32,
        )

    def _aggregate(self, unit_scores):
        scores = np.full(self.n, -np.inf)
        np.maximum.at(scores, self.parents, unit_scores)
        return scores

    def rank(self, question):
        q_sparse = self.vectorizer.transform([question])
        sparse = self._aggregate((self.sparse_matrix @ q_sparse.T).toarray().ravel())
        q_dense = np.asarray(
            self.embedder.encode(
                [EMBEDDING_QUERY_PREFIX + question], normalize_embeddings=True
            ),
            dtype=np.float32,
        ).ravel()
        dense = self._aggregate(self.dense_matrix @ q_dense)
        fused = np.zeros(self.n)
        for ranked in (np.argsort(-sparse), np.argsort(-dense)):
            for rank, idx in enumerate(ranked[:CANDIDATE_DEPTH]):
                fused[idx] += 1.0 / (RRF_K + rank + 1)
        # 두 검색기 모두 상위에 못 올린 조는 fused=0이므로 밀집 점수 순으로 뒤에 붙는다.
        return np.lexsort((-dense, -fused))

    def retrieve(self, question, k=4):
        return [self.articles[i] for i in self.rank(question)[:k]]


# -------------------------------------------------------------------------------------
# 1.3 Qwen2.5-Instruct 답변 생성
# -------------------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "너는 카카오 약관 질의응답 도우미다. 아래 규칙을 반드시 지킨다.\n"
    "1. 제공된 근거 조문에 있는 내용만 사용한다. 근거 밖의 사실을 추가하지 않는다.\n"
    "2. 숫자, 기간, 조건, 예외는 근거 문장 그대로 옮긴다. 바꾸거나 반올림하지 않는다.\n"
    "3. 여러 약관에 같은 내용이 있으면 실제로 사용한 근거를 모두 남긴다.\n"
    "4. 반드시 아래 JSON 형식만 출력한다. 다른 말을 덧붙이지 않는다.\n"
    '{"answer": "답변 본문", "used": ["S1", "S2"]}'
)


class AnswerGenerator:
    def __init__(self, model_name=GENERATION_MODEL):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype="float16",
                bnb_4bit_quant_type="nf4",
            ),
            device_map="auto",
        )
        self.model.eval()

    def generate(self, question, candidates):
        evidence = "\n\n".join(
            f"[S{i}] {a['citation']}\n{a['text']}" for i, a in enumerate(candidates, 1)
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"근거 조문:\n{evidence}\n\n질문: {question}"},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        output = self.model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        return self.tokenizer.decode(
            output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )


def _parse_generation(text, candidates):
    """모델 출력에서 (답변, 사용한 근거 인덱스)를 뽑는다. 실패하면 안전하게 되돌린다."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            answer = str(data.get("answer", "")).strip()
            used = []
            for token in data.get("used", []):
                m = re.fullmatch(r"S(\d+)", str(token).strip())
                if m and 1 <= int(m.group(1)) <= len(candidates):
                    used.append(int(m.group(1)) - 1)
            if answer:
                # 중복 제거 + 검색 순위 보존
                ordered = sorted(dict.fromkeys(used))
                return answer, ordered[:4]
        except json.JSONDecodeError:
            pass
    # JSON 파싱 실패: 본문을 그대로 쓰고 근거는 검색 1위로 되돌린다
    return text.strip() or "근거 조문에서 답을 찾지 못했습니다.", []


# -------------------------------------------------------------------------------------
# 1.4 전역 초기화 — 서버 기동 전에 한 번만 수행한다
# -------------------------------------------------------------------------------------
_SNAPSHOT = load_terms_snapshot()
_EMBEDDER = SentenceTransformer(EMBEDDING_MODEL, revision=EMBEDDING_REVISION)
_RETRIEVER = HybridRetriever(_SNAPSHOT, _EMBEDDER)
_GENERATOR = AnswerGenerator()
print(
    f"[결과기 준비] 조 {len(_SNAPSHOT['articles'])} / 항 {len(_SNAPSHOT['units'])} / "
    f"임베딩 {_RETRIEVER.dense_matrix.shape}"
)


def answer_question(question: str):
    """공통 러너가 질문마다 호출하는 고정 진입점입니다."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question은 비어 있지 않은 문자열이어야 합니다.")
    question = question.strip()
    candidates = _RETRIEVER.retrieve(question, k=4)
    raw = _GENERATOR.generate(question, candidates)
    answer, used = _parse_generation(raw, candidates)
    if not used:
        used = [0]  # 반환 계약상 retrieved는 최소 1개
    retrieved = [[candidates[i]["doc"], candidates[i]["article"]] for i in used]
    return {"answer": answer, "retrieved": retrieved}
