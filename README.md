# kakao-rag-challenge

카카오부트캠프 성능개선대회 — 카카오 약관 QA RAG 결과기

## 대회 개요

- 카카오 약관 질문에 답하는 RAG 결과기를 만들어 겨루는 대회
- 팀 인원: 2명
- 로컬 오픈소스 모델 사용 : Qwen2.5-Instruct 계열

### 진행 구조

| 일차 | 내용 |
| --- | --- |
| 1일차 | 결과기 개발 |
| 2일차 | 13:30 제출 마감 → 비공개 30문항 채점 (예선, 상위 5팀 선발) → 학생 평가기 제작 |
| 3일차 | 본선 5팀 익명 교차평가 → 발표 → 시상식 |

본선 점수: 발표 50 + 결과기 30 + 교차평가 20

## 레포 구조

```
repo/
├── src/
├── data/
├── docs/
│   └── TROUBLESHOOTING.md
├── main.py
├── pyproject.toml
└── README.md
```

## 대상 약관 문서

| # | 문서명 | 기준일 | 원문 |
| --- | --- | --- | --- |
| 1 | 카카오계정 약관 | 2026년 5월 29일 | [약관 원문](https://www.kakao.com/policy/terms?type=a&lang=ko) |
| 2 | 카카오 위치정보 이용약관 | 2026년 7월 16일 | [약관 원문](https://www.kakao.com/policy/location?lang=ko) |
| 3 | 카카오 통합서비스약관 | 2026년 5월 29일 | [약관 원문](https://www.kakao.com/policy/terms?type=ts&lang=ko) |
| 4 | 카카오 통합 약관 (아카이브) | 2022년 8월 25일 | [공식 아카이브](https://www.kakao.com/policy/kakaoTerms) |

