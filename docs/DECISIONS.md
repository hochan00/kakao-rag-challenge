# 의사결정 기록 (DECISIONS)

여러 대안 중 왜 이걸 선택했는지 기록한다. 정량적 실험 결과는 `experiments.md`, 환경/버그 이슈는 `TROUBLESHOOTING.md`에 별도 기록.

각 결정은 아래 형식을 따른다.

```
## [카테고리] 결정 제목

상황: 뭘 정해야 했는지 (예: 임베딩 모델 후보 3개 중 선택)
결정: 뭘 선택했는지
이유: 왜 그걸 골랐는지 (탈락한 대안과 비교 포함)
```

---

## [생성 모델] 7B 양자화 방식 — bitsandbytes(NF4) 채택, GPTQ 경로는 폐기

**상황:** T4(16GB)에서 Qwen2.5-7B-Instruct를 돌리려면 4bit 양자화가 필요하다. 두 방식을 검토: ① `Qwen2.5-7B-Instruct-GPTQ-Int4` 사전 양자화 체크포인트(`auto-gptq`/`gptqmodel` + `optimum` 필요) ② fp16 원본 + `bitsandbytes`로 로드 시점 4bit 양자화.

**시도 순서 (전부 Colab T4에서 실측):**
1. GPTQ-Int4 + `auto-gptq` → `pip install` 단계에서 `CalledProcessError`. (stderr 미확보. auto-gptq가 Colab의 torch/CUDA에 맞는 prebuilt wheel이 없어 소스 빌드하다 실패하는 흔한 패턴으로 추정)
2. bitsandbytes(NF4) → **설치·로딩 모두 정상.** 다만 fp16 원본 ~15.2GB를 받아야 함.
3. GPTQ-Int4 + `gptqmodel` → 설치 시 pip이 numpy를 교체해 세션이 깨짐(`ImportError: cannot import name '_center'`). `docs/TROUBLESHOOTING.md` 참고.
4. 3번에 `numpy==2.0.2`(Colab 기본값) 고정 추가 → numpy 문제는 해소됐고 임베딩 단계까지 정상 진행. 그러나 `optimum`이 없어 실패.
5. 4번에 `optimum` 추가 → **최종 실패.** `optimum`은 `gptqmodel>=7.0.0`을 요구하는데, numpy를 2.0.2에 고정한 탓에 pip이 numpy를 맞추느라 `gptqmodel 2.2.0`으로 내려가 버렸다. 세 요구(optimum ↔ gptqmodel 7.x ↔ numpy 2.0.2)가 동시에 성립하지 않는다.

**결정:** **bitsandbytes(NF4) 채택.** fp16 원본을 받아 로드 시점에 4bit 양자화한다. numpy는 일부러 핀을 걸지 않되(핀이 다른 패키지를 옛 버전으로 끌어내리는 게 5번의 실패 원인이었다), 설치 후 numpy가 바뀌지 않았는지 검사해 바뀌면 즉시 명확한 에러로 중단시킨다.

**이유:**
- **GPTQ 경로는 의존성이 서로 모순되어 막혔다** (위 5번). Colab 환경에서 해소할 방법을 찾지 못했다.
- **다운로드 용량은 애초에 문제가 아니었다.** 운영진 확인 결과 **채점 대상은 문항당 `/answer` 응답 시간(120초 제한)뿐이고, 모델 다운로드·로딩 시간은 무관**하다. 다운로드는 1번 셀에서 일어나므로 측정되지 않는다. 처음에 이걸 핵심 병목으로 보고 GPTQ를 선택했던 판단은 **전제가 틀렸다.**
- 참고로 관측된 다운로드 속도 편차가 커서(7.98 ~ 80.8MB/s) 15.2GB 소요 시간 추정도 크게 흔들렸는데, 위 사실로 이 논점 자체가 무의미해졌다.
- **탈락한 대안(GPTQ-Int4):** 보정 데이터셋 기반이라 이론상 품질이 약간 유리할 수 있으나, Colab에서 설치 자체가 불가능해 검증할 수 없었다.

**남은 확인 사항:** bitsandbytes NF4의 품질이 3B fp16 대비 실제로 나은지는 자체 평가셋으로 측정해야 한다(`experiments.md`). 3B 비교군은 `src/result_generator_3b.ipynb`로 분리해 두었다.
