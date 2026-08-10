"""result_generator.ipynb의 1번 셀에 14조 결과기 구현을 주입한다.

원본 셀의 고정 영역(머리말 계약 주석, OFFICIAL_DOCUMENT_NAMES 블록, FastAPI 연결부)은
그대로 두고 "1. 팀별 자유 구현 영역"부터 FastAPI 블록 직전까지만 교체한다.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from indexing.notebook_payload import make_payload  # noqa: E402

ROOT = Path(__file__).parent.parent
NOTEBOOK = ROOT / "src" / "result_generator.ipynb"
IMPL = ROOT / "src" / "indexing" / "notebook_impl.py"
SNAPSHOT = ROOT / "artifacts" / "terms_snapshot.json"

FREE_START = "# =====================================================================================\n# 1. 팀별 자유 구현 영역"
FIXED_START = "# =====================================================================================\n# 2. 고정 FastAPI 연결 영역"


def chunk_payload(payload: str, width: int = 100) -> str:
    """base64 문자열을 파이썬 소스에 넣을 수 있게 줄 단위 리터럴로 나눈다."""
    return "\n".join(
        f'    "{payload[i:i + width]}"' for i in range(0, len(payload), width)
    )


def build_cell_source(original: str) -> str:
    head_end = original.index(FREE_START)
    fixed_start = original.index(FIXED_START)
    payload, sha = make_payload(SNAPSHOT)
    impl = IMPL.read_text(encoding="utf-8")
    impl = impl.replace("__SHA256__", sha).replace(
        "__PAYLOAD__", chunk_payload(payload)
    )
    return original[:head_end] + impl + "\n\n" + original[fixed_start:]


def main():
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    original = "".join(nb["cells"][0]["source"])
    if FREE_START not in original or FIXED_START not in original:
        raise SystemExit("1번 셀에서 고정 영역 경계를 찾지 못했습니다.")
    new_source = build_cell_source(original)
    nb["cells"][0]["source"] = new_source.splitlines(keepends=True)
    nb["cells"][0]["outputs"] = []
    nb["cells"][0]["execution_count"] = None
    NOTEBOOK.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"1번 셀 {len(new_source):,}자 주입 완료 → {NOTEBOOK}")
    print(f"2번 셀은 손대지 않음 ({len(''.join(nb['cells'][1]['source'])):,}자)")


if __name__ == "__main__":
    main()
