"""스냅샷을 노트북 1번 셀에 넣을 gzip+base64 문자열로 만든다."""
import base64
import gzip
import hashlib
import json
from pathlib import Path


def make_payload(snapshot_path: Path) -> tuple[str, str]:
    """(base64 문자열, 원본 JSON의 sha256)을 반환한다."""
    raw = snapshot_path.read_bytes()
    payload = base64.b64encode(gzip.compress(raw, 9)).decode("ascii")
    return payload, hashlib.sha256(raw).hexdigest()


def load_payload(payload: str, expected_sha256: str) -> dict:
    """노트북에서 쓰는 역함수. 무결성 실패 시 즉시 중단한다."""
    raw = gzip.decompress(base64.b64decode(payload))
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise RuntimeError("내장 약관 스냅샷 무결성 검증 실패")
    return json.loads(raw.decode("utf-8"))


def main():
    payload, sha = make_payload(Path("artifacts/terms_snapshot.json"))
    out = Path("artifacts/notebook_payload.txt")
    out.write_text(payload, encoding="ascii")
    (out.with_suffix(".sha256")).write_text(sha, encoding="ascii")
    snap = load_payload(payload, sha)  # 왕복 검증
    print(f"payload {len(payload):,}자 → {out}")
    print(f"sha256  {sha}")
    print(f"왕복 검증 통과: 조 {len(snap['articles'])} / 항 {len(snap['units'])}")


if __name__ == "__main__":
    main()
