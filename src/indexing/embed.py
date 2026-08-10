import hashlib
import json
from pathlib import Path

import numpy as np


def save_dense_index(
    snapshot_path: Path,
    out_dir: Path,
    vectors: np.ndarray,
    parents: np.ndarray,
    *,
    model_name: str,
    model_revision: str,
    query_prefix: str,
    passage_prefix: str,
) -> dict:
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    vectors = np.asarray(vectors, dtype=np.float32)
    parents = np.asarray(parents, dtype=np.int64)
    if vectors.ndim != 2 or vectors.shape[0] != len(snapshot["units"]):
        raise ValueError("embedding row count does not match snapshot units")
    if parents.shape != (len(snapshot["units"]),):
        raise ValueError("parent row count does not match snapshot units")
    if not np.isfinite(vectors).all():
        raise ValueError("non-finite embedding")
    if not np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-4):
        raise ValueError("embeddings are not L2-normalized")

    out_dir.mkdir(parents=True, exist_ok=True)
    vectors_path = out_dir / "dense_embeddings.npy"
    parents_path = out_dir / "dense_parents.npy"
    np.save(vectors_path, vectors, allow_pickle=False)
    np.save(parents_path, parents, allow_pickle=False)
    manifest = {
        "snapshot_sha256": hashlib.sha256(snapshot_path.read_bytes()).hexdigest(),
        "model_name": model_name,
        "model_revision": model_revision,
        "query_prefix": query_prefix,
        "passage_prefix": passage_prefix,
        "rows": vectors.shape[0],
        "dimension": vectors.shape[1],
        "dtype": str(vectors.dtype),
        "normalized": True,
        "embeddings_sha256": hashlib.sha256(vectors_path.read_bytes()).hexdigest(),
        "parents_sha256": hashlib.sha256(parents_path.read_bytes()).hexdigest(),
    }
    (out_dir / "dense_index_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def build_dense_index(
    snapshot_path: Path,
    out_dir: Path,
    *,
    model_name: str,
    model_revision: str,
    query_prefix: str,
    passage_prefix: str,
) -> dict:
    from sentence_transformers import SentenceTransformer

    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    model = SentenceTransformer(model_name, revision=model_revision)
    texts = [passage_prefix + u["embed_text"] for u in snapshot["units"]]
    tokenized = model.tokenizer(texts, padding=False, truncation=False)["input_ids"]
    over = [(i, len(ids)) for i, ids in enumerate(tokenized)
            if len(ids) > model.max_seq_length]
    if over:
        raise ValueError(f"would truncate units: {over}")
    vectors = model.encode(
        texts, normalize_embeddings=True, show_progress_bar=True
    )
    parents = np.array([u["parent"] for u in snapshot["units"]])
    return save_dense_index(
        snapshot_path, out_dir, vectors, parents,
        model_name=model_name, model_revision=model_revision,
        query_prefix=query_prefix, passage_prefix=passage_prefix,
    )


def main():
    selection = json.loads(
        Path("artifacts/selected_model.json").read_text(encoding="utf-8")
    )
    manifest = build_dense_index(
        Path("artifacts/terms_snapshot.json"), Path("artifacts"),
        model_name=selection["model_name"],
        model_revision=selection["model_revision"],
        query_prefix=selection["query_prefix"],
        passage_prefix=selection["passage_prefix"],
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
