import hashlib
import json

import numpy as np

from indexing.embed import save_dense_index


def test_save_dense_index_records_reproducibility_metadata(tmp_path):
    snapshot_path = tmp_path / "terms_snapshot.json"
    snapshot = {
        "units": [
            {"parent": 0, "embed_text": "A"},
            {"parent": 1, "embed_text": "B"},
        ]
    }
    snapshot_path.write_text(
        json.dumps(snapshot, ensure_ascii=False), encoding="utf-8"
    )
    vectors = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    parents = np.array([0, 1], dtype=np.int64)

    manifest = save_dense_index(
        snapshot_path, tmp_path, vectors, parents,
        model_name="org/model", model_revision="a" * 40,
        query_prefix="query: ", passage_prefix="passage: ",
    )

    assert np.array_equal(
        np.load(tmp_path / "dense_embeddings.npy", allow_pickle=False), vectors
    )
    assert np.array_equal(
        np.load(tmp_path / "dense_parents.npy", allow_pickle=False), parents
    )
    assert manifest["rows"] == 2
    assert manifest["dimension"] == 2
    assert manifest["dtype"] == "float32"
    assert manifest["model_revision"] == "a" * 40
    assert manifest["snapshot_sha256"] == hashlib.sha256(
        snapshot_path.read_bytes()
    ).hexdigest()
