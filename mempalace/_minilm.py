"""Standalone all-MiniLM-L6-v2 ONNX embedder (no chromadb).

Reproduces chromadb's ``ONNXMiniLM_L6_V2`` byte-for-byte: same model archive,
same SHA-256, same cache directory, same tokenizer settings and mean pooling,
so palaces embedded by either produce identical vectors. Used when chromadb is
not installed (it is an optional extra on this branch).
"""

from __future__ import annotations

import hashlib
import os
import tarfile
import urllib.request
from functools import cached_property
from pathlib import Path
from typing import List, Optional

import numpy as np

from ._env import env_int

MODEL_NAME = "all-MiniLM-L6-v2"
DOWNLOAD_PATH = Path.home() / ".cache" / "chroma" / "onnx_models" / MODEL_NAME
EXTRACTED_FOLDER_NAME = "onnx"
ARCHIVE_FILENAME = "onnx.tar.gz"
MODEL_DOWNLOAD_URL = "https://chroma-onnx-models.s3.amazonaws.com/all-MiniLM-L6-v2/onnx.tar.gz"
MODEL_SHA256 = "913d7300ceae3b2dbc2c50d1de4baacab4be7b9380491c27fab7418616a16ec3"
# sentence-transformers' max_seq_length for this model; chromadb uses the same.
DEFAULT_MAX_TOKENS = 256
DEFAULT_BATCH_SIZE = 32


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class StandaloneMiniLM:
    DOWNLOAD_PATH = DOWNLOAD_PATH
    EXTRACTED_FOLDER_NAME = EXTRACTED_FOLDER_NAME

    def __init__(self, preferred_providers: Optional[List[str]] = None) -> None:
        import onnxruntime  # noqa: F401  (fail early, like chromadb)

        self._preferred_providers = list(preferred_providers) if preferred_providers else None

    @staticmethod
    def name() -> str:
        return "onnx_mini_lm_l6_v2"

    def max_tokens(self) -> int:
        return env_int("MEMPALACE_MINILM_MAX_TOKENS", DEFAULT_MAX_TOKENS)

    @cached_property
    def ort(self):
        import onnxruntime

        return onnxruntime

    @cached_property
    def tokenizer(self):
        from tokenizers import Tokenizer

        tok = Tokenizer.from_file(str(self.DOWNLOAD_PATH / self.EXTRACTED_FOLDER_NAME / "tokenizer.json"))
        n = self.max_tokens()
        tok.enable_truncation(max_length=n)
        tok.enable_padding(pad_id=0, pad_token="[PAD]", length=n)
        return tok

    @cached_property
    def model(self):
        so = self.ort.SessionOptions()
        so.log_severity_level = 3
        so.graph_optimization_level = self.ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        providers = [p for p in (self._preferred_providers or self.ort.get_available_providers())
                     if p != "CoreMLExecutionProvider"]
        return self.ort.InferenceSession(
            str(self.DOWNLOAD_PATH / self.EXTRACTED_FOLDER_NAME / "model.onnx"),
            providers=providers, sess_options=so)

    def _download_model_if_not_exists(self) -> None:
        folder = self.DOWNLOAD_PATH / self.EXTRACTED_FOLDER_NAME
        needed = ("config.json", "model.onnx", "special_tokens_map.json",
                  "tokenizer_config.json", "tokenizer.json", "vocab.txt")
        if all((folder / f).exists() for f in needed):
            return
        os.makedirs(self.DOWNLOAD_PATH, exist_ok=True)
        archive = str(self.DOWNLOAD_PATH / ARCHIVE_FILENAME)
        if not os.path.exists(archive) or _sha256(archive) != MODEL_SHA256:
            urllib.request.urlretrieve(MODEL_DOWNLOAD_URL, archive)  # noqa: S310 — fixed https URL
            if _sha256(archive) != MODEL_SHA256:
                os.remove(archive)
                raise ValueError("MiniLM archive failed SHA-256 verification")
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(path=self.DOWNLOAD_PATH, filter="data")

    @staticmethod
    def _normalize(v):
        norm = np.linalg.norm(v, axis=1)
        norm[norm == 0] = 1e-12
        return v / norm[:, np.newaxis]

    def _forward(self, documents, batch_size=DEFAULT_BATCH_SIZE):
        out = []
        for i in range(0, len(documents), batch_size):
            encoded = [self.tokenizer.encode(d) for d in documents[i:i + batch_size]]
            ids = np.array([e.ids for e in encoded], dtype=np.int64)
            mask = np.array([e.attention_mask for e in encoded], dtype=np.int64)
            hidden = self.model.run(None, {"input_ids": ids, "attention_mask": mask,
                                           "token_type_ids": np.zeros_like(ids)})[0]
            m = np.broadcast_to(np.expand_dims(mask, -1), hidden.shape)
            emb = np.sum(hidden * m, 1) / np.clip(m.sum(1), a_min=1e-9, a_max=None)
            out.append(self._normalize(emb).astype(np.float32))
        return np.concatenate(out)

    def __call__(self, input):  # noqa: A002
        self._download_model_if_not_exists()
        return [np.array(e, dtype=np.float32) for e in self._forward(list(input))]

    def embed_query(self, input):  # noqa: A002
        return self.__call__(input)
