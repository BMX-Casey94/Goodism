"""Optional local embedding provider (no API calls).

Uses sentence-transformers when installed. Selected by setting
KB_EMBEDDING_MODEL=local:<model-name>, e.g.
  KB_EMBEDDING_MODEL=local:BAAI/bge-small-en-v1.5

Local embeddings keep document content off third-party APIs, which
matters while the corpus is still private.
"""
from __future__ import annotations


class LocalEmbedder:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.model = f"local:{model_name}"  # matches Embedder.model interface
        self._encoder = None

    def _get_encoder(self):
        if self._encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError(
                    "sentence-transformers is required for local embeddings"
                ) from exc
            self._encoder = SentenceTransformer(self.model_name)
        return self._encoder

    def embed(self, texts: list[str]) -> list[list[float]]:
        encoder = self._get_encoder()
        vectors = encoder.encode(texts, normalize_embeddings=True)
        return [list(map(float, v)) for v in vectors]
