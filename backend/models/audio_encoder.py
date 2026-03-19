import hashlib


class AudioEncoder:
    """Audio encoder proxy for pace and prosody signals."""

    def __init__(self, embedding_dim: int = 24) -> None:
        self.embedding_dim = embedding_dim

    def _hash_to_vector(self, value: str) -> list[float]:
        digest = hashlib.sha256(value.encode("utf-8")).digest()
        values = [b / 255.0 for b in digest]
        if self.embedding_dim <= len(values):
            return values[: self.embedding_dim]
        repeats = (self.embedding_dim // len(values)) + 1
        return (values * repeats)[: self.embedding_dim]

    @staticmethod
    def _clamp_10(value: float) -> float:
        return max(0.0, min(10.0, value))

    def infer(self, chunk_text: str) -> dict:
        words = chunk_text.split()
        punctuation_count = sum(1 for c in chunk_text if c in ",;:!?")
        pace = len(words) / 10.0
        prosody = 5.0 + min(punctuation_count, 8) * 0.5

        return {
            "embedding": self._hash_to_vector(f"audio::{chunk_text}"),
            "voice_pace": self._clamp_10(10.0 - abs(4.5 - pace) * 1.5),
            "prosody": self._clamp_10(prosody),
        }
