import hashlib


class VisualEncoder:
    """Visual encoder proxy for chunk-level delivery signals."""

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

    def infer(self, slide_context: str, chunk_id: int, user_stage: str) -> dict:
        text = slide_context if slide_context else "no-slides"
        density = min(len(text.split()), 120) / 120.0
        stage_bonus = 0.8 if user_stage.lower() in {"seed", "series-a", "series a"} else 0.3

        return {
            "embedding": self._hash_to_vector(f"visual::{chunk_id}::{text}::{user_stage}"),
            "delivery_clarity": self._clamp_10(4.2 + density * 3.8),
            "presenter_confidence": self._clamp_10(4.0 + (chunk_id % 5) * 0.9 + stage_bonus),
        }
