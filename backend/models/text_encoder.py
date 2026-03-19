import hashlib
import re


class TextEncoder:
    """Deterministic text encoder that mimics trainable inference behavior."""

    def __init__(self, embedding_dim: int = 24) -> None:
        self.embedding_dim = embedding_dim

    def _hash_to_vector(self, value: str) -> list[float]:
        digest = hashlib.sha256(value.encode("utf-8")).digest()
        values = [b / 255.0 for b in digest]
        if self.embedding_dim <= len(values):
            return values[: self.embedding_dim]
        repeats = (self.embedding_dim // len(values)) + 1
        return (values * repeats)[: self.embedding_dim]

    def _detect_language(self, chunk_text: str, language_hint: str) -> str:
        tamil_chars = len(re.findall(r"[\u0B80-\u0BFF]", chunk_text))
        english_chars = len(re.findall(r"[A-Za-z]", chunk_text))
        hint = language_hint.lower()

        if tamil_chars > 0 and english_chars > 0:
            return "ta-en"
        if tamil_chars > 0:
            return "ta"
        if "ta" in hint and "en" in hint:
            return "ta-en"
        return "en"

    def _normalize(self, chunk_text: str) -> str:
        return " ".join(chunk_text.replace("\n", " ").split()).strip()

    @staticmethod
    def _clamp_10(value: float) -> float:
        return max(0.0, min(10.0, value))

    def infer(self, chunk_text: str, language_hint: str, slide_context: str) -> dict:
        normalized_text = self._normalize(chunk_text)
        words = normalized_text.split()
        unique_ratio = len(set(words)) / max(1, len(words))
        language = self._detect_language(normalized_text, language_hint)
        lowered = normalized_text.lower()

        problem_signal = 2.0 if "problem" in lowered else 0.0
        market_signal = 2.0 if "market" in lowered else 0.0
        traction_signal = 2.0 if any(k in lowered for k in ["pilot", "growth", "revenue"]) else 0.0
        business_model_signal = 2.0 if any(k in lowered for k in ["price", "subscription", "saas", "margin"]) else 0.0
        team_signal = 2.0 if any(k in lowered for k in ["team", "founder", "experience"]) else 0.0
        language_bonus = 1.0 if language == "ta-en" else 0.4

        return {
            "embedding": self._hash_to_vector(f"text::{normalized_text}::{slide_context}"),
            "language_detected": language,
            "problem_clarity": self._clamp_10(4.0 + problem_signal + unique_ratio * 1.5),
            "market_opportunity": self._clamp_10(4.0 + market_signal + unique_ratio * 1.3),
            "solution_uniqueness": self._clamp_10(4.5 + unique_ratio * 3.0),
            "traction_evidence": self._clamp_10(3.5 + traction_signal + min(len(words), 40) * 0.08),
            "business_model_strength": self._clamp_10(3.5 + business_model_signal + min(len(words), 30) * 0.07),
            "team_readiness": self._clamp_10(3.0 + team_signal + language_bonus),
        }
