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

    @staticmethod
    def _latin_tokens(text: str) -> list[str]:
        return re.findall(r"[A-Za-z']+", text.lower())

    def _english_evidence(self, text: str) -> float:
        tokens = self._latin_tokens(text)
        if not tokens:
            return 0.0

        common_en = {
            "the", "a", "an", "and", "or", "for", "to", "of", "in", "on", "with", "by", "from",
            "is", "are", "was", "were", "be", "been", "being", "we", "our", "you", "your", "they",
            "this", "that", "these", "those", "it", "as", "at", "will", "can", "have", "has", "had",
            "startup", "market", "problem", "solution", "revenue", "growth", "pilot", "team", "product",
        }
        hits = sum(1 for t in tokens if t in common_en)
        lexical_ratio = hits / max(1, len(tokens))

        vowels = sum(1 for ch in "".join(tokens) if ch in "aeiou")
        vowel_ratio = vowels / max(1, len("".join(tokens)))

        # Weighted lexical+orthographic signal for English-like Latin text.
        return min(1.0, (lexical_ratio * 0.8) + (vowel_ratio * 0.7))

    def _detect_language(self, chunk_text: str, language_hint: str) -> str:
        text = chunk_text.strip()
        hint = language_hint.lower().strip()

        if not text:
            return "en"

        tamil_chars = len(re.findall(r"[\u0B80-\u0BFF]", text))
        english_chars = len(re.findall(r"[A-Za-z]", text))
        letter_total = tamil_chars + english_chars
        tamil_ratio = (tamil_chars / letter_total) if letter_total else 0.0
        english_ratio = (english_chars / letter_total) if letter_total else 0.0
        english_evidence = self._english_evidence(text)

        # Strong script signal wins first.
        if english_chars > 0 and (english_ratio >= 0.8 or (tamil_chars == 0 and english_evidence >= 0.18)):
            return "en"
        if tamil_chars > 0 and tamil_ratio >= 0.8:
            return "ta"

        # Mixed script when both are materially present.
        if tamil_chars > 0 and english_chars > 0 and tamil_ratio >= 0.12 and english_ratio >= 0.12:
            return "ta-en"
        if tamil_chars > 0 and english_chars == 0:
            return "ta"
        if english_chars > 0 and tamil_chars == 0:
            return "en"

        # Try probabilistic language detection for Latin-script or sparse text.
        try:
            from langdetect import detect_langs  # type: ignore

            predictions = detect_langs(text)
            probs = {item.lang: item.prob for item in predictions}
            ta_prob = probs.get("ta", 0.0)
            en_prob = probs.get("en", 0.0)

            if ta_prob >= 0.22 and en_prob >= 0.22:
                return "ta-en"
            if en_prob >= 0.45 and english_evidence >= 0.15:
                return "en"
            if ta_prob >= 0.6:
                return "ta"
            if en_prob >= 0.55:
                return "en"
        except Exception:
            pass

        if english_evidence >= 0.18:
            return "en"

        # Conservative fallback for genuinely ambiguous short text.
        if "ta" in hint:
            return "ta"
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
