import hashlib
import re


def _hash_to_vector(text: str, dim: int = 16) -> list[float]:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    values = [b / 255.0 for b in digest]
    if dim <= len(values):
        return values[:dim]
    repeats = (dim // len(values)) + 1
    return (values * repeats)[:dim]


def _clamp_10(value: float) -> float:
    return max(0.0, min(10.0, value))


def _detect_language(chunk_text: str, language_hint: str) -> str:
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


def _normalize_en_ta_text(chunk_text: str) -> str:
    compact = " ".join(chunk_text.replace("\n", " ").split())
    return compact.strip()


class TextFeatureExtractor:
    def extract(self, chunk_text: str, language_hint: str, slide_context: str) -> dict:
        normalized_text = _normalize_en_ta_text(chunk_text)
        words = normalized_text.split()
        unique_ratio = len(set(words)) / max(1, len(words))
        language = _detect_language(normalized_text, language_hint)

        problem_signal = 2.0 if "problem" in normalized_text.lower() else 0.0
        market_signal = 2.0 if "market" in normalized_text.lower() else 0.0
        traction_signal = 2.0 if any(k in normalized_text.lower() for k in ["pilot", "growth", "revenue"]) else 0.0
        business_model_signal = 2.0 if any(k in normalized_text.lower() for k in ["price", "subscription", "saas", "margin"]) else 0.0
        team_signal = 2.0 if any(k in normalized_text.lower() for k in ["team", "founder", "experience"]) else 0.0
        language_bonus = 1.0 if language == "ta-en" else 0.4

        return {
            "embedding": _hash_to_vector(f"text::{normalized_text}::{slide_context}", dim=24),
            "language_detected": language,
            "problem_clarity": _clamp_10(4.0 + problem_signal + unique_ratio * 1.5),
            "market_opportunity": _clamp_10(4.0 + market_signal + unique_ratio * 1.3),
            "solution_uniqueness": _clamp_10(4.5 + unique_ratio * 3.0),
            "traction_evidence": _clamp_10(3.5 + traction_signal + min(len(words), 40) * 0.08),
            "business_model_strength": _clamp_10(3.5 + business_model_signal + min(len(words), 30) * 0.07),
            "team_readiness": _clamp_10(3.0 + team_signal + language_bonus),
        }


class VisualFeatureExtractor:
    def extract(self, slide_context: str, chunk_id: int, user_stage: str) -> dict:
        text = slide_context if slide_context else "no-slides"
        vector = _hash_to_vector(f"visual::{chunk_id}::{text}::{user_stage}", dim=24)
        density = min(len(text.split()), 120) / 120.0
        stage_bonus = 0.8 if user_stage.lower() in {"seed", "series-a", "series a"} else 0.3
        return {
            "embedding": vector,
            "delivery_clarity": _clamp_10(4.2 + density * 3.8),
            "presenter_confidence": _clamp_10(4.0 + (chunk_id % 5) * 0.9 + stage_bonus),
        }


class AudioFeatureExtractor:
    def extract(self, chunk_text: str) -> dict:
        words = chunk_text.split()
        punctuation_count = sum(1 for c in chunk_text if c in ",;:!?")
        pace = len(words) / 10.0
        prosody = 5.0 + min(punctuation_count, 8) * 0.5

        return {
            "embedding": _hash_to_vector(f"audio::{chunk_text}", dim=24),
            "voice_pace": _clamp_10(10.0 - abs(4.5 - pace) * 1.5),
            "prosody": _clamp_10(prosody),
        }
