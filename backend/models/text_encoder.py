import hashlib
import logging
import re

import numpy as np
from app.core.config import settings


logger = logging.getLogger(__name__)


class TextEncoder:
    """Deterministic text encoder that mimics trainable inference behavior."""

    def __init__(
        self,
        embedding_dim: int = 24,
        use_heuristic: bool = True,
        model_name: str = "all-MiniLM-L6-v2",
        hidden_dim: int = 128,
    ) -> None:
        self.embedding_dim = embedding_dim
        self.use_heuristic = use_heuristic
        self.model_name = model_name
        self.hidden_dim = hidden_dim
        self._nn_model = None
        self._nn_input_dim = 384
        self._nn_device = "cpu"
        self._w1: np.ndarray | None = None
        self._b1: np.ndarray | None = None
        self._w2: np.ndarray | None = None
        self._b2: np.ndarray | None = None

        if not self.use_heuristic:
            self._initialize_neural_backend()

    def _initialize_neural_backend(self) -> None:
        try:
            import torch  # type: ignore

            requested = settings.nn_device.strip().lower()
            use_cuda = torch.cuda.is_available() and requested in {"auto", "cuda", "gpu"}
            self._nn_device = "cuda" if use_cuda else "cpu"
        except Exception:
            self._nn_device = "cpu"

        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except Exception as exc:
            logger.warning(
                "Neural text encoder disabled (sentence-transformers unavailable): %s",
                exc,
            )
            return

        try:
            self._nn_model = SentenceTransformer(self.model_name, device=self._nn_device)
            self._nn_input_dim = int(self._nn_model.get_sentence_embedding_dimension())
            self.embedding_dim = self._nn_input_dim
            self._initialize_mlp_weights()
        except Exception as exc:
            logger.warning(
                "Neural text encoder disabled (model load failed: %s): %s",
                self.model_name,
                exc,
            )
            self._nn_model = None

    def _initialize_mlp_weights(self) -> None:
        rng = np.random.default_rng(7)
        in_dim = self._nn_input_dim
        hid_dim = self.hidden_dim
        out_dim = 6

        self._w1 = rng.normal(0.0, 1.0 / np.sqrt(in_dim), size=(in_dim, hid_dim)).astype(np.float32)
        self._b1 = np.zeros((hid_dim,), dtype=np.float32)
        self._w2 = rng.normal(0.0, 1.0 / np.sqrt(hid_dim), size=(hid_dim, out_dim)).astype(np.float32)

        score_priors = np.array([5.2, 5.1, 5.4, 4.9, 4.8, 4.9], dtype=np.float32) / 10.0
        score_priors = np.clip(score_priors, 1e-4, 1 - 1e-4)
        self._b2 = np.log(score_priors / (1.0 - score_priors)).astype(np.float32)

    def _mlp_predict(self, embedding: list[float]) -> dict[str, float]:
        if any(x is None for x in [self._w1, self._b1, self._w2, self._b2]):
            raise RuntimeError("Neural head is not initialized")

        x = np.array(embedding, dtype=np.float32)
        h = np.maximum((x @ self._w1) + self._b1, 0.0)
        y = (h @ self._w2) + self._b2
        probs = 1.0 / (1.0 + np.exp(-y))
        scaled = np.clip(probs * 10.0, 0.0, 10.0)

        return {
            "problem_clarity": float(scaled[0]),
            "market_opportunity": float(scaled[1]),
            "solution_uniqueness": float(scaled[2]),
            "traction_evidence": float(scaled[3]),
            "business_model_strength": float(scaled[4]),
            "team_readiness": float(scaled[5]),
        }

    def _semantic_embedding(self, normalized_text: str) -> list[float]:
        if self._nn_model is None:
            return self._hash_to_vector(f"text::{normalized_text}")
        text = normalized_text if normalized_text else " "
        encoded = self._nn_model.encode(text, normalize_embeddings=True)
        return np.asarray(encoded, dtype=np.float32).tolist()

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

    def _heuristic_scores(self, normalized_text: str, language: str) -> dict[str, float]:
        words = normalized_text.split()
        unique_ratio = len(set(words)) / max(1, len(words))
        lowered = normalized_text.lower()

        problem_signal = 2.0 if "problem" in lowered else 0.0
        market_signal = 2.0 if "market" in lowered else 0.0
        traction_signal = 2.0 if any(k in lowered for k in ["pilot", "growth", "revenue"]) else 0.0
        business_model_signal = 2.0 if any(k in lowered for k in ["price", "subscription", "saas", "margin"]) else 0.0
        team_signal = 2.0 if any(k in lowered for k in ["team", "founder", "experience"]) else 0.0
        language_bonus = 1.0 if language == "ta-en" else 0.4

        return {
            "problem_clarity": self._clamp_10(4.0 + problem_signal + unique_ratio * 1.5),
            "market_opportunity": self._clamp_10(4.0 + market_signal + unique_ratio * 1.3),
            "solution_uniqueness": self._clamp_10(4.5 + unique_ratio * 3.0),
            "traction_evidence": self._clamp_10(3.5 + traction_signal + min(len(words), 40) * 0.08),
            "business_model_strength": self._clamp_10(3.5 + business_model_signal + min(len(words), 30) * 0.07),
            "team_readiness": self._clamp_10(3.0 + team_signal + language_bonus),
        }

    def infer(self, chunk_text: str, language_hint: str, slide_context: str) -> dict:
        normalized_text = self._normalize(chunk_text)
        language = self._detect_language(normalized_text, language_hint)
        heuristic_targets = self._heuristic_scores(normalized_text, language)

        if self.use_heuristic or self._nn_model is None:
            output_scores = heuristic_targets
            embedding = self._hash_to_vector(f"text::{normalized_text}::{slide_context}")
        else:
            embedding = self._semantic_embedding(normalized_text)
            output_scores = self._mlp_predict(embedding)

        return {
            "embedding": embedding,
            "language_detected": language,
            "problem_clarity": output_scores["problem_clarity"],
            "market_opportunity": output_scores["market_opportunity"],
            "solution_uniqueness": output_scores["solution_uniqueness"],
            "traction_evidence": output_scores["traction_evidence"],
            "business_model_strength": output_scores["business_model_strength"],
            "team_readiness": output_scores["team_readiness"],
            "heuristic_targets": heuristic_targets,
        }
