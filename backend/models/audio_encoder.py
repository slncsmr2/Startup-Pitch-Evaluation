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

    def infer(self, chunk_text: str, audio_metadata: object | None = None) -> dict:
        words = chunk_text.split()
        punctuation_count = sum(1 for c in chunk_text if c in ",;:!?")

        duration_sec = float(getattr(audio_metadata, "duration_sec", 5.0))
        silence_ratio = float(getattr(audio_metadata, "silence_ratio", 0.1))
        clipping_ratio = float(getattr(audio_metadata, "clipping_ratio", 0.0))
        quality_score = float(getattr(audio_metadata, "audio_quality_score", 0.7))
        speech_density = float(getattr(audio_metadata, "speech_density", max(0.0, 1.0 - silence_ratio)))
        pitch_variation = float(getattr(audio_metadata, "pitch_variation", 0.0))
        energy_variation = float(getattr(audio_metadata, "energy_variation", 0.0))

        words_per_second = len(words) / max(1.0, duration_sec)
        pace = (10.0 - abs(2.4 - words_per_second) * 3.0) * (0.55 + speech_density * 0.45)
        prosody = (
            4.0
            + min(punctuation_count, 8) * 0.25
            + quality_score * 1.8
            + pitch_variation * 2.2
            + energy_variation * 1.6
            + speech_density * 1.2
            - (silence_ratio * 1.8 + clipping_ratio * 2.2)
        )

        return {
            "embedding": self._hash_to_vector(
                f"audio::{chunk_text}::dur={duration_sec:.2f}::sil={silence_ratio:.2f}::clip={clipping_ratio:.2f}::sd={speech_density:.2f}::pv={pitch_variation:.2f}::ev={energy_variation:.2f}"
            ),
            "voice_pace": self._clamp_10(pace),
            "prosody": self._clamp_10(prosody),
        }
