import hashlib
import logging
from pathlib import Path
from app.core.config import settings


logger = logging.getLogger(__name__)


class AudioEncoder:
    """Audio encoder proxy for pace and prosody signals."""

    def __init__(
        self,
        embedding_dim: int = 24,
        use_heuristic: bool = True,
        sample_rate: int = 16000,
        feature_type: str = "mfcc",
    ) -> None:
        self.embedding_dim = embedding_dim
        self.use_heuristic = use_heuristic
        self.sample_rate = sample_rate
        self.feature_type = feature_type.strip().lower()

        self._torch = None
        self._torchaudio = None
        self._mfcc_transform = None
        self._cnn = None
        self._scoring_mlp = None
        self._device = "cpu"

        if not self.use_heuristic:
            self._initialize_neural_backend()

    def _initialize_neural_backend(self) -> None:
        if self.feature_type != "mfcc":
            logger.warning("Unsupported audio feature type '%s'; defaulting to mfcc", self.feature_type)
            self.feature_type = "mfcc"

        try:
            import torch  # type: ignore
            import torchaudio  # type: ignore
        except Exception as exc:
            logger.warning("Neural audio encoder disabled (torch/torchaudio unavailable): %s", exc)
            return

        self._torch = torch
        self._torchaudio = torchaudio
        requested = settings.nn_device.strip().lower()
        use_cuda = torch.cuda.is_available() and requested in {"auto", "cuda", "gpu"}
        self._device = "cuda" if use_cuda else "cpu"
        self.embedding_dim = 128

        self._mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=self.sample_rate,
            n_mfcc=40,
            melkwargs={
                "n_fft": 400,
                "hop_length": 160,
                "n_mels": 40,
                "center": True,
            },
        )

        self._cnn = torch.nn.Sequential(
            torch.nn.Conv1d(40, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(kernel_size=2),
            torch.nn.Conv1d(64, 128, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool1d(1),
        )

        self._scoring_mlp = torch.nn.Sequential(
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 2),
            torch.nn.Sigmoid(),
        )

        torch.manual_seed(11)
        self._mfcc_transform.to(self._device)
        self._cnn.to(self._device)
        self._scoring_mlp.to(self._device)
        self._cnn.eval()
        self._scoring_mlp.eval()

    def _neural_backend_ready(self) -> bool:
        return all(
            component is not None
            for component in [self._torch, self._torchaudio, self._mfcc_transform, self._cnn, self._scoring_mlp]
        )

    def _load_waveform(self, audio_metadata: object | None):
        if not self._neural_backend_ready():
            return None

        audio_path = str(getattr(audio_metadata, "audio_file_path", "")).strip()
        if not audio_path:
            return None

        path = Path(audio_path)
        if not path.is_file():
            return None

        try:
            waveform, sr = self._torchaudio.load(str(path))
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            if int(sr) != int(self.sample_rate):
                waveform = self._torchaudio.functional.resample(
                    waveform,
                    orig_freq=int(sr),
                    new_freq=self.sample_rate,
                )
            return waveform.to(self._device)
        except Exception as exc:
            logger.warning("Neural audio waveform load failed for %s: %s", audio_path, exc)
            return None

    def _pad_or_trim_frames(self, mfcc):
        frames = int(mfcc.shape[-1])
        target = 128
        if frames == target:
            return mfcc
        if frames > target:
            return mfcc[..., :target]

        pad = target - frames
        return self._torch.nn.functional.pad(mfcc, (0, pad))

    def _neural_infer(self, waveform):
        with self._torch.no_grad():
            mfcc = self._mfcc_transform(waveform)
            mfcc = self._pad_or_trim_frames(mfcc)

            encoded = self._cnn(mfcc).squeeze(-1)
            scores = self._scoring_mlp(encoded).squeeze(0)

            voice_pace = float(scores[0].item() * 10.0)
            prosody = float(scores[1].item() * 10.0)
            embedding = encoded.squeeze(0).detach().cpu().numpy().astype("float32").tolist()

        return {
            "embedding": embedding,
            "voice_pace": self._clamp_10(voice_pace),
            "prosody": self._clamp_10(prosody),
        }

    def _heuristic_infer(
        self,
        chunk_text: str,
        duration_sec: float,
        extraction_status: str,
        silence_ratio: float,
        clipping_ratio: float,
        quality_score: float,
        speech_density: float,
        pitch_variation: float,
        energy_variation: float,
    ) -> dict:
        words = chunk_text.split()
        punctuation_count = sum(1 for c in chunk_text if c in ",;:!?")
        words_per_second = len(words) / max(1.0, duration_sec)
        missing_audio_signal = extraction_status in {"missing-video", "backend-unavailable", "skipped"}

        if missing_audio_signal:
            # Keep ratings neutral when runtime dependencies/media are unavailable.
            pace = max(5.2, 6.2 - abs(2.4 - words_per_second) * 1.3)
            prosody = 5.8 + min(punctuation_count, 8) * 0.2
        else:
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
        duration_sec = float(getattr(audio_metadata, "duration_sec", 5.0))
        extraction_status = str(getattr(audio_metadata, "extraction_status", "skipped"))
        silence_ratio = float(getattr(audio_metadata, "silence_ratio", 0.1))
        clipping_ratio = float(getattr(audio_metadata, "clipping_ratio", 0.0))
        quality_score = float(getattr(audio_metadata, "audio_quality_score", 0.7))
        speech_density = float(getattr(audio_metadata, "speech_density", max(0.0, 1.0 - silence_ratio)))
        pitch_variation = float(getattr(audio_metadata, "pitch_variation", 0.0))
        energy_variation = float(getattr(audio_metadata, "energy_variation", 0.0))

        if self.use_heuristic:
            return self._heuristic_infer(
                chunk_text=chunk_text,
                duration_sec=duration_sec,
                extraction_status=extraction_status,
                silence_ratio=silence_ratio,
                clipping_ratio=clipping_ratio,
                quality_score=quality_score,
                speech_density=speech_density,
                pitch_variation=pitch_variation,
                energy_variation=energy_variation,
            )

        if extraction_status != "success":
            return self._heuristic_infer(
                chunk_text=chunk_text,
                duration_sec=duration_sec,
                extraction_status=extraction_status,
                silence_ratio=silence_ratio,
                clipping_ratio=clipping_ratio,
                quality_score=quality_score,
                speech_density=speech_density,
                pitch_variation=pitch_variation,
                energy_variation=energy_variation,
            )

        waveform = self._load_waveform(audio_metadata)
        if waveform is None:
            return self._heuristic_infer(
                chunk_text=chunk_text,
                duration_sec=duration_sec,
                extraction_status=extraction_status,
                silence_ratio=silence_ratio,
                clipping_ratio=clipping_ratio,
                quality_score=quality_score,
                speech_density=speech_density,
                pitch_variation=pitch_variation,
                energy_variation=energy_variation,
            )

        try:
            return self._neural_infer(waveform)
        except Exception as exc:
            logger.warning("Neural audio inference failed; falling back to heuristics: %s", exc)
            return self._heuristic_infer(
                chunk_text=chunk_text,
                duration_sec=duration_sec,
                extraction_status=extraction_status,
                silence_ratio=silence_ratio,
                clipping_ratio=clipping_ratio,
                quality_score=quality_score,
                speech_density=speech_density,
                pitch_variation=pitch_variation,
                energy_variation=energy_variation,
            )
