"""
Audio processing module for chunk extraction and feature generation.

Handles per-5-second chunk audio extraction and mel-spectrogram feature computation.
Uses local ffmpeg extraction and waveform statistics with deterministic fallback.
"""

import logging
import hashlib
import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class AudioChunkMetadata:
    """Metadata for audio chunk and computed features."""
    chunk_id: int
    start_sec: int
    end_sec: int
    duration_sec: float
    sample_rate: int
    num_samples: int
    # Mel-spectrogram features
    mel_shape: tuple[int, int]  # (n_mels, time_bins)
    mel_mean: float  # Mean energy
    mel_std: float  # Energy variance
    # Confidence and quality
    silence_ratio: float  # Proportion of silence detected
    clipping_ratio: float  # Proportion of audio clipping
    audio_quality_score: float  # 0-1 quality estimate
    audio_hash: str  # Deterministic hash for consistency
    audio_file_path: str  # Deterministic local audio chunk path
    speech_density: float = 0.0  # 0-1 ratio of active speech-like signal
    pitch_variation: float = 0.0  # 0-1 proxy for pitch dynamics
    energy_variation: float = 0.0  # 0-1 proxy for loudness dynamics
    extraction_status: str = "skipped"


class AudioProcessor:
    """
    Extracts audio from video per 5-second chunk and computes mel features.
    
    Design:
    - Extracts mono wav chunks via ffmpeg
    - Computes waveform-based quality features
    - Falls back to deterministic metadata when dependencies/media are unavailable
    """

    def __init__(
        self,
        audio_extraction_enabled: bool = False,
        sample_rate: int = 16000,
        output_root: str = "outputs/audio_chunks",
    ):
        """
        Initialize audio processor.
        
        Args:
            audio_extraction_enabled: If False, generates metadata only.
                If True (future), extracts audio using ffmpeg and computes mel features.
            sample_rate: Audio sample rate (Hz). Default 16kHz (Whisper standard).
        """
        self.audio_extraction_enabled = audio_extraction_enabled
        self.sample_rate = sample_rate
        self.n_mels = 64  # Mel spectrograms: 64 frequency bins
        self.output_root = Path(output_root)
        logger.info(
            f"AudioProcessor initialized | audio_extraction_enabled={audio_extraction_enabled} | "
            f"sample_rate={sample_rate}Hz | n_mels={self.n_mels}"
        )

    def extract_audio_chunk(
        self,
        video_file_path: str,
        chunk_id: int,
        start_sec: int,
        end_sec: int,
    ) -> AudioChunkMetadata:
        """
        Extract audio from chunk window and compute mel features.
        
        Args:
            video_file_path: Path to video file
            chunk_id: Chunk identifier
            start_sec: Chunk start time (seconds)
            end_sec: Chunk end time (seconds)
        
        Returns:
            AudioChunkMetadata with extracted features and quality metrics
        """
        duration_sec = max(1, end_sec - start_sec)
        num_samples = int(duration_sec * self.sample_rate)
        
        # Mel spectrogram shape: (n_mels, time_bins)
        # time_bins ≈ num_samples / hop_length; hop_length typically 512 for 16kHz
        hop_length = 512
        n_time_bins = max(1, num_samples // hop_length)
        mel_shape = (self.n_mels, n_time_bins)
        
        # Deterministic audio hash
        audio_hash = self._compute_audio_hash(video_file_path, chunk_id, start_sec, end_sec)
        audio_file_path = self._build_audio_chunk_path(video_file_path, chunk_id)

        mel_mean = 0.0
        mel_std = 0.0
        silence_ratio = 0.0
        clipping_ratio = 0.0
        quality_score = 1.0
        speech_density = 0.0
        pitch_variation = 0.0
        energy_variation = 0.0
        extraction_status = "skipped"

        if not self.audio_extraction_enabled:
            logger.debug("Audio extraction skipped for chunk %s (not enabled)", chunk_id)
        elif not Path(video_file_path).is_file():
            logger.warning("Audio extraction skipped; video missing: %s", video_file_path)
            quality_score = 0.25
            extraction_status = "missing-video"
        else:
            Path(audio_file_path).parent.mkdir(parents=True, exist_ok=True)
            ok = self._extract_audio_with_ffmpeg(
                video_file_path=video_file_path,
                output_audio_file_path=audio_file_path,
                start_sec=start_sec,
                end_sec=end_sec,
            )
            if not ok:
                quality_score = 0.2
                extraction_status = "backend-unavailable"
            else:
                stats = self._read_waveform_stats(audio_file_path)
                if stats is None:
                    quality_score = 0.2
                    extraction_status = "backend-unavailable"
                else:
                    num_samples = stats["num_samples"]
                    n_time_bins = max(1, num_samples // hop_length)
                    mel_shape = (self.n_mels, n_time_bins)
                    mel_mean = stats["mean_abs"]
                    mel_std = stats["std_abs"]
                    silence_ratio = stats["silence_ratio"]
                    clipping_ratio = stats["clipping_ratio"]
                    speech_density = stats["speech_density"]
                    pitch_variation = stats["pitch_variation"]
                    energy_variation = stats["energy_variation"]
                    quality_score = self._quality_score_from_stats(silence_ratio, clipping_ratio, mel_mean)
                    extraction_status = "success"
        
        return AudioChunkMetadata(
            chunk_id=chunk_id,
            start_sec=start_sec,
            end_sec=end_sec,
            duration_sec=float(duration_sec),
            sample_rate=self.sample_rate,
            num_samples=num_samples,
            mel_shape=mel_shape,
            mel_mean=mel_mean,
            mel_std=mel_std,
            silence_ratio=silence_ratio,
            clipping_ratio=clipping_ratio,
            audio_quality_score=quality_score,
            audio_hash=audio_hash,
            audio_file_path=audio_file_path,
            speech_density=speech_density,
            pitch_variation=pitch_variation,
            energy_variation=energy_variation,
            extraction_status=extraction_status,
        )

    def _compute_audio_hash(self, video_file: str, chunk_id: int, start_sec: int, end_sec: int) -> str:
        """Deterministic hash for audio metadata consistency."""
        data = f"{video_file}|{chunk_id}|{start_sec}|{end_sec}|sr={self.sample_rate}"
        return hashlib.md5(data.encode()).hexdigest()[:12]

    def _build_audio_chunk_path(self, video_file: str, chunk_id: int) -> str:
        """Deterministic local path where extracted chunk audio would be stored."""
        safe_video = Path(video_file).name.replace(".", "_")
        return str(self.output_root / safe_video / f"chunk_{chunk_id:04d}.wav")

    def _extract_audio_with_ffmpeg(
        self,
        video_file_path: str,
        output_audio_file_path: str,
        start_sec: int,
        end_sec: int,
    ) -> bool:
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            str(start_sec),
            "-to",
            str(end_sec),
            "-i",
            video_file_path,
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(self.sample_rate),
            output_audio_file_path,
        ]
        try:
            completed = subprocess.run(cmd, check=False, capture_output=True, text=True)
            if completed.returncode != 0:
                logger.warning("ffmpeg audio extraction failed: %s", completed.stderr.strip())
                return False
            return True
        except Exception as exc:
            logger.warning("ffmpeg unavailable for audio extraction: %s", exc)
            return False

    @staticmethod
    def _read_waveform_stats(audio_file_path: str) -> dict[str, float | int] | None:
        try:
            import numpy as np  # type: ignore

            with wave.open(audio_file_path, "rb") as wav_file:
                frames = wav_file.readframes(wav_file.getnframes())
                if not frames:
                    return None
                samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0

            abs_samples = np.abs(samples)
            silence_ratio = float(np.mean(abs_samples < 0.01))
            clipping_ratio = float(np.mean(abs_samples >= 0.98))

            frame_len = 400
            hop = 160
            if samples.shape[0] < frame_len:
                padded = np.pad(samples, (0, frame_len - samples.shape[0]))
                framed = np.expand_dims(padded, axis=0)
            else:
                frame_count = 1 + (samples.shape[0] - frame_len) // hop
                framed = np.stack([samples[i * hop:i * hop + frame_len] for i in range(frame_count)])

            rms = np.sqrt(np.mean(framed ** 2, axis=1) + 1e-8)
            mean_rms = float(np.mean(rms))
            std_rms = float(np.std(rms))
            energy_variation = max(0.0, min(1.0, std_rms / max(1e-6, mean_rms + 1e-6)))
            speech_density = max(0.0, min(1.0, 1.0 - silence_ratio))

            pitch_values = []
            min_lag = int(16000 / 320)
            max_lag = int(16000 / 70)
            for frame, frame_rms in zip(framed, rms):
                if frame_rms < 0.015:
                    continue
                ac = np.correlate(frame, frame, mode="full")[len(frame) - 1:]
                if ac.shape[0] <= max_lag:
                    continue
                ac[:min_lag] = 0
                lag = int(np.argmax(ac[min_lag:max_lag]) + min_lag)
                if lag <= 0:
                    continue
                freq = 16000.0 / lag
                if 70.0 <= freq <= 320.0:
                    pitch_values.append(freq)

            if len(pitch_values) >= 2:
                pv = np.array(pitch_values, dtype=np.float32)
                pitch_variation = max(0.0, min(1.0, float(np.std(pv) / max(1.0, np.mean(pv)))))
            else:
                pitch_variation = 0.0

            return {
                "num_samples": int(samples.shape[0]),
                "mean_abs": float(np.mean(abs_samples)),
                "std_abs": float(np.std(abs_samples)),
                "silence_ratio": silence_ratio,
                "clipping_ratio": clipping_ratio,
                "speech_density": float(round(speech_density, 4)),
                "pitch_variation": float(round(pitch_variation, 4)),
                "energy_variation": float(round(energy_variation, 4)),
            }
        except Exception as exc:
            logger.warning("Waveform stats unavailable: %s", exc)
            return None

    @staticmethod
    def _quality_score_from_stats(silence_ratio: float, clipping_ratio: float, mean_abs: float) -> float:
        energy_bonus = 0.2 if mean_abs >= 0.02 else 0.0
        raw = 1.0 - (silence_ratio * 0.6 + clipping_ratio * 0.8) + energy_bonus
        return round(max(0.0, min(1.0, raw)), 3)
