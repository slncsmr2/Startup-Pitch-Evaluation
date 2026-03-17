"""
Audio processing module for chunk extraction and feature generation.

Handles per-5-second chunk audio extraction and mel-spectrogram feature computation.
Designed for local processing with no external API calls per design constraints.
"""

import logging
from dataclasses import dataclass

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


class AudioProcessor:
    """
    Extracts audio from video per 5-second chunk and computes mel features.
    
    Design:
    - Interface-first: can use librosa/scipy locally or external service
    - Current implementation: stub generating deterministic metadata
    - Mel features persisted to chunk-based directories
    """

    def __init__(self, audio_extraction_enabled: bool = False, sample_rate: int = 16000):
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
        duration_sec = end_sec - start_sec
        num_samples = int(duration_sec * self.sample_rate)
        
        # Mel spectrogram shape: (n_mels, time_bins)
        # time_bins ≈ num_samples / hop_length; hop_length typically 512 for 16kHz
        hop_length = 512
        n_time_bins = max(1, num_samples // hop_length)
        mel_shape = (self.n_mels, n_time_bins)
        
        # Deterministic audio hash
        audio_hash = self._compute_audio_hash(video_file_path, chunk_id, start_sec, end_sec)
        audio_file_path = self._build_audio_chunk_path(video_file_path, chunk_id)
        
        if self.audio_extraction_enabled:
            # Future: actual extraction with librosa/scipy here
            logger.debug(f"Audio extraction enabled but not yet implemented for chunk {chunk_id}")
            # Stub values pending extraction
            mel_mean = 0.0
            mel_std = 0.0
            silence_ratio = 0.0
            clipping_ratio = 0.0
            quality_score = 0.5  # Unknown quality
        else:
            # Stub: generate reasonable default values
            mel_mean = 0.0
            mel_std = 0.0
            silence_ratio = 0.0  # Assume no silence for heuristic mode
            clipping_ratio = 0.0  # Assume no clipping
            quality_score = 1.0  # Assume OK in heuristic mode (will be estimated later)
            logger.debug(f"Audio extraction skipped for chunk {chunk_id} (not enabled)")
        
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
        )

    def _compute_audio_hash(self, video_file: str, chunk_id: int, start_sec: int, end_sec: int) -> str:
        """Deterministic hash for audio metadata consistency."""
        import hashlib
        data = f"{video_file}|{chunk_id}|{start_sec}|{end_sec}|sr={self.sample_rate}"
        return hashlib.md5(data.encode()).hexdigest()[:12]

    def _build_audio_chunk_path(self, video_file: str, chunk_id: int) -> str:
        """Deterministic local path where extracted chunk audio would be stored."""
        safe_video = video_file.split("/")[-1].replace(".", "_")
        return f"audio/{safe_video}/chunk_{chunk_id:04d}.wav"
