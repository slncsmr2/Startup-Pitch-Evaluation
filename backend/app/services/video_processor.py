"""
Video processing module for frame extraction and visual feature persistence.

Handles per-5-second chunk frame extraction from video inputs.
Currently supports local video processing; external API calls disabled per design constraints.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class FrameExtractMetadata:
    """Metadata for extracted frames from a chunk."""
    chunk_id: int
    start_sec: int
    end_sec: int
    frame_count: int
    frame_dir: str  # Directory where frames would be persisted
    frame_hash: str  # Deterministic hash for consistency
    extraction_status: str  # "success", "pending", "skipped"


class VideoProcessor:
    """
    Extracts frames from video per 5-second chunk.
    
    Design:
    - Interface-first: real implementation uses ffmpeg or similar
    - Current implementation: stub that generates deterministic metadata
    - Frames persisted to chunk_id-based directories
    """

    def __init__(self, frame_extraction_enabled: bool = False):
        """
        Initialize video processor.
        
        Args:
            frame_extraction_enabled: If False, generates metadata only (for non-ML phases).
                If True (future), actually extracts frames using ffmpeg/cv2.
        """
        self.frame_extraction_enabled = frame_extraction_enabled
        logger.info(f"VideoProcessor initialized | frame_extraction_enabled={frame_extraction_enabled}")

    def extract_frames_for_chunk(
        self,
        video_file_path: str,
        video_duration_sec: int,
        chunk_id: int,
        start_sec: int,
        end_sec: int,
        frames_per_chunk: int = 5,
    ) -> FrameExtractMetadata:
        """
        Extract frames from chunk window.
        
        Args:
            video_file_path: Path to video file
            video_duration_sec: Total video duration
            chunk_id: Chunk identifier
            start_sec: Chunk start time (seconds)
            end_sec: Chunk end time (seconds)
            frames_per_chunk: Number of evenly-spaced frames to extract
        
        Returns:
            FrameExtractMetadata with extraction status and locations
        """
        # Deterministic frame hash based on video properties and chunk timing
        frame_hash = self._compute_frame_hash(video_file_path, chunk_id, start_sec, end_sec)
        
        frame_dir = f"frames/{video_file_path.split('/')[-1].replace('.', '_')}/chunk_{chunk_id:04d}"

        if self.frame_extraction_enabled:
            # Future: actual ffmpeg extraction here
            logger.debug(f"Frame extraction enabled but not yet implemented for chunk {chunk_id}")
            status = "pending"
        else:
            # Stub: just generate metadata for now
            status = "skipped"
            logger.debug(f"Frame extraction skipped for chunk {chunk_id} (not enabled)")

        return FrameExtractMetadata(
            chunk_id=chunk_id,
            start_sec=start_sec,
            end_sec=end_sec,
            frame_count=frames_per_chunk,
            frame_dir=frame_dir,
            frame_hash=frame_hash,
            extraction_status=status,
        )

    def _compute_frame_hash(self, video_file: str, chunk_id: int, start_sec: int, end_sec: int) -> str:
        """Deterministic hash for frame metadata consistency."""
        import hashlib
        data = f"{video_file}|{chunk_id}|{start_sec}|{end_sec}"
        return hashlib.md5(data.encode()).hexdigest()[:12]
