import logging
from dataclasses import dataclass

from app.core.config import settings
from app.schemas import PitchInput
from app.services.audio_processor import AudioChunkMetadata, AudioProcessor
from app.services.transcriber import BaseLocalTranscriber, build_local_transcriber
from app.services.video_processor import FrameExtractMetadata, VideoProcessor

logger = logging.getLogger(__name__)


@dataclass
class ChunkAlignmentMetadata:
    """Metadata tracking text/audio/visual alignment for a chunk."""
    video_metadata: FrameExtractMetadata | None
    audio_metadata: AudioChunkMetadata
    text_excerpt: str
    slide_context: str


@dataclass
class PitchChunk:
    chunk_id: int
    start_sec: int
    end_sec: int
    text: str
    audio_chunk_ref: str
    video_chunk_ref: str
    slide_context: str
    # Phase 2: Enhanced metadata for true timeline chunking
    alignment: ChunkAlignmentMetadata


def _resolve_transcript(payload: PitchInput) -> str:
    if payload.transcript.strip():
        return payload.transcript.strip()
    if payload.video and payload.video.transcript_text.strip():
        return payload.video.transcript_text.strip()
    return ""


def _resolve_slide_context(payload: PitchInput) -> str:
    if payload.slides:
        return " ".join(f"{slide.title} {slide.content}".strip() for slide in payload.slides).strip()
    return " ".join(payload.slide_text).strip()


def temporal_synchronize_and_segment(payload: PitchInput, window_seconds: int = 5) -> list[PitchChunk]:
    """
    Synchronizes transcript/video/slide context using true timeline chunking.
    
    Creates aligned chunks based on video duration, with deterministic metadata
    for text/audio/visual alignment across all modalities.
    
    Args:
        payload: PitchInput with video, transcript, and slides
        window_seconds: Chunk duration in seconds (default 5)
    
    Returns:
        List of PitchChunk with full alignment metadata
    """
    transcript = _resolve_transcript(payload)
    slide_context = _resolve_slide_context(payload)
    
    # Get video duration (true timeline source of truth)
    video_duration_sec = payload.video.duration_sec if payload.video else 60
    video_file = payload.video.file_name if payload.video else "unknown.mp4"
    logger.info(f"Temporal segmentation | video={video_file} | duration_sec={video_duration_sec}")
    
    # Initialize processors for audio/video chunk metadata
    video_processor = VideoProcessor(frame_extraction_enabled=False)
    audio_processor = AudioProcessor(audio_extraction_enabled=False)
    transcriber: BaseLocalTranscriber | None = None
    if settings.use_local_transcriber:
        transcriber = build_local_transcriber(
            backend=settings.local_transcriber_backend,
            model_path=settings.local_transcriber_model_path,
            min_audio_quality=settings.transcriber_min_audio_quality,
        )
        logger.info(
            "Local transcriber enabled | backend=%s | model_path=%s",
            settings.local_transcriber_backend,
            settings.local_transcriber_model_path or "<unset>",
        )
    
    # Distribute transcript across timeline chunks
    sentences = [s.strip() for s in transcript.replace("\n", " ").split(".") if s.strip()]
    if not sentences:
        sentences = [transcript if transcript else "No transcript available"]
    
    chunks: list[PitchChunk] = []
    num_chunks = max(1, (video_duration_sec + window_seconds - 1) // window_seconds)
    words = " ".join(sentences).split()
    words_per_chunk = max(1, len(words) // num_chunks) if words else 1
    
    for chunk_idx in range(num_chunks):
        start_sec = chunk_idx * window_seconds
        end_sec = min((chunk_idx + 1) * window_seconds, video_duration_sec)
        
        # Assign text proportionally to timeline
        text_start_word = chunk_idx * words_per_chunk
        text_end_word = min((chunk_idx + 1) * words_per_chunk, len(words))
        chunk_text = " ".join(words[text_start_word:text_end_word]) if text_start_word < len(words) else ""
        if not chunk_text:
            chunk_text = sentences[min(chunk_idx, len(sentences) - 1)].strip()
        
        # Extract video frame metadata
        video_metadata = video_processor.extract_frames_for_chunk(
            video_file_path=video_file,
            video_duration_sec=video_duration_sec,
            chunk_id=chunk_idx,
            start_sec=start_sec,
            end_sec=end_sec,
        )
        
        # Extract audio chunk metadata
        audio_metadata = audio_processor.extract_audio_chunk(
            video_file_path=video_file,
            chunk_id=chunk_idx,
            start_sec=start_sec,
            end_sec=end_sec,
        )
        
        # Build alignment metadata
        alignment = ChunkAlignmentMetadata(
            video_metadata=video_metadata,
            audio_metadata=audio_metadata,
            text_excerpt=chunk_text,
            slide_context=slide_context,
        )

        if transcriber is not None:
            transcription = transcriber.transcribe_chunk(
                audio_file_path=audio_metadata.audio_file_path,
                audio_metadata=audio_metadata,
                fallback_text=chunk_text,
                language_hint=payload.language_hint,
            )
            chunk_text = transcription.text
            alignment.text_excerpt = chunk_text
            logger.debug(
                "Chunk %s transcription | backend=%s | status=%s | confidence=%.2f",
                chunk_idx,
                transcription.backend,
                transcription.status,
                transcription.confidence,
            )
        
        chunks.append(
            PitchChunk(
                chunk_id=chunk_idx,
                start_sec=start_sec,
                end_sec=end_sec,
                text=chunk_text,
                audio_chunk_ref=f"audio_{chunk_idx}",
                video_chunk_ref=f"video_{chunk_idx}",
                slide_context=slide_context,
                alignment=alignment,
            )
        )
    
    logger.info(f"Generated {len(chunks)} aligned chunks | window_seconds={window_seconds}")
    return chunks
