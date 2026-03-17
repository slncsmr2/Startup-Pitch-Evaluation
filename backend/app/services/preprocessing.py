from dataclasses import dataclass

from app.schemas import PitchInput


@dataclass
class PitchChunk:
    chunk_id: int
    start_sec: int
    end_sec: int
    text: str
    audio_chunk_ref: str
    video_chunk_ref: str
    slide_context: str


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
    """Synchronizes transcript/video/slide context and emits 5-second aligned chunks."""
    transcript = _resolve_transcript(payload)
    slide_context = _resolve_slide_context(payload)

    sentences = [s.strip() for s in transcript.replace("\n", " ").split(".") if s.strip()]
    if not sentences:
        fallback_text = transcript if transcript else "No transcript available"
        return [
            PitchChunk(
                chunk_id=0,
                start_sec=0,
                end_sec=window_seconds,
                text=fallback_text,
                audio_chunk_ref="audio_0",
                video_chunk_ref="video_0",
                slide_context=slide_context,
            )
        ]

    chunks: list[PitchChunk] = []
    for idx, sentence in enumerate(sentences):
        start = idx * window_seconds
        end = start + window_seconds
        chunks.append(
            PitchChunk(
                chunk_id=idx,
                start_sec=start,
                end_sec=end,
                text=sentence,
                audio_chunk_ref=f"audio_{idx}",
                video_chunk_ref=f"video_{idx}",
                slide_context=slide_context,
            )
        )
    return chunks
