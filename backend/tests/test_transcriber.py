from app.core.config import settings
from app.schemas import PitchInput, PitchVideoInput
from app.services.audio_processor import AudioChunkMetadata
from app.services.preprocessing import temporal_synchronize_and_segment
from app.services.transcriber import build_local_transcriber


def test_transcriber_falls_back_when_audio_file_missing() -> None:
    transcriber = build_local_transcriber(
        min_audio_quality=0.35,
        transcriber_backend="openai",
        openai_api_key="",
        openai_model_name="whisper-1",
    )
    metadata = AudioChunkMetadata(
        chunk_id=0,
        start_sec=0,
        end_sec=5,
        duration_sec=5.0,
        sample_rate=16000,
        num_samples=80000,
        mel_shape=(64, 156),
        mel_mean=0.0,
        mel_std=0.0,
        silence_ratio=0.0,
        clipping_ratio=0.0,
        audio_quality_score=1.0,
        audio_hash="abc123",
        audio_file_path="audio/missing/chunk_0000.wav",
    )

    result = transcriber.transcribe_chunk(
        audio_file_path=metadata.audio_file_path,
        audio_metadata=metadata,
        fallback_text="deterministic fallback",
        language_hint="en-ta",
    )

    assert result.text == "deterministic fallback"
    assert result.status == "fallback-no-audio-file"
    assert 0 <= result.confidence <= 1


def test_preprocessing_uses_local_transcriber_with_fallback() -> None:
    previous_flag = settings.use_local_transcriber
    previous_api_key = settings.openai_api_key

    try:
        settings.use_local_transcriber = True
        settings.openai_api_key = ""

        payload = PitchInput(
            title="FallbackTest",
            transcript="hello world this is stable fallback text",
            language_hint="en-ta",
            video=PitchVideoInput(
                file_name="fallback_test.mp4",
                duration_sec=10,
                transcript_text="",
            ),
        )

        chunks = temporal_synchronize_and_segment(payload, window_seconds=5)

        assert len(chunks) == 2
        assert all(chunk.text.strip() for chunk in chunks)
        assert chunks[0].alignment.text_excerpt == chunks[0].text
    finally:
        settings.use_local_transcriber = previous_flag
        settings.openai_api_key = previous_api_key


def test_openai_transcriber_backend_selection_and_fallback() -> None:
    transcriber = build_local_transcriber(
        min_audio_quality=0.35,
        transcriber_backend="openai",
        openai_api_key="",
        openai_model_name="whisper-1",
    )

    metadata = AudioChunkMetadata(
        chunk_id=0,
        start_sec=0,
        end_sec=5,
        duration_sec=5.0,
        sample_rate=16000,
        num_samples=80000,
        mel_shape=(64, 156),
        mel_mean=0.0,
        mel_std=0.0,
        silence_ratio=0.0,
        clipping_ratio=0.0,
        audio_quality_score=0.1,
        audio_hash="abc123",
        audio_file_path="audio/missing/chunk_0000.wav",
    )

    result = transcriber.transcribe_chunk(
        audio_file_path=metadata.audio_file_path,
        audio_metadata=metadata,
        fallback_text="fallback transcript",
        language_hint="en",
    )

    assert transcriber.backend_name == "openai-whisper-api"
    assert result.text == "fallback transcript"
    assert result.status == "fallback-low-quality"


def test_auto_backend_selection_returns_supported_transcriber() -> None:
    transcriber = build_local_transcriber(
        min_audio_quality=0.35,
        transcriber_backend="auto",
        openai_api_key="",
        openai_model_name="whisper-1",
    )

    assert transcriber.backend_name in {"openai-whisper-api", "faster-whisper-local"}
