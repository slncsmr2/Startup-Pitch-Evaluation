from __future__ import annotations

from app.pipeline import StartupPitchPipeline
from app.schemas import PitchInput, PitchVideoInput
from app.services.audio_processor import AudioChunkMetadata, AudioProcessor
from app.services.transcriber import build_local_transcriber
from app.services.video_processor import VideoProcessor


def test_silent_audio_chunk_uses_fallback_text() -> None:
    transcriber = build_local_transcriber(
        backend="faster-whisper",
        model_path="",
        min_audio_quality=0.35,
    )

    silent_metadata = AudioChunkMetadata(
        chunk_id=0,
        start_sec=0,
        end_sec=5,
        duration_sec=5.0,
        sample_rate=16000,
        num_samples=80000,
        mel_shape=(64, 156),
        mel_mean=0.0,
        mel_std=0.0,
        silence_ratio=0.99,
        clipping_ratio=0.0,
        audio_quality_score=0.85,
        audio_hash="silent-001",
        audio_file_path="audio/missing/chunk_0000.wav",
    )

    result = transcriber.transcribe_chunk(
        audio_file_path=silent_metadata.audio_file_path,
        audio_metadata=silent_metadata,
        fallback_text="",
        language_hint="en-ta",
    )

    assert result.status == "fallback-silent"
    assert result.text == "[inaudible chunk]"
    assert 0 <= result.confidence <= 1


def test_short_video_pipeline_still_returns_output() -> None:
    payload = PitchInput(
        title="Short Video",
        transcript="quick pitch for short video case",
        language_hint="en",
        video=PitchVideoInput(
            file_name="short_video.mp4",
            file_format="mp4",
            duration_sec=5,
            transcript_text="",
        ),
    )

    response = StartupPitchPipeline(window_seconds=5).evaluate(payload, request_id="short-video-case")

    assert response.request_id == "short-video-case"
    assert len(response.chunk_reports) == 1
    assert 0 <= response.summary.overall_score <= 10


def test_corrupt_video_name_is_handled_gracefully() -> None:
    payload = PitchInput(
        title="Corrupt File",
        transcript="This pitch should still produce a report even with corrupt file naming.",
        language_hint="en",
        video=PitchVideoInput(
            file_name="corrupt_file$$$.mp4",
            file_format="mp4",
            duration_sec=30,
            transcript_text="",
        ),
    )

    response = StartupPitchPipeline(window_seconds=5).evaluate(payload, request_id="corrupt-file-case")

    assert response.request_id == "corrupt-file-case"
    assert len(response.chunk_reports) > 0
    assert response.summary.investment_band in {"high-potential", "watchlist", "early-risk"}


def test_mixed_english_tamil_detection_returns_ta_en() -> None:
    mixed_english_tamil_text = "This startup helps farmers with crop pricing and தமிழ் சந்தை access."
    payload = PitchInput(
        title="English Tamil Mix",
        transcript=mixed_english_tamil_text,
        language_hint="en-ta",
        video=PitchVideoInput(
            file_name="eng_tamil_mix.mp4",
            file_format="mp4",
            duration_sec=10,
            transcript_text="",
        ),
    )

    response = StartupPitchPipeline(window_seconds=5).evaluate(payload, request_id="eng-ta-case")

    assert response.summary.language_detected == "ta-en"


def test_video_processor_and_audio_processor_metadata_shapes() -> None:
    video_processor = VideoProcessor(frame_extraction_enabled=False)
    audio_processor = AudioProcessor(audio_extraction_enabled=False)

    frame_meta = video_processor.extract_frames_for_chunk(
        video_file_path="demo.mp4",
        video_duration_sec=60,
        chunk_id=1,
        start_sec=5,
        end_sec=10,
    )
    audio_meta = audio_processor.extract_audio_chunk(
        video_file_path="demo.mp4",
        chunk_id=1,
        start_sec=5,
        end_sec=10,
    )

    assert frame_meta.frame_count == 5
    assert frame_meta.extraction_status == "skipped"
    assert audio_meta.sample_rate == 16000
    assert audio_meta.mel_shape[0] == 64
    assert 0 <= audio_meta.audio_quality_score <= 1
