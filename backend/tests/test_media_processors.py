from app.services.audio_processor import AudioProcessor
from app.services.video_processor import VideoProcessor


def test_video_processor_missing_video_returns_safe_metadata() -> None:
    processor = VideoProcessor(frame_extraction_enabled=True)
    metadata = processor.extract_frames_for_chunk(
        video_file_path="does_not_exist.mp4",
        video_duration_sec=10,
        chunk_id=0,
        start_sec=0,
        end_sec=5,
        frames_per_chunk=5,
    )

    assert metadata.extraction_status == "missing-video"
    assert metadata.frame_count == 0
    assert 0.0 <= metadata.face_ratio <= 1.0
    assert 0.0 <= metadata.motion_score <= 1.0
    assert 0.0 <= metadata.eye_contact_score <= 1.0
    assert 0.0 <= metadata.pose_ratio <= 1.0
    assert 0.0 <= metadata.gesture_energy <= 1.0


def test_audio_processor_missing_video_returns_safe_metadata() -> None:
    processor = AudioProcessor(audio_extraction_enabled=True)
    metadata = processor.extract_audio_chunk(
        video_file_path="does_not_exist.mp4",
        chunk_id=0,
        start_sec=0,
        end_sec=5,
    )

    assert metadata.extraction_status == "missing-video"
    assert metadata.audio_quality_score <= 0.25
    assert metadata.num_samples > 0
    assert 0.0 <= metadata.speech_density <= 1.0
    assert 0.0 <= metadata.pitch_variation <= 1.0
    assert 0.0 <= metadata.energy_variation <= 1.0
