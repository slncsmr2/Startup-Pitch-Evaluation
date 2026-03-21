from types import SimpleNamespace

from models.audio_encoder import AudioEncoder
from models.visual_encoder import VisualEncoder


def test_audio_encoder_uses_neutral_scores_when_audio_unavailable() -> None:
    encoder = AudioEncoder(embedding_dim=8)
    metadata = SimpleNamespace(
        duration_sec=5,
        extraction_status="backend-unavailable",
        silence_ratio=0.0,
        clipping_ratio=0.0,
        audio_quality_score=0.0,
        speech_density=0.0,
        pitch_variation=0.0,
        energy_variation=0.0,
    )

    result = encoder.infer("No transcript available", audio_metadata=metadata)

    assert result["voice_pace"] >= 5.2
    assert result["prosody"] >= 5.8


def test_visual_encoder_uses_neutral_scores_when_visual_unavailable() -> None:
    encoder = VisualEncoder(embedding_dim=8)
    metadata = SimpleNamespace(
        frame_count=0,
        start_sec=0,
        end_sec=5,
        extraction_status="backend-unavailable",
        face_ratio=0.0,
        motion_score=0.0,
        eye_contact_score=0.0,
        pose_ratio=0.0,
        gesture_energy=0.0,
    )

    result = encoder.infer("Problem Market Traction", chunk_id=0, user_stage="seed", video_metadata=metadata)

    assert result["delivery_clarity"] >= 5.0
    assert result["presenter_confidence"] >= 5.0
