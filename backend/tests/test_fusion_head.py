from models.fusion_head import FusionHead


def test_attention_strengths_are_normalized_and_bounded() -> None:
    text_w, visual_w, audio_w = FusionHead._compute_attention_from_strengths(0.1, 0.1, 10.0)

    assert abs((text_w + visual_w + audio_w) - 1.0) < 1e-6
    assert text_w >= 0.15
    assert visual_w >= 0.15
    assert audio_w >= 0.15


def test_deterministic_fallback_fusion_returns_common_dim_vector() -> None:
    head = FusionHead()
    result = head._deterministic_fallback_fusion(
        text_embedding=[0.1] * 384,
        visual_embedding=[0.2] * 256,
        audio_embedding=[0.3] * 128,
    )

    assert len(result["vector"]) == head.common_dim
    assert set(result["attention"].keys()) == {"text", "visual", "audio"}
