def fuse_modalities(text_features: dict, visual_features: dict, audio_features: dict) -> dict:
    """Simple weighted fusion placeholder for cross-attention style aggregation."""
    text_w = 0.45
    visual_w = 0.30
    audio_w = 0.25

    text_vec = text_features["embedding"]
    visual_vec = visual_features["embedding"]
    audio_vec = audio_features["embedding"]

    fused_vector = [
        (text_w * t) + (visual_w * v) + (audio_w * a)
        for t, v, a in zip(text_vec, visual_vec, audio_vec)
    ]

    return {
        "vector": fused_vector,
        "attention": {
            "text": text_w,
            "visual": visual_w,
            "audio": audio_w,
        },
    }
