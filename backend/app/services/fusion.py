from models.fusion_head import FusionHead


_fusion_head = FusionHead()


def fuse_modalities(text_features: dict, visual_features: dict, audio_features: dict) -> dict:
    return _fusion_head.infer(
        text_embedding=text_features["embedding"],
        visual_embedding=visual_features["embedding"],
        audio_embedding=audio_features["embedding"],
    )
