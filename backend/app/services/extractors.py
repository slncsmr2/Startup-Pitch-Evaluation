from models.audio_encoder import AudioEncoder
from models.text_encoder import TextEncoder
from models.visual_encoder import VisualEncoder


class TextFeatureExtractor:
    def __init__(self) -> None:
        self.model = TextEncoder(embedding_dim=24)

    def extract(self, chunk_text: str, language_hint: str, slide_context: str) -> dict:
        return self.model.infer(chunk_text, language_hint, slide_context)


class VisualFeatureExtractor:
    def __init__(self) -> None:
        self.model = VisualEncoder(embedding_dim=24)

    def extract(self, slide_context: str, chunk_id: int, user_stage: str, video_metadata: object | None = None) -> dict:
        return self.model.infer(slide_context, chunk_id, user_stage, video_metadata=video_metadata)


class AudioFeatureExtractor:
    def __init__(self) -> None:
        self.model = AudioEncoder(embedding_dim=24)

    def extract(self, chunk_text: str, audio_metadata: object | None = None) -> dict:
        return self.model.infer(chunk_text, audio_metadata=audio_metadata)
