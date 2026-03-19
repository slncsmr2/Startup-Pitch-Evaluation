class FusionHead:
    """Cross-modal fusion head with adaptive attention from modality energy."""

    @staticmethod
    def _mean(values: list[float]) -> float:
        return sum(values) / max(1, len(values))

    def infer(self, text_embedding: list[float], visual_embedding: list[float], audio_embedding: list[float]) -> dict:
        text_energy = self._mean(text_embedding)
        visual_energy = self._mean(visual_embedding)
        audio_energy = self._mean(audio_embedding)

        total = max(1e-6, text_energy + visual_energy + audio_energy)
        text_w = text_energy / total
        visual_w = visual_energy / total
        audio_w = audio_energy / total

        fused_vector = [
            (text_w * t) + (visual_w * v) + (audio_w * a)
            for t, v, a in zip(text_embedding, visual_embedding, audio_embedding)
        ]

        return {
            "vector": fused_vector,
            "attention": {
                "text": round(text_w, 4),
                "visual": round(visual_w, 4),
                "audio": round(audio_w, 4),
            },
        }
