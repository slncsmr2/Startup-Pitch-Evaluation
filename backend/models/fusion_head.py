import logging
from app.core.config import settings

logger = logging.getLogger(__name__)

class FusionHead:
    """Cross-modal fusion head with adaptive attention from modality energy."""
    def __init__(self, text_dim=384, visual_dim=256, audio_dim=128, common_dim=256):
        self.use_heuristic = settings.use_heuristic_pipeline
        self.text_dim = text_dim
        self.visual_dim = visual_dim
        self.audio_dim = audio_dim
        self.common_dim = common_dim
        
        self._torch = None
        self._attention_layer = None
        self._device = "cpu"
        
        if not self.use_heuristic:
            self._initialize_neural_backend()

    def _initialize_neural_backend(self) -> None:
        try:
            import torch
            import torch.nn as nn
            self._torch = torch
            requested = settings.nn_device.strip().lower()
            use_cuda = torch.cuda.is_available() and requested in {"auto", "cuda", "gpu"}
            self._device = "cuda" if use_cuda else "cpu"
            
            class CrossModalAttentionLayer(nn.Module):
                def __init__(self, text_dim=384, visual_dim=256, audio_dim=128, common_dim=256):
                    super().__init__()
                    self.proj_text = nn.Linear(text_dim, common_dim)
                    self.proj_visual = nn.Linear(visual_dim, common_dim)
                    self.proj_audio = nn.Linear(audio_dim, common_dim)

                    self.attn_text = nn.Linear(common_dim, 1)
                    self.attn_visual = nn.Linear(common_dim, 1)
                    self.attn_audio = nn.Linear(common_dim, 1)

                def forward(self, text_emb, visual_emb, audio_emb):
                    t_proj = self.proj_text(text_emb)
                    v_proj = self.proj_visual(visual_emb)
                    a_proj = self.proj_audio(audio_emb)

                    s_t = self.attn_text(t_proj)
                    s_v = self.attn_visual(v_proj)
                    s_a = self.attn_audio(a_proj)

                    # For torchaudio/torch compat
                    scores = torch.cat([s_t, s_v, s_a], dim=-1)
                    attn_weights = torch.softmax(scores, dim=-1)

                    attn_t, attn_v, attn_a = attn_weights.split(1, dim=-1)

                    fused = (attn_t * t_proj) + (attn_v * v_proj) + (attn_a * a_proj)
                    return fused, attn_weights

            self._attention_layer = CrossModalAttentionLayer(
                self.text_dim, self.visual_dim, self.audio_dim, self.common_dim
            )
            self._attention_layer.to(self._device)
            self._attention_layer.eval()
        except ImportError:
            logger.warning("Neural fusion disabled. Enable by installing torch.")
            self.use_heuristic = True

    @staticmethod
    def _mean(values: list[float]) -> float:
        return sum(values) / max(1, len(values))

    def infer(self, text_embedding: list[float], visual_embedding: list[float], audio_embedding: list[float]) -> dict:
        if self.use_heuristic or len(text_embedding) == 24 or self._torch is None:
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
        
        with self._torch.no_grad():
            t_tensor = self._torch.tensor(text_embedding, dtype=self._torch.float32, device=self._device).unsqueeze(0)
            v_tensor = self._torch.tensor(visual_embedding, dtype=self._torch.float32, device=self._device).unsqueeze(0)
            a_tensor = self._torch.tensor(audio_embedding, dtype=self._torch.float32, device=self._device).unsqueeze(0)

            fused, attn_weights = self._attention_layer(t_tensor, v_tensor, a_tensor)

            return {
                "vector": fused.squeeze(0).tolist(),
                "attention": {
                    "text": round(attn_weights[0, 0].item(), 4),
                    "visual": round(attn_weights[0, 1].item(), 4),
                    "audio": round(attn_weights[0, 2].item(), 4),
                },
            }
